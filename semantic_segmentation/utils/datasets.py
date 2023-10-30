import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import skimage.transform as skt
import scipy.signal as ssg
from copy import deepcopy


def imresize(image, size, interp="nearest"):
    """Wrapper over `skimage.transform.resize` to mimic `scipy.misc.imresize`.
    Since `scipy.misc.imresize` has been removed in version 1.3.*, instead use
    `skimage.transform.resize`. The "lanczos" and "cubic" interpolation methods
    are not supported by `skimage.transform.resize`, however there is now
    "biquadratic", "biquartic", and "biquintic".
    Parameters
    ----------
    image : :obj:`numpy.ndarray`
        The image to resize.
    size : int, float, or tuple
        * int   - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image.
    interp : :obj:`str`, optional
        Interpolation to use for re-sizing ("neartest", "bilinear",
        "biquadratic", "bicubic", "biquartic", "biquintic"). Default is
        "nearest".
    Returns
    -------
    :obj:`np.ndarray`
        The resized image.
    """
    skt_interp_map = {"nearest": 0, "bilinear": 1, "biquadratic": 2,
                      "bicubic": 3, "biquartic": 4, "biquintic": 5}
    if interp in ("lanczos", "cubic"):
        raise ValueError("\"lanczos\" and \"cubic\""
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation \"{}\" not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size \"{}\".".format(type(size)))

    return skt.resize(image.astype(float),
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")


def inpaint(data, rescale_factor=1.0):
    """ Fills in the zero pixels in the image.
    Parameters
    ----------
    rescale_factor : float
        amount to rescale the image for inpainting, smaller numbers increase speed
    Returns
    -------
    :obj:`DepthImage`
        depth image with zero pixels filled in
    """
    # get original shape
    orig_shape = (data.shape[0], data.shape[1])

    # form inpaint kernel
    inpaint_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # resize the image
    resized_data = imresize(image=data, size=rescale_factor, interp='nearest')

    # inpaint the smaller image
    cur_data = deepcopy(resized_data)
    zeros = (cur_data == 0)
    while np.any(zeros):
        neighbors = ssg.convolve2d((cur_data != 0), inpaint_kernel,
                                   mode='same', boundary='symm')
        avg_depth = ssg.convolve2d(cur_data, inpaint_kernel,
                                   mode='same', boundary='symm')
        avg_depth[neighbors > 0] = avg_depth[neighbors > 0] / \
                                   neighbors[neighbors > 0]
        avg_depth[neighbors == 0] = 0
        avg_depth[resized_data > 0] = resized_data[resized_data > 0]
        cur_data = avg_depth
        zeros = (cur_data == 0)

    # fill in zero pixels with inpainted and resized image
    inpainted_im = deepcopy(cur_data)
    filled_data = imresize(image=inpainted_im, size=orig_shape, interp="bilinear")
    new_data = np.copy(data)
    new_data[data == 0] = filled_data[data == 0]
    return new_data

def compute_normals(depth_image):
    # 计算x和y方向的梯度
    gx, gy = np.gradient(depth_image)
    # 假设z方向的单位值。这是一个任意选择，但通常设置为1
    gz = np.ones_like(depth_image)
    # 使用梯度计算法向量
    normals = np.stack((-gx * 5.0, -gy * 5.0, gz), axis=-1)
    # 对法向量进行归一化
    norms = np.sqrt(np.sum(normals ** 2, axis=-1, keepdims=True))
    normals /= norms
    # 处理归一化中的NaN值（可能出现在depth_image中有0值的位置）
    normals[np.isnan(normals)] = 0
    return normals

def compute_angles_from_normals(normals):
    eps = 1e-6
    normals = np.clip(normals, -1 + eps, 1 - eps)
    # 计算方位角 (azimuth angle)
    azimuth = np.arctan2(normals[:, :, 1], normals[:, :, 0])
    # 计算仰角 (elevation angle)
    elevation = np.arcsin(normals[:, :, 2])
    return azimuth, elevation

def encode_to_single_channel(azimuth, elevation):
    # 映射角度到[0, 1]的范围
    azimuth_normalized = (azimuth + np.pi) / (2 * np.pi)
    elevation_normalized = (elevation + np.pi / 2) / np.pi
    # 编码为单通道值
    encoded_value = azimuth_normalized + 2 * elevation_normalized
    # 归一化为[0,1]范围
    encoded_value = (encoded_value - encoded_value.min()) / (encoded_value.max() - encoded_value.min())
    return encoded_value


def line_to_paths_fn_nyudv2(x, input_names):
    return x.decode('utf-8').strip('\n').split('\t')


line_to_paths_fn = {'nyudv2': line_to_paths_fn_nyudv2}


class SegDataset(Dataset):
    """Multi-Modality Segmentation dataset.

    Works with any datasets that contain image
    and any number of 2D-annotations.

    Args:
        data_file (string): Path to the data file with annotations.
        data_dir (string): Directory with all the images.
        line_to_paths_fn (callable): function to convert a line of data_file
            into paths (img_relpath, msk_relpath, ...).
        masks_names (list of strings): keys for each annotation mask
                                        (e.g., 'segm', 'depth').
        transform_trn (callable, optional): Optional transform
            to be applied on a sample during the training stage.
        transform_val (callable, optional): Optional transform
            to be applied on a sample during the validation stage.
        stage (str): initial stage of dataset - either 'train' or 'val'.

    """

    def __init__(self, dataset, data_file, data_dir, input_names, input_mask_idxs,
                 transform_trn=None, transform_val=None, stage='train', ignore_label=None):
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = [line_to_paths_fn[dataset](l, input_names) for l in datalist]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = stage
        self.input_names = input_names
        self.input_mask_idxs = input_mask_idxs
        self.ignore_label = ignore_label

    def set_stage(self, stage):
        """Define which set of transformation to use.

        Args:
            stage (str): either 'train' or 'val'

        """
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        idxs = self.input_mask_idxs
        names = [os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]]
        sample = {}
        for i, key in enumerate(self.input_names):
            sample[key] = self.read_image(names[idxs[i]], key)
        try:
            mask = np.array(Image.open(names[idxs[-1]]))
        except FileNotFoundError:  # for sunrgbd
            path = names[idxs[-1]]
            num_idx = int(path[-10:-4]) + 5050
            path = path[:-10] + '%06d' % num_idx + path[-4:]
            mask = np.array(Image.open(path))
        assert len(mask.shape) == 2, 'Masks must be encoded without colourmap'
        sample['inputs'] = self.input_names
        sample['mask'] = mask
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)
        del sample['inputs']
        return sample

    @staticmethod
    def read_image_(x, key):
        img = cv2.imread(x)
        if key == 'depth':
            img = cv2.applyColorMap(cv2.convertScaleAbs(255 - img, alpha=1), cv2.COLORMAP_JET)
        return img

    @staticmethod
    def read_image(x, key):
        """Simple image reader

        Args:
            x (str): path to image.

        Returns image as `np.array`.

        """
        img_arr = np.array(Image.open(x))
        if len(img_arr.shape) == 2:  # grayscale
            img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
        return img_arr

    # @staticmethod
    # def read_image(x, key):
    #     def __rescale_array(arr, min_val, max_val):
    #         return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * (max_val - min_val) + min_val
    #
    #     """Simple image reader
    #
    #     Args:
    #         x (str): path to image.
    #
    #     Returns image as `np.array`.
    #
    #     """
    #     img_arr = np.array(Image.open(x))
    #     if len(img_arr.shape) == 2:  # grayscale
    #         img_arr = inpaint(img_arr, rescale_factor=0.5)  # ****
    #         img_arr_scale = img_arr * 0.001
    #         normals = compute_normals(img_arr)
    #         azimuth, elevation = compute_angles_from_normals(normals)
    #         encoded_image = encode_to_single_channel(azimuth, elevation) # ****
    #         gx, gy = np.gradient(img_arr.astype(np.float32))
    #         gradients = np.zeros([gx.shape[0], gx.shape[1], 2])
    #         gradients[:, :, 0] = gx
    #         gradients[:, :, 1] = gy
    #         gradient_mags = np.linalg.norm(gradients, axis=2)
    #         # img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    #         min_val = np.min(img_arr_scale)
    #         max_val = np.max(img_arr_scale)
    #         encoded_image_rescaled = __rescale_array(encoded_image, min_val, max_val)
    #         gradient_mags_rescaled = __rescale_array(gradient_mags, min_val, max_val)
    #         img_arr = np.stack([img_arr_scale, encoded_image_rescaled, gradient_mags_rescaled], axis=-1)
    #     return img_arr
