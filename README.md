# Multimodal Token Fusion for Vision Transformers

By Yikai Wang, Xinghao Chen, Lele Cao, Wenbing Huang, Fuchun Sun, Yunhe Wang.

[**[Paper]**](https://arxiv.org/pdf/2204.08721.pdf)

This repository is a PyTorch implementation of "Multimodal Token Fusion for Vision Transformers", in CVPR 2022. 

<div align="center">
   <img src="./figs/framework.png" width="960">
</div>

Homogeneous predictions,
<div align="center">
   <img src="./figs/homogeneous.png" width="720">
</div>

Heterogeneous predictions,
<div align="center">
   <img src="./figs/heterogeneous.png" width="720">
</div>


## Datasets

For semantic segmentation task on NYUDv2 ([official dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)), we provide a link to download the dataset [here](https://drive.google.com/drive/folders/1mXmOXVsd5l9-gYHk92Wpn6AcKAbE0m3X?usp=sharing). The provided dataset is originally preprocessed in this [repository](https://github.com/DrSleep/light-weight-refinenet), and we add depth data in it.

For image-to-image translation task, we use the sample dataset of [Taskonomy](http://taskonomy.stanford.edu/), where a link to download the sample dataset is [here](https://github.com/alexsax/taskonomy-sample-model-1.git).

Please modify the data paths in the codes, where we add comments 'Modify data path'.


## Dependencies
```
python==3.6
pytorch==1.7.1
torchvision==0.8.2
numpy==1.19.2
```


## Semantic Segmentation


First, 
```
cd semantic_segmentation
```

Download the [segformer](https://github.com/NVlabs/SegFormer) pretrained model (pretrained on ImageNet) from [weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia), e.g., mit_b3.pth. Move this pretrained model to folder 'pretrained'.

Training script for segmentation with RGB and Depth input,
```
python main.py --backbone mit_b3 -c exp_name --lamda 1e-6 --gpu 0 1 2
```

Evaluation script,
```
python main.py --gpu 0 --resume path_to_pth --evaluate  # optionally use --save-img to visualize results
```

Checkpoint models, training logs, mask ratios and the **single-scale** performance on NYUDv2 are provided as follows:

| Method | Backbone | Pixel Acc. (%) | Mean Acc. (%) | Mean IoU (%) | Download | 
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|[CEN](https://github.com/yikaiw/CEN)| ResNet101 | 76.2 | 62.8 | 51.1 | [Google Drive](https://drive.google.com/drive/folders/1wim_cBG-HW0bdipwA1UbnGeDwjldPIwV?usp=sharing)|
|[CEN](https://github.com/yikaiw/CEN)| ResNet152 | 77.0 | 64.4 | 51.6 | [Google Drive](https://drive.google.com/drive/folders/1DGF6vHLDgBgLrdUNJOLYdoXCuEKbIuRs?usp=sharing)|
|Ours| SegFormer-B3 | 78.7 | 67.5 | 54.8 | [Google Drive](https://drive.google.com/drive/folders/14fi8aABFYqGF7LYKHkiJazHA58OBW1AW?usp=sharing)|


Mindspore implementation is available at: https://gitee.com/mindspore/models/tree/master/research/cv/TokenFusion

## Image-to-Image Translation

First, 
```
cd image2image_translation
```
Training script, from Shade and Texture to RGB,
```
python main.py --gpu 0 -c exp_name
```
This script will auto-evaluate on the validation dataset every 5 training epochs. 

Predicted images will be automatically saved during training, in the following folder structure:

```
code_root/ckpt/exp_name/results
  ├── input0  # 1st modality input
  ├── input1  # 2nd modality input
  ├── fake0   # 1st branch output 
  ├── fake1   # 2nd branch output
  ├── fake2   # ensemble output
  ├── best    # current best output
  │    ├── fake0
  │    ├── fake1
  │    └── fake2
  └── real    # ground truth output
```

Checkpoint models:

| Method | Task | FID | KID | Download | 
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| [CEN](https://github.com/yikaiw/CEN) |Texture+Shade->RGB | 62.6 | 1.65 | - |
| Ours | Texture+Shade->RGB | 45.5 | 1.00 | [Google Drive](https://drive.google.com/drive/folders/1vkcDv5bHKXZKxCg4dC7R56ts6nLLt6lh?usp=sharing)|

## 3D Object Detection (under construction)

Data preparation, environments, and training scripts follow [Group-Free](https://github.com/zeliu98/Group-Free-3D) and [ImVoteNet](https://github.com/facebookresearch/imvotenet).

E.g.,
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 2229 --nproc_per_node 4 train_dist.py --max_epoch 600 --val_freq 25 --save_freq 25 --lr_decay_epochs 420 480 540 --num_point 20000 --num_decoder_layers 6 --size_cls_agnostic --size_delta 0.0625 --heading_delta 0.04 --center_delta 0.1111111111111 --weight_decay 0.00000001 --query_points_generator_loss_coef 0.2 --obj_loss_coef 0.4 --dataset sunrgbd --data_root . --use_img --log_dir log/exp_name
```

## Citation

If you find our work useful for your research, please consider citing the following paper.
```
@inproceedings{wang2022tokenfusion,
  title={Multimodal Token Fusion for Vision Transformers},
  author={Wang, Yikai and Chen, Xinghao and Cao, Lele and Huang, Wenbing and Sun, Fuchun and Wang, Yunhe},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```


# 笔记
在网址中下载[数据集](https://drive.google.com/drive/folders/1mXmOXVsd5l9-gYHk92Wpn6AcKAbE0m3X)，并将数据集放在路径: `/data/net/dl_data/ProjectDatasets_bkx/NYUDv2`中

SegFormer预训练模型请从[此repo](https://github.com/NVlabs/SegFormer)提供的[此链接](https://connecthkuhk-my.sharepoint.com/personal/xieenze_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxieenze%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fsegformer%2Fpretrained%5Fmodels&ga=1)下载，并将下载好的预训练模型解压到`TokenFusion/semantic_segmentation/pretrained/`中。

预训练模型(backbone:SegFormer-B3)下载：[下载地址](https://drive.google.com/drive/folders/14fi8aABFYqGF7LYKHkiJazHA58OBW1AW)，下载`model-best.pth.tar`到`TokenFusion/semantic_segmentation/pretrained_models/`文件夹中。


环境配置：
```bash
conda create -n tokenfusion python=3.7
conda activate tokenfusion
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

pip3 install opencv-python matplotlib
pip3 install timm
pip3 install tensorboard tensorboardX
```

安装mmcv，请遵循[此网址](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)安装
```bash
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'
>> 1.12.1
>> 11.3
pip3 install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```

验证：  \
注意：原始repo中验证没有给定backbone，代码里是mit_b1，会导致推理错误！推理的时候一定要给定backbone。
```bash
python3 main.py --gpu 0 --backbone mit_b3 --resume ./pretrained/mit_b3.pth --evaluate
```

训练：
```bash
#python main.py --backbone mit_b3 -c exp_name --lamda 1e-6 --gpu 0 1 2
python main.py --backbone mit_b3 -c exp_name --lamda 1e-6 --gpu 0
# 启动tensorboard可视化
tensorboard --logdir=logs
```

其他：  \
[论文笔记——Segformer: 一种基于Transformer的语义分割方法](https://zhuanlan.zhihu.com/p/441975127)