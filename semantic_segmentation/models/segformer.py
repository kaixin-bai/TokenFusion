import torch
import torch.nn as nn
import torch.nn.functional as F
from . import mix_transformer
from mmcv.cnn import ConvModule
from .modules import num_parallel
from torch.nn import GroupNorm


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# class SegFormerHead(nn.Module):
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#     def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=20, **kwargs):
#         super(SegFormerHead, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         assert len(feature_strides) == len(self.in_channels)
#         assert min(feature_strides) == feature_strides[0]
#         self.feature_strides = feature_strides
#
#         c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
#
#         #decoder_params = kwargs['decoder_params']
#         #embedding_dim = decoder_params['embed_dim']
#
#         self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
#         self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
#         self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
#         self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
#         self.dropout = nn.Dropout2d(0.1)
#
#         self.linear_fuse = ConvModule(
#             in_channels=embedding_dim*4,
#             out_channels=embedding_dim,
#             kernel_size=1,
#             norm_cfg=dict(type='BN', requires_grad=True)
#         )
#
#         self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
#
#     def forward(self, x):
#         c1, c2, c3, c4 = x
#
#         ############## MLP decoder on C1-C4 ###########
#         n, _, h, w = c4.shape
#
#         _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
#
#         _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
#
#         _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
#         _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
#
#         _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
#
#         _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
#
#         x = self.dropout(_c)
#         x = self.linear_pred(x)
#
#         return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction_ratio, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ModifiedSegFormerHead(nn.Module):
    def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=20, **kwargs):
        super(ModifiedSegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.ca_c1 = ChannelAttention(2 * c1_in_channels)  # 2 times due to concatenation
        self.ca_c2 = ChannelAttention(2 * c2_in_channels)
        self.ca_c3 = ChannelAttention(2 * c3_in_channels)
        self.ca_c4 = ChannelAttention(2 * c4_in_channels)

        self.gn_c1 = GroupNorm(32, c1_in_channels)
        self.gn_c2 = GroupNorm(32, c2_in_channels)
        self.gn_c3 = GroupNorm(32, c3_in_channels)
        self.gn_c4 = GroupNorm(32, c4_in_channels)

        # 1x1 convolution to reduce channel dimensions after concatenation
        self.conv_c1 = nn.Conv2d(2 * c1_in_channels, c1_in_channels, 1)
        self.conv_c2 = nn.Conv2d(2 * c2_in_channels, c2_in_channels, 1)
        self.conv_c3 = nn.Conv2d(2 * c3_in_channels, c3_in_channels, 1)
        self.conv_c4 = nn.Conv2d(2 * c4_in_channels, c4_in_channels, 1)

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, s_features, m_features):
        s_c1, s_c2, s_c3, s_c4 = s_features
        m_c1, m_c2, m_c3, m_c4 = m_features

        # Concatenate the features and apply channel attention
        concat_c1 = torch.cat([s_c1, m_c1], dim=1)
        concat_c2 = torch.cat([s_c2, m_c2], dim=1)
        concat_c3 = torch.cat([s_c3, m_c3], dim=1)
        concat_c4 = torch.cat([s_c4, m_c4], dim=1)

        fused_c1 = self.gn_c1(self.conv_c1(self.ca_c1(concat_c1)))
        fused_c2 = self.gn_c2(self.conv_c2(self.ca_c2(concat_c2)))
        fused_c3 = self.gn_c3(self.conv_c3(self.ca_c3(concat_c3)))
        fused_c4 = self.gn_c4(self.conv_c4(self.ca_c4(concat_c4)))

        # Rest of the decoder part remains the same
        n, _, h, w = fused_c4.shape

        _c4 = self.linear_c4(fused_c4).permute(0,2,1).reshape(n, -1, fused_c4.shape[2], fused_c4.shape[3])
        _c4 = F.interpolate(_c4, size=fused_c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(fused_c3).permute(0,2,1).reshape(n, -1, fused_c3.shape[2], fused_c3.shape[3])
        _c3 = F.interpolate(_c3, size=fused_c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(fused_c2).permute(0,2,1).reshape(n, -1, fused_c2.shape[2], fused_c2.shape[3])
        _c2 = F.interpolate(_c2, size=fused_c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(fused_c1).permute(0,2,1).reshape(n, -1, fused_c1.shape[2], fused_c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class WeTr(nn.Module):
    def __init__(self, backbone: str, num_classes=20, embedding_dim=256, pretrained=True):
        super().__init__()
        self.num_classes = num_classes  # 40
        self.embedding_dim = embedding_dim  # 256
        self.feature_strides = [4, 8, 16, 32]
        self.num_parallel = num_parallel  # 2
        # self.in_channels = [32, 64, 160, 256]
        # self.in_channels = [64, 128, 320, 512]

        # 先定义encoder的结构
        self.encoder = getattr(mix_transformer,
                               backbone)()  # getattr用于获取对象的属性或方法，根据backbone的str来读取mix_transformer中叫做backbone的类的属性
        self.in_channels = self.encoder.embed_dims  # [64, 128, 320, 512]
        ## initilize encoder
        # 根据定义的encoder的结构来读取对应的segformer的pretrain model
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            state_dict = expand_state_dict(self.encoder.state_dict(), state_dict, self.num_parallel)
            self.encoder.load_state_dict(state_dict, strict=True)
        print('debug self.in_channels: ', self.in_channels)
        self.decoder = ModifiedSegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                             embedding_dim=self.embedding_dim, num_classes=self.num_classes)

        self.alpha = nn.Parameter(torch.ones(self.num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)

    def get_param_groups(self):
        param_groups = [[], [], []]
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)
        return param_groups

    def forward(self, x):
        """
        这个类其实就是一个segformer，不同的是它把encoder和decoder分开写，然后把encoder里每个block中的经过softmax的score保存在masks中，
        然后decoder依旧是传入x，但是masks就直接传出。
        """
        x, masks = self.encoder(x)  # 输入的x: {list:2} 每个元组都是一个Tensor [1,3,486,625]，分别是rgb和depth，depth是三通道，直接扩充的
        # x = [self.decoder(x[0]), self.decoder(x[1])] # x为{list:2}, 每个元素是一个{list:4}，其中是4个Tensor，shape分别是[1,64,117,157],[1,128,59,79],[1,320,30,40],[1,512,15,20]
        x = [self.decoder(s_features=x[2], m_features=x[0]), self.decoder(s_features=x[2], m_features=x[1])]
        ens = 0
        alpha_soft = F.softmax(self.alpha)
        for l in range(self.num_parallel):
            ens += alpha_soft[l] * x[l].detach()
        x.append(ens)  # 输入ens:[1,40,117,157]；输出x:{list3},每个Tensor[1,40,117,157]
        return x, masks  # masks是个{list:8}，每个又是个{list:2}，然后这8个里面的shape是[1,18369],[1,18369],[1,4661],[1,4661],[1,1200],[1,1200],[1,300],[1,300]


def expand_state_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace('module.', '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            ln = '.ln_%d' % i
            replace = True if ln in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(ln, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict


if __name__ == "__main__":
    # import torch.distributed as dist
    # dist.init_process_group('gloo', init_method='file:///temp/somefile', rank=0, world_size=1)
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeTr('mit_b1', num_classes=20, embedding_dim=256, pretrained=True).cuda()
    wetr.get_param_groupsv()
    dummy_input = torch.rand(2, 3, 512, 512).cuda()
    wetr(dummy_input)
