# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .modules import ModuleParallel, LayerNormParallel, num_parallel, TokenExchange, TokenArgmax, TokenFuse


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ModuleParallel(nn.Linear(in_features, hidden_features))
        """
        这个DWConv就是segformer的block的mix-ffn公式里的Conv3x3
        "Mix-FFN将3×3卷积和一个MLP混合到每个FFN中。在实验中证明3×3卷积足以为Transformer提供位置信息。使用深度卷积来减少参数数量并提高效率"
        """
        self.dwconv = DWConv(hidden_features)
        self.act = ModuleParallel(act_layer())
        self.fc2 = ModuleParallel(nn.Linear(hidden_features, out_features))
        self.drop = ModuleParallel(nn.Dropout(drop))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = [self.dwconv(x[0], H, W), self.dwconv(x[1], H, W)]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # return x

        # SCRATCHING VISUAL TRANSFORMER’S BACK WITH UNIFORM ATTENTION
        # https://arxiv.org/pdf/2210.08457.pdf
        # return (x + x.mean(dim=1, keepdim=True)) * 0.5
        return [(tensor + tensor.mean(dim=1, keepdim=True)) * 0.5 for tensor in x]



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = ModuleParallel(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = ModuleParallel(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = ModuleParallel(nn.Dropout(attn_drop))
        self.proj = ModuleParallel(nn.Linear(dim, dim))
        self.proj_drop = ModuleParallel(nn.Dropout(proj_drop))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = ModuleParallel(nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio))
            self.norm = LayerNormParallel(dim)
        self.exchange = TokenFuse(dim)
        # self.exchange = TokenExchange()
        # self.exchange = TokenArgmax()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x[0].shape
        q = self.q(x)
        q = [q_.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) for q_ in q]

        if self.sr_ratio > 1:
            x = [x_.permute(0, 2, 1).reshape(B, C, H, W) for x_ in x]
            x = self.sr(x)
            x = [x_.reshape(B, C, -1).permute(0, 2, 1) for x_ in x]
            x = self.norm(x)
            kv = self.kv(x)
            kv = [kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for kv_ in kv]
        else:
            kv = self.kv(x)
            kv = [kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for kv_ in kv]
        k, v = [kv[0][0], kv[1][0]], [kv[0][1], kv[1][1]]

        attn = [(q_ @ k_.transpose(-2, -1)) * self.scale for (q_, k_) in zip(q, k)]
        attn = [attn_.softmax(dim=-1) for attn_ in attn]
        attn = self.attn_drop(attn)

        x = [(attn_ @ v_).transpose(1, 2).reshape(B, N, C) for (attn_, v_) in zip(attn, v)]
        x = self.proj(x)
        x = self.proj_drop(x)

        # if mask is not None:
        #     x = [x_ * mask_.unsqueeze(2) for (x_, mask_) in zip(x, mask)]
        #     x = self.exchange(x, mask, mask_threshold=0.02)
        x = self.exchange(x)

        return x


class PredictorLG(nn.Module):
    """ Image to Patch Embedding from DydamicVit
    """

    def __init__(self, embed_dim=384):
        super().__init__()
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        ) for _ in range(num_parallel)])

    def forward(self, x):
        x = [self.score_nets[i](x[i]) for i in range(num_parallel)]
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNormParallel, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.score = PredictorLG(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        """
        Dropout(丢弃法)：
        1.在每个训练迭代中，每个神经元都有一定的概率被丢弃（不参与计算），概率由dropout的参数控制
        2.通常应用于全连接层或卷积层的输出，目的是防止模型过度依赖特定的神经元，从而提高模型的泛化能力
        -------------------------
        DropPath(随机深度):
        1.在训练时随机丢弃整个模块（例如整个层或一组连续的层）的技术。在每个训练迭代中，每个模块都有一定的概率被丢弃，概率由DropPath的参数控制
        2.通常应用于深度神经网络的层级结构，目的是防止模型对于某些特定层级过度依赖，增强模型的鲁棒性
        -------------------------
        二者只在训练阶段使用，在推理（测试）阶段不会应用任何随机丢弃
        都是用于正则化神经网络，帮助防止过拟合
        """
        self.drop_path = ModuleParallel(DropPath(drop_path)) if drop_path > 0. else ModuleParallel(nn.Identity())
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        """
        self.mlp是segformer的block中mix-ffn的部分
        MixFFN是SegFormer模型中的一种前馈网络结构，用于进行非线性特征转换
        """
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.exchange = TokenExchange()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # if m.bias is not None:
            #     m.bias.data.zero_()
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask=None):
        mask = None
        B = x[0].shape[0]
        # norm1 = self.norm1(x)
        # score = self.score(norm1)
        # mask = [F.gumbel_softmax(score_.reshape(B, -1, 2), hard=True)[:, :, 0] for score_ in score]
        # if mask is not None:
        #     norm = [norm_ * mask_.unsqueeze(2) for (norm_, mask_) in zip(norm, mask)]
        f = self.drop_path(self.attn(self.norm1(x), H, W))
        x = [x_ + f_ for (x_, f_) in zip(x, f)]
        f = self.drop_path(self.mlp(self.norm2(x), H, W))
        x = [x_ + f_ for (x_, f_) in zip(x, f)]
        # if mask is not None:
        #     x = self.exchange(x, mask, mask_threshold=0.02)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        """
        这行代码创建了一个名为self.proj的属性，并将其初始化为一个ModuleParallel实例。这个实例使用nn.Conv2d来执行对输入图像的卷积操作，从而
        将输入图像中的每个小块（patch）转换为一个高维的向量（embedding）。这个操作被称为"Patch Embedding"，常用于处理图像数据。
        """
        self.proj = ModuleParallel(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                             padding=(patch_size[0] // 2, patch_size[1] // 2)))
        self.norm = LayerNormParallel(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)  # {list:2}中的每个tensor从[batch_size,3,500,500]变为[batch_size,64,125,125]
        _, _, H, W = x[0].shape
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]
        x = self.norm(x)
        return x, H, W


"""
segformer的实现
------------------------------------------------------------------------------------------------------------------------
self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4：这些是SegFormer中的Patch Embedding模块，用于将输入
图像切分成不同大小的图像块（patch），并将每个图像块转换为高维向量（embedding）。这些Patch Embedding模块将图像按不同比例缩小，以便多尺度的分割任务。
------------------------------------------------------------------------------------------------------------------------
self.block1, self.block2, self.block3, self.block4：这些是SegFormer中的Transformer编码器模块，称为“Block”。
每个Block由多个Transformer层组成，用于对输入的图像块进行特征编码。depths参数指定了每个Block中的Transformer层数。
------------------------------------------------------------------------------------------------------------------------
self.score_predictor：这是用于预测每个图像块的分类得分的模块，它对应每个Block，即self.block1、self.block2等。在SegFormer中，对于每个Block，使用了独立的预测模块来产生图像块的得分。
------------------------------------------------------------------------------------------------------------------------
def forward_features(self, x):：这个方法执行SegFormer的特征提取部分，将输入的图像经过Patch Embedding和Transformer编码器模块，得到多尺度的特征表示。
------------------------------------------------------------------------------------------------------------------------
def forward(self, x):：这个方法将调用forward_features方法来进行特征提取，注意力掩码被存储在masks列表中，用于后续的处理。
"""


# class MixVisionTransformer(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
#                  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0., norm_layer=LayerNormParallel,
#                  depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
#         super().__init__()
#         self.num_classes = num_classes
#         self.depths = depths  # [3,4,18,3], 3+4+18+3=28, depth是segformer的4个block中结构的重复次数
#         self.embed_dims = embed_dims
#
#         """
#         这里和传统的segformer稍微不同的是，patch_embed使用OverlapPatchEmbed,会对输入进行proj操作(对传入列表内的每个部分都进行flatten后LayerNormParallel操作)
#         如果传入是list为1就和传统的segformer一样
#         """
#         # patch_embed
#         self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
#                                               embed_dim=embed_dims[0])
#         self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
#                                               embed_dim=embed_dims[1])
#         self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
#                                               embed_dim=embed_dims[2])
#         self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
#                                               embed_dim=embed_dims[3])
#
#         predictor_list = [PredictorLG(embed_dims[i]) for i in range(len(depths))]
#         self.score_predictor = nn.ModuleList(predictor_list)
#
#         # transformer encoder
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#         cur = 0
#         self.block1 = nn.ModuleList([Block(
#             dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[0])
#             for i in range(depths[0])])
#         self.norm1 = norm_layer(embed_dims[0])
#
#         cur += depths[0]
#         self.block2 = nn.ModuleList([Block(
#             dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[1])
#             for i in range(depths[1])])
#         self.norm2 = norm_layer(embed_dims[1])
#
#         cur += depths[1]
#         self.block3 = nn.ModuleList([Block(
#             dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[2])
#             for i in range(depths[2])])
#         self.norm3 = norm_layer(embed_dims[2])
#
#         cur += depths[2]
#         self.block4 = nn.ModuleList([Block(
#             dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[3])
#             for i in range(depths[3])])
#         self.norm4 = norm_layer(embed_dims[3])
#
#         # classification head
#         # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     '''
#     def init_weights(self, pretrained=None):
#         if isinstance(pretrained, str):
#             logger = get_root_logger()
#             load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
#     '''
#
#     def reset_drop_path(self, drop_path_rate):
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
#         cur = 0
#         for i in range(self.depths[0]):
#             self.block1[i].drop_path.drop_prob = dpr[cur + i]
#
#         cur += self.depths[0]
#         for i in range(self.depths[1]):
#             self.block2[i].drop_path.drop_prob = dpr[cur + i]
#
#         cur += self.depths[1]
#         for i in range(self.depths[2]):
#             self.block3[i].drop_path.drop_prob = dpr[cur + i]
#
#         cur += self.depths[2]
#         for i in range(self.depths[3]):
#             self.block4[i].drop_path.drop_prob = dpr[cur + i]
#
#     def freeze_patch_emb(self):
#         self.patch_embed1.requires_grad = False
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better
#
#     def get_classifier(self):
#         return self.head
#
#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#
#     def forward_features(self, x):
#         B = x[0].shape[0]  # batch size
#         outs0, outs1 = [], []
#         masks = []
#
#         # stage 1
#         x, H, W = self.patch_embed1(x)
#         for i, blk in enumerate(self.block1):
#             score = self.score_predictor[0](x)
#             mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]  # mask_: [B, N]
#             masks.append(mask)
#             x = blk(x, H, W, mask)
#         x = self.norm1(x)
#         x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
#         outs0.append(x[0])
#         outs1.append(x[1])
#
#         # stage 2
#         x, H, W = self.patch_embed2(x)
#         for i, blk in enumerate(self.block2):
#             score = self.score_predictor[1](x)
#             mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]  # mask_: [B, N]
#             masks.append(mask)
#             x = blk(x, H, W, mask)
#         x = self.norm2(x)
#         x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
#         outs0.append(x[0])
#         outs1.append(x[1])
#
#         # stage 3
#         x, H, W = self.patch_embed3(x)
#         for i, blk in enumerate(self.block3):
#             score = self.score_predictor[2](x)
#             mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]  # mask_: [B, N]
#             masks.append(mask)
#             x = blk(x, H, W, mask)
#         x = self.norm3(x)
#         x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
#         outs0.append(x[0])
#         outs1.append(x[1])
#
#         # stage 4
#         x, H, W = self.patch_embed4(x)
#         for i, blk in enumerate(self.block4):
#             score = self.score_predictor[3](x)
#             mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]  # mask_: [B, N]
#             masks.append(mask)
#             x = blk(x, H, W, mask)
#         x = self.norm4(x)
#         x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
#         outs0.append(x[0])
#         outs1.append(x[1])
#
#         return [outs0, outs1], masks
#
#     def forward(self, x):
#         x, masks = self.forward_features(x)
#         return x, masks  # x:list4中的tensor:[1,64,117,157],[1,128,59,79],[1,320,30,40],[1,512,15,20]

class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=LayerNormParallel,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths  # [3,4,18,3], 3+4+18+3=28, depth是segformer的4个block中结构的重复次数
        self.embed_dims = embed_dims

        """
        这里和传统的segformer稍微不同的是，patch_embed使用OverlapPatchEmbed,会对输入进行proj操作(对传入列表内的每个部分都进行flatten后LayerNormParallel操作)
        如果传入是list为1就和传统的segformer一样
        """
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        predictor_list = [PredictorLG(embed_dims[i]) for i in range(len(depths))]
        self.score_predictor = nn.ModuleList(predictor_list)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])  # depths = [3,4,18,3]
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    '''
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
    '''

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x[0].shape[0]  # batch size
        outs0, outs1, outs_single = [], [], []
        masks = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            score = self.score_predictor[0](x)
            mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]  # mask_: [B, N]
            masks.append(mask)
            x = blk(x, H, W, mask)
            x_single = blk(x, H, W, None)
        x, x_single = self.norm1(x), self.norm1(x_single)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        x_single = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x_single]
        outs0.append(x[0])
        outs1.append(x[1])
        outs_single.append(x_single[0])

        # stage 2
        x, H, W = self.patch_embed2(x)
        x_single, _, _ = self.patch_embed2(x_single)
        for i, blk in enumerate(self.block2):
            score = self.score_predictor[1](x)
            mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]  # mask_: [B, N]
            masks.append(mask)
            x = blk(x, H, W, mask)
            x_single = blk(x_single, H, W, None)
        x, x_single = self.norm2(x), self.norm2(x_single)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        x_single = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x_single]
        outs0.append(x[0])
        outs1.append(x[1])
        outs_single.append(x_single[0])

        # stage 3
        x, H, W = self.patch_embed3(x)
        x_single, _, _ = self.patch_embed3(x_single)
        for i, blk in enumerate(self.block3):
            score = self.score_predictor[2](x)
            mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]  # mask_: [B, N]
            masks.append(mask)
            x = blk(x, H, W, mask)
            x_single = blk(x_single, H, W, None)
        x, x_single = self.norm3(x), self.norm3(x_single)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        x_single = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x_single]
        outs0.append(x[0])
        outs1.append(x[1])
        outs_single.append(x_single[0])

        # stage 4
        x, H, W = self.patch_embed4(x)
        x_single, _, _ = self.patch_embed4(x_single)
        for i, blk in enumerate(self.block4):
            score = self.score_predictor[3](x)
            mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]  # mask_: [B, N]
            masks.append(mask)
            x = blk(x, H, W, mask)
            x_single = blk(x_single, H, W, None)
        x, x_single = self.norm4(x), self.norm4(x_single)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        x_single = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x_single]
        outs0.append(x[0])
        outs1.append(x[1])
        outs_single.append(x_single[0])
        return [outs0, outs1, outs_single], masks

    def forward(self, x):
        x, masks = self.forward_features(x)
        return x, masks  # x:list4中的tensor:[1,64,117,157],[1,128,59,79],[1,320,30,40],[1,512,15,20]


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
