import torch.nn as nn
import torch.nn.functional as F
import torch

num_parallel = 2

# class TokenFuse(nn.Module):
#     def __init__(self, dim, temperature=0.2):
#         super(TokenFuse, self).__init__()
#         # 初始化一个空间注意力层，用于深度模态
#         self.spatial_attn = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.ReLU(),
#             nn.Linear(dim, 1)
#         )
#         self.temperature = temperature
#         # self.min_val = min_val
#         # self.max_val = max_val
#
#     def forward(self, x):
#         x0, x1 = x[0], x[1]
#         # 使用x1 (即深度模态)计算空间注意力权重
#         logits = self.spatial_attn(x1)  # [B, N, 1]
#         # 用温度缩放调整logits
#         scaled_logits = logits / self.temperature
#         # 使用sigmoid获得attention权重，并限制范围
#         attention_weights = torch.sigmoid(scaled_logits)
#         # attention_weights = torch.clamp(attention_weights, self.min_val, self.max_val)
#         # 使用attention_weights进行融合
#         x_fusion = attention_weights * x1 + (1 - attention_weights) * x0
#
#         # # ======
#         # import matplotlib.pyplot as plt
#         # # 创建一个简单的函数，将输入数据进行可视化
#         # def visualize_tensor(tensor, title):
#         #     tensor_np = tensor.cpu().detach().numpy()
#         #     plt.imshow(tensor_np, cmap='jet')
#         #     plt.colorbar()
#         #     plt.title(title)
#         #     plt.show()
#         # # 创建一个简单的函数，将输入数据进行可视化
#         # def visualize_channels(tensor, title_prefix):
#         #     B, N, C = tensor.shape
#         #     tensor_np = tensor.cpu().detach().numpy().reshape(B, H, W, C)
#         #     # 创建一个绘图网格
#         #     fig, axs = plt.subplots(8, 8, figsize=(15, 15))
#         #     for i in range(8):
#         #         for j in range(8):
#         #             ax = axs[i][j]
#         #             if i * 8 + j < C:
#         #                 ax.imshow(tensor_np[0, :, :, i * 8 + j], cmap='jet')
#         #                 ax.set_title(f'{title_prefix} - Channel {i * 8 + j}')
#         #                 ax.axis('off')
#         #     plt.tight_layout()
#         #     plt.show()
#         # print("debug attention_weights.shape[1]: ", attention_weights.shape[1])
#         # if attention_weights.shape[1] == 18369:
#         #     H, W = 117, 157
#         #     # 将attention_weights调整为图像的形状并可视化
#         #     attention_weights_reshaped = attention_weights.reshape(H, W)
#         #     visualize_tensor(attention_weights_reshaped, "Attention Weights")
#         #     # 对x0的每个通道进行可视化
#         #     visualize_channels(x0, "x0")
#         #     # 对x1的每个通道进行可视化
#         #     visualize_channels(x1, "x1")
#         # # ======
#
#         return [x0, x_fusion]  # 注意这里，x1没有被更新


class TokenFuse(nn.Module):
    def __init__(self, dim):
        super(TokenFuse, self).__init__()
        self.dim = dim
        self.num_heads = 8
        self.head_dim = dim // self.num_heads
        assert self.head_dim * self.num_heads == dim, "dim must be divisible by num_heads"

        # 定义线性层用于计算Q, K, V
        self.qkv_rgb = nn.Linear(dim, dim * 3)  # 用于RGB特征
        self.qkv_depth = nn.Linear(dim, dim * 2)  # 用于深度特征（只计算K和V）
        self.fc_out = nn.Linear(dim, dim)

        # 自适应权重机制
        self.adaptive_weights = nn.Parameter(torch.ones(2))

    def forward(self, x):
        x_rgb, x_depth = x

        B, N, C = x_rgb.size()

        # 处理RGB特征
        qkv_rgb = self.qkv_rgb(x_rgb).reshape(B, N, 3, self.num_heads, self.head_dim)
        q_rgb, k_rgb, v_rgb = qkv_rgb[:, :, 0, :, :], qkv_rgb[:, :, 1, :, :], qkv_rgb[:, :, 2, :, :]

        # 处理深度特征（只计算K和V）
        kv_depth = self.qkv_depth(x_depth).reshape(B, N, 2, self.num_heads, self.head_dim)
        k_depth, v_depth = kv_depth[:, :, 0, :, :], kv_depth[:, :, 1, :, :]

        # 交叉注意力
        attn_scores_rgb = torch.einsum("bnqd,bnkd->bnqk", q_rgb, k_depth) / self.head_dim ** 0.5
        attn_scores_depth = torch.einsum("bnqd,bnkd->bnqk", q_rgb, k_rgb) / self.head_dim ** 0.5

        attn_weights_rgb = F.softmax(attn_scores_rgb, dim=-1)
        attn_weights_depth = F.softmax(attn_scores_depth, dim=-1)

        # 计算加权的值
        weighted_values_rgb = torch.einsum("bnqk,bnvd->bnqd", attn_weights_rgb, v_depth)
        weighted_values_depth = torch.einsum("bnqk,bnvd->bnqd", attn_weights_depth, v_rgb)

        # 自适应权重融合
        adaptive_rgb_weight, adaptive_depth_weight = torch.sigmoid(self.adaptive_weights)
        x_fusion = adaptive_rgb_weight * weighted_values_rgb + adaptive_depth_weight * weighted_values_depth
        x_fusion = x_fusion.reshape(B, N, C)

        # 输出转换
        x_fusion = self.fc_out(x_fusion)

        return [x_rgb, x_fusion]



class TokenExchange(nn.Module):
    """
    作者提供的tokenFusion方法：对x通过mask进行处理，把mask<0.02的部分用另一个模态的x替换
    即在一个模态上未被激活的部分可能在另一个模态上会被激活，即有更强的特征，将这部分换掉来让两个模态互通
    """
    def __init__(self):
        super(TokenExchange, self).__init__()

    def forward(self, x, mask, mask_threshold):
        # x: [B, N, C], mask: [B, N, 1]
        x0, x1 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x0[mask[0] >= mask_threshold] = x[0][mask[0] >= mask_threshold]
        x0[mask[0] < mask_threshold] = x[1][mask[0] < mask_threshold]
        x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
        x1[mask[1] < mask_threshold] = x[0][mask[1] < mask_threshold]
        return [x0, x1]


class TokenArgmax(nn.Module):
    """
    我们的一种尝试，不再使用mask_threshold,而是直接取两个模态里被激活的最大部分，最后返回的[new_x, new_x]两个一样，这两个会在后续的Block中被残差
    Epoch 299  (ens)     glob_acc=76.46    mean_acc=63.89    IoU=50.67
    测试之后发现训练的收敛变快，但是验证集上效果不如tokenfusion
    """
    def __init__(self):
        super(TokenArgmax, self).__init__()

    def forward(self, x, mask, mask_threshold):
        # x: [B, N, C], mask: [B, N, 1]
        new_x = torch.zeros_like(x[0])
        new_x[mask[0] >= mask[1]] = x[0][mask[0] >= mask[1]]
        new_x[mask[1] > mask[0]] = x[1][mask[1] > mask[0]]
        return [new_x, new_x]


class ModuleParallel(nn.Module):
    """
    ModuleParallel类允许我们将一个PyTorch模块应用于多个输入张量并行进行处理。在前向传播时，我们可以将多个输入张量传递给这个模块，并且模块将会
    并行地处理每个输入，并返回一个包含处理结果的列表。
    """

    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'ln_' + str(i), nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]
