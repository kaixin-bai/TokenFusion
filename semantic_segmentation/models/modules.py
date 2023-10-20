import torch.nn as nn
import torch

num_parallel = 2

class TokenFuse(nn.Module):
    def __init__(self, dim, temperature=0.5, min_val=0.1, max_val=0.9):
        super(TokenFuse, self).__init__()
        # 初始化一个空间注意力层，用于深度模态
        self.spatial_attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.temperature = temperature
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x0, x1 = x[0], x[1]
        # 使用x1 (即深度模态)计算空间注意力权重
        logits = self.spatial_attn(x1)  # [B, N, 1]
        # 用温度缩放调整logits
        scaled_logits = logits / self.temperature
        # 使用sigmoid获得attention权重，并限制范围
        attention_weights = torch.sigmoid(scaled_logits)
        attention_weights = torch.clamp(attention_weights, self.min_val, self.max_val)
        # 使用attention_weights进行融合
        x_fusion = attention_weights * x1 + (1 - attention_weights) * x0
        return [x_fusion, x1]  # 注意这里，x1没有被更新


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
