"""
简化版 EIT 重建模型 - 针对收敛问题优化
========================================
使用线性投影 + 残差修正的架构，更容易训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearEITModel(nn.Module):
    """
    线性映射 + 残差修正

    思路：
    1. 线性层直接映射 208 → 2824（类似传统反演）
    2. 残差网络精调
    """

    def __init__(self,
                 input_dim: int = 208,
                 n_elems: int = 2824,
                 hidden_dim: int = 512):
        super().__init__()

        self.n_elems = n_elems
        self.sigma_min = 0.005
        self.sigma_max = 0.1

        # 1. 线性投影（直接映射）
        self.linear_proj = nn.Linear(input_dim, n_elems)

        # 2. 残差精调网络
        self.refine = nn.Sequential(
            nn.Linear(n_elems, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_elems),
        )

        # 初始化线性层为输出均值
        self._init_weights()

    def _init_weights(self):
        # 初始化线性层输出均值
        nn.init.constant_(self.linear_proj.bias, 0.02)  # 接近背景值
        nn.init.normal_(self.linear_proj.weight, 0, 0.01)

    def forward(self, voltages: torch.Tensor) -> dict:
        """
        参数:
            voltages: (B, 1, input_dim) 或 (B, input_dim)

        返回:
            dict: {'sigma': (B, n_elems)}
        """
        if voltages.dim() == 3:
            voltages = voltages.squeeze(1)

        # 线性投影
        linear_out = self.linear_proj(voltages)  # (B, n_elems)

        # 残差精调
        residual = self.refine(linear_out)  # (B, n_elems)

        # 组合
        sigma_raw = linear_out + residual

        # 限制范围
        sigma = torch.clamp(sigma_raw, self.sigma_min, self.sigma_max)

        return {'sigma': sigma}


class DeepEITModel(nn.Module):
    """
    深度模型 - 更强的表达能力
    """

    def __init__(self,
                 input_dim: int = 208,
                 n_elems: int = 2824,
                 hidden_dim: int = 1024,
                 n_layers: int = 8):
        super().__init__()

        self.n_elems = n_elems
        self.sigma_min = 0.005
        self.sigma_max = 0.1

        # 输入编码
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]

        # 深度残差块
        for _ in range(n_layers):
            layers.append(ResidualBlock(hidden_dim))

        self.encoder = nn.Sequential(*layers)

        # 输出解码 - 两阶段
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, n_elems),
        )

    def forward(self, voltages: torch.Tensor) -> dict:
        if voltages.dim() == 3:
            voltages = voltages.squeeze(1)

        h = self.encoder(voltages)
        sigma_raw = self.decoder(h)
        sigma = torch.clamp(sigma_raw, self.sigma_min, self.sigma_max)

        return {'sigma': sigma}


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


class PerceptualLoss(nn.Module):
    """
    感知损失 - 关注结构相似性
    """

    def __init__(self, n_elems: int = 2824):
        super().__init__()
        # 简单的局部均值池化
        self.pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算局部结构差异
        """
        # 局部均值
        pred_local = self.pool(pred.unsqueeze(1)).squeeze(1)
        target_local = self.pool(target.unsqueeze(1)).squeeze(1)

        # 局部结构损失
        return F.mse_loss(pred_local, target_local)


if __name__ == "__main__":
    # 测试
    B, n_meas, n_elems = 4, 208, 2824
    voltages = torch.randn(B, 1, n_meas)

    print("测试 LinearEITModel...")
    model = LinearEITModel(n_meas, n_elems)
    out = model(voltages)
    print(f"  输出: {out['sigma'].shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n测试 DeepEITModel...")
    model2 = DeepEITModel(n_meas, n_elems)
    out2 = model2(voltages)
    print(f"  输出: {out2['sigma'].shape}")
    print(f"  参数量: {sum(p.numel() for p in model2.parameters()):,}")

    print("\n✅ 测试通过")
