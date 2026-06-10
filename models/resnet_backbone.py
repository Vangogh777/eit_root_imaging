"""
深度残差骨干网络 (ResNet Backbone)
====================================
用于从编码特征到电导率图的核心重建骨干。
提供多种残差架构选择，可根据精度/速度需求切换。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    基础残差块
    MLP 版本的残差连接
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.net(x)))


class ConvResidualBlock(nn.Module):
    """
    1D 卷积残差块
    适用于一维序列数据的残差连接（如单元序列）
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out + residual)
        return out


class ResNetBackbone(nn.Module):
    """
    深度残差骨干网络（MLP 版本）
    从编码器的隐特征解码到电导率分布

    支持三种模式:
        - "mlp": 全连接残差块（默认，速度快）
        - "conv1d": 1D卷积残差块（适合单元序列）
        - "hybrid": MLP + 卷积混合
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512,
                 n_elems: int = 1500, n_blocks: int = 8,
                 mode: str = "mlp", dropout: float = 0.1):
        super().__init__()
        self.mode = mode

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 残差块堆叠
        blocks = []
        if mode == "mlp":
            for _ in range(n_blocks):
                blocks.append(ResidualBlock(hidden_dim, dropout))
        elif mode == "conv1d":
            # 升维到 1D 卷积所需维度
            blocks.append(nn.Unflatten(-1, (hidden_dim, 1)))
            for _ in range(n_blocks):
                blocks.append(ConvResidualBlock(hidden_dim, dropout=dropout))
            blocks.append(nn.Flatten(-2, -1))
        elif mode == "hybrid":
            for i in range(n_blocks):
                if i % 2 == 0:
                    blocks.append(ResidualBlock(hidden_dim, dropout))
                else:
                    blocks.append(nn.Sequential(
                        nn.Unflatten(-1, (hidden_dim, 1)),
                        ConvResidualBlock(hidden_dim, dropout=dropout),
                        nn.Flatten(-2, -1),
                    ))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.blocks = nn.ModuleList(blocks)

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, n_elems),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        参数:
            z: (B, input_dim) 编码器输出的隐特征

        返回:
            sigma: (B, n_elems) 预测电导率
        """
        h = self.input_proj(z)

        for block in self.blocks:
            h = block(h)

        sigma = self.output_head(h)
        return sigma


class SimplifiedResNet(nn.Module):
    """
    简化残差网络（轻量级快速版本）
    用于原型验证
    """

    def __init__(self, input_dim: int = 512, n_elems: int = 1500):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            ResidualBlock(256, dropout=0.1),
            ResidualBlock(256, dropout=0.1),
            ResidualBlock(256, dropout=0.1),
            nn.Linear(256, n_elems),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
