"""
注意力门控模块 (Attention Gate)
用于 SF-SBLC 中基础层校正和多频融合的软门控选择。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    注意力门控模块
    从编码器特征中学习软权重，选择性增强/抑制不同空间位置的特征。

    输入:
        g: (B, C_g, N) 门控信号（来自较深层）
        x: (B, C_x, N) 跳跃连接（来自较浅层）

    输出:
        out: (B, C_out, N) 注意力加权的特征
    """

    def __init__(self, in_channels_g: int, in_channels_x: int,
                 out_channels: int, reduction: int = 2):
        super().__init__()

        self.conv_g = nn.Conv1d(in_channels_g, out_channels, kernel_size=1)
        self.conv_x = nn.Conv1d(in_channels_x, out_channels, kernel_size=1)
        self.conv_psi = nn.Conv1d(out_channels, 1, kernel_size=1)

        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # 1. 对齐通道
        g1 = self.conv_g(g)          # (B, C_out, N)
        x1 = self.conv_x(x)          # (B, C_out, N)

        # 2. 融合 + 门控
        psi = self.activation(g1 + x1)
        psi = self.conv_psi(psi)     # (B, 1, N)
        attn = torch.sigmoid(psi)    # (B, 1, N) 软注意力权重

        # 3. 应用到跳跃连接
        out = x1 * attn
        return out


class ChannelAttention(nn.Module):
    """
    通道注意力（SE-style）
    用于多频率融合时选择重要频率通道
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, C, N)
        返回:
            (B, C, N) 通道加权的输出
        """
        b, c, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class SpatialAttention(nn.Module):
    """
    空间注意力
    用于在单元级别选择重要空间位置
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size,
                               padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, C, N)
        返回:
            (B, C, N) 空间加权的输出
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(concat))
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    通道注意力 + 空间注意力串联
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
