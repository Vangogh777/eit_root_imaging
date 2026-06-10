"""
频谱感知卷积层 (Spectral Convolution)
=======================================
为多频率 EIT 数据设计的专用卷积层。
每频率独立编码后融合，保留频谱信息。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    一维频谱感知卷积

    输入: (B, n_freq, n_meas)
    输出: (B, hidden_dim, n_meas')

    对每个频率独立做 1D 卷积 → 然后用 1x1 卷积融合频率信息。
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1,
                 n_frequencies: int = 6, dropout: float = 0.1):
        super().__init__()
        self.n_freq = n_frequencies

        # 每个频率共享权重的 1D 卷积
        self.conv1d = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                padding=padding)
        self.bn1d = nn.BatchNorm1d(out_channels)

        # 频率融合：1x1 卷积跨频率维度
        self.freq_fusion = nn.Conv1d(n_frequencies, n_frequencies, kernel_size=1)

        # 门控机制：频率选择门
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, n_freq, n_meas)

        返回:
            out: (B, n_freq, out_channels)  经过下采样后的输出
        """
        B, F, M = x.shape

        # 1. 每个频率独立 1D 卷积
        # reshape: (B*F, 1, M) → conv → (B*F, C, M')
        x_reshaped = x.view(B * F, 1, M)
        h = self.conv1d(x_reshaped)       # (B*F, C, M')
        h = self.bn1d(h)
        h = self.activation(h)

        # 2. 全局池化到每个频率一个特征向量
        h_pool = h.mean(dim=-1)           # (B*F, C)
        h_pool = h_pool.view(B, F, -1)   # (B, F, C)

        # 3. 频率融合
        h_fused = h_pool.transpose(1, 2)  # (B, C, F)
        h_fused = self.freq_fusion(h_fused)  # (B, C, F)
        h_fused = h_fused.transpose(1, 2)  # (B, F, C)

        # 4. 频率门控
        gate = self.gate(h_pool.mean(dim=1, keepdim=True))  # (B, 1, C)
        out = h_fused * gate

        return self.dropout(out)


class SpectralConv2d(nn.Module):
    """
    二维频谱感知卷积（用于网络中间层）
    输入: (B, C, H, W) 其中 C = n_freq 或 hidden_dim
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1,
                 n_frequencies: int = 6):
        super().__init__()

        self.freq_conv = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.freq_pool = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 并行频率注意力
        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.freq_conv(x)
        h = self.bn(h)
        h = self.activation(h)

        attn = self.freq_attn(h).unsqueeze(-1).unsqueeze(-1)
        out = h * attn

        return out
