"""Voltage encoders for residual EIT models."""

import torch
import torch.nn as nn

from models.conv_spatial_eit import ConvEncoder


class VoltageGlobalEncoder(nn.Module):
    """
    Encode boundary voltages into a global conditioning vector.

    Simplified for single-frequency input: (batch, 1, 208)
    No multi-frequency fusion.
    """

    def __init__(
        self,
        n_frequencies: int = 1,  # 改为单频
        n_meas: int = 208,
        hidden_dim: int = 256,
        base_ch: int = 96,
    ):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_meas = n_meas
        # 去掉多频融合，直接使用卷积编码器
        self.encoder = ConvEncoder(in_channels=1, base_ch=base_ch)
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, voltages: torch.Tensor) -> torch.Tensor:
        B = voltages.shape[0]
        # 输入: (B, 1, 208) 或 (B, 208)
        if voltages.dim() == 2:
            # (B, 208) -> (B, 1, 1, 208)
            x = voltages.unsqueeze(1).unsqueeze(2)
        elif voltages.dim() == 3:
            # (B, 1, 208) -> (B, 1, 1, 208)
            if voltages.shape[1] == 1:
                x = voltages.unsqueeze(2)
            else:
                # 如果输入是多频，只取第一个频率
                x = voltages[:, 0:1].unsqueeze(2)
        else:
            raise ValueError(f"Expected voltages with 2 or 3 dims, got {voltages.shape}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        scale = x.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-8)
        x = x / scale
        feat, _ = self.encoder(x)
        return self.global_proj(feat)
