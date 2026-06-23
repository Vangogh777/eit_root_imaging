"""Voltage encoders for residual EIT models."""

import torch
import torch.nn as nn

from models.conv_spatial_eit import ConvEncoder, FrequencyCrossAttention


class VoltageGlobalEncoder(nn.Module):
    """
    Encode boundary voltages into a global conditioning vector.

    Unlike ConvSpatialEIT, this encoder does not sample convolutional features
    onto the FEM mesh. Mesh-space features in the residual route come from
    sigma_0 and J^T r.
    """

    def __init__(
        self,
        n_frequencies: int = 6,
        n_meas: int = 208,
        hidden_dim: int = 256,
        base_ch: int = 96,
    ):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_meas = n_meas
        self.freq_fusion = FrequencyCrossAttention(n_freq=n_frequencies, d_model=64)
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
        if voltages.dim() == 3:
            x = voltages.view(B, self.n_frequencies, 13, self.n_meas // 13)
        elif voltages.dim() == 4:
            x = voltages
        else:
            raise ValueError(f"Expected voltages with 3 or 4 dims, got {voltages.shape}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.freq_fusion(x)
        scale = x.flatten(1).abs().max(dim=1)[0].view(B, 1, 1, 1).clamp_min(1e-8)
        x = x / scale
        feat, _ = self.encoder(x)
        return self.global_proj(feat)
