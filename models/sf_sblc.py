"""
SF-SBLC 完整网络
================
Spatial-Frequency Shared and Base Layer Correction
融合：
  1. Shared Encoder — 多频共享编码
  2. Base Layer Correction — 基础层校正（系统伪影抑制）
  3. Frequency Fusion — 频率融合解码
  4. ResNet Backbone — 深度残差重建骨干

输入:  边界电压 (B, n_freq, n_meas)
输出:  电导率映射 (B, n_elems)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保项目根在 sys.path 中（兼容直接运行和导入运行）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.shared_encoder import SharedEncoder
from models.base_layer_correction import BaseLayerCorrection
from models.frequency_fusion import FrequencyFusionDecoder
from models.resnet_backbone import ResNetBackbone


class SFSBLC(nn.Module):
    """
    SF-SBLC 完整网络

    用法:
        model = SFSBLC(input_dim=256, n_frequencies=6, n_elems=1500)
        sigma = model(voltages)  # (B, n_elems)

    训练模式:
        - 有监督: loss = MSE(sigma_pred, sigma_gt)
        - 无监督: loss = ||F(sigma_pred) - V||² + TV(sigma_pred)
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512,
                 n_frequencies: int = 6, n_elems: int = 1500,
                 n_encoder_layers: int = 4, n_res_blocks: int = 8,
                 dropout: float = 0.1, use_attention: bool = True,
                 backbone_mode: str = "mlp"):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_freq = n_frequencies
        self.n_elems = n_elems

        # 1. 多频共享编码器
        self.encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_frequencies=n_frequencies,
            n_layers=n_encoder_layers,
            dropout=dropout,
        )

        # 2. 基础层校正 (BLC)
        self.blc = BaseLayerCorrection(
            hidden_dim=hidden_dim,
            n_frequencies=n_frequencies,
            n_elems=n_elems,
            dropout=dropout,
        )

        # 3. 频率融合解码器
        self.fusion_decoder = FrequencyFusionDecoder(
            hidden_dim=hidden_dim,
            n_frequencies=n_frequencies,
            n_elems=n_elems,
            n_res_blocks=n_res_blocks,
            dropout=dropout,
            use_attention=use_attention,
        )

        # 4. 深度残差骨干
        self.backbone = ResNetBackbone(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_elems=n_elems,
            n_blocks=n_res_blocks,
            mode=backbone_mode,
            dropout=dropout,
        )

        # 5. 全局残差连接（从BLC基础层直接到输出）
        self.global_residual = nn.Linear(n_elems, n_elems)

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, voltages: torch.Tensor) -> dict:
        """
        前向传播

        参数:
            voltages: (B, n_freq, n_meas) 边界电压

        返回:
            dict 包含:
                - 'sigma': (B, n_elems) 预测电导率
                - 'base_map': (B, n_elems) 基础层估计（可解释性）
                - 'freq_weights': (B, F) 频率注意力权重
                - 'blc_gates': (B, F) BLC校正门控值
        """
        # 1. 编码
        h = self.encoder(voltages)  # (B, F, D)

        # 2. 基础层校正
        h_corrected, base_map, blc_gates = self.blc(h)

        # 3. 频率融合解码
        sigma_fused, freq_weights = self.fusion_decoder(h_corrected)

        # 4. 深层次细化（残差骨干）
        # 将多频特征池化为统一表示作为骨干输入
        h_pool = h_corrected.mean(dim=1)       # (B, D) 所有频率平均
        sigma_refined = self.backbone(h_pool)  # (B, n_elems)

        # 5. 融合 + 全局残差
        sigma = sigma_fused + sigma_refined

        # 基础层残差（直接从BLC输出跳到最终输出）
        base_residual = self.global_residual(base_map)
        sigma = sigma + 0.1 * base_residual

        return {
            'sigma': sigma,
            'base_map': base_map,
            'freq_weights': freq_weights,
            'blc_gates': blc_gates,
        }

    def predict(self, voltages: torch.Tensor) -> torch.Tensor:
        """
        推理便捷接口

        参数:
            voltages: (B, n_freq, n_meas) 或 (n_freq, n_meas)

        返回:
            sigma: (B, n_elems) 或 (n_elems,)
        """
        if voltages.dim() == 2:
            voltages = voltages.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(voltages)
            sigma = out['sigma']

        if voltages.size(0) == 1:
            sigma = sigma.squeeze(0)

        return sigma


class SFSBLC_Light(nn.Module):
    """
    SF-SBLC 轻量版（快速原型验证）
    减少参数量，加快训练速度
    """

    def __init__(self, input_dim: int = 256, n_frequencies: int = 6,
                 n_elems: int = 1500):
        super().__init__()

        # 简化编码器
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * n_frequencies, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        # 简化重建
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, n_elems),
        )

    def forward(self, voltages: torch.Tensor) -> dict:
        z = self.encoder(voltages)
        sigma = self.decoder(z)
        return {'sigma': sigma, 'base_map': None, 'freq_weights': None, 'blc_gates': None}


if __name__ == "__main__":
    # 快速测试
    model = SFSBLC(input_dim=256, n_frequencies=6, n_elems=1500)
    x = torch.randn(4, 6, 256)  # (B, F, M)
    out = model(x)
    print(f"输出 sigma shape: {out['sigma'].shape}")
    print(f"基础层: {out['base_map'].shape if out['base_map'] is not None else 'None'}")
    print(f"参数总量: {sum(p.numel() for p in model.parameters()):,}")
    print("[测试通过] SF-SBLC 前向传播正常")
