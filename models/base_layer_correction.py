"""
基础层校正模块 (Base Layer Correction, BLC)
=============================================
核心功能：从全局信息中估计和校正重建图像中的系统伪影。

BLC 的工作流程:
  1. 从编码器特征中提取"基础层"（全局结构估计）
  2. 学习一个校正项来补偿系统偏差
  3. 与细粒度特征融合

这种设计的灵感来自 SF-SBLC 中的"Shared and Base Layer Correction"概念。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLayerCorrection(nn.Module):
    """
    基础层校正模块

    输入: 编码器特征 (B, F, hidden_dim)
    输出: 校正后的特征 (B, F, hidden_dim)

    结构:
        - 全局基础层提取器
        - 校正预测器
        - 残差连接
    """

    def __init__(self, hidden_dim: int = 512, n_frequencies: int = 6,
                 n_elems: int = 1500, dropout: float = 0.1):
        super().__init__()
        self.n_freq = n_frequencies
        self.n_elems = n_elems

        # 1. 全局基础层提取器
        # 将多频编码特征聚合为"基础层"（低分辨率全局结构）
        # 输入: (B, F) — 各频率隐特征均值
        # 输出: (B, n_elems)
        self.base_extractor = nn.Sequential(
            nn.Linear(n_frequencies, hidden_dim // 4),   # (B, F) → (B, H/4)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, n_elems),         # (B, H/4) → (B, n_elems)
            nn.Tanh(),                                    # 输出归一化到 [-1, 1]
        )

        # 2. 校正预测器
        # 细粒度特征 → 残差校正
        self.correction_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),                   # 校正量归一化
        )

        # 3. 每个频率的独立校正头
        self.freq_correction = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
            ) for _ in range(n_frequencies)
        ])

        # 4. 校正门控：多少校正量被应用
        # 输入: h_f (B,D) + base_scalar (B,1) → (B, D+1)
        self.correction_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        参数:
            h: (B, F, hidden_dim) 编码器输出

        返回:
            h_corrected: (B, F, hidden_dim) 校正后的特征
            correction_maps: (B, n_elems) 基础层估计（用于可视化）
            gate_values: (B, F) 每频率门控值
        """
        B, F, D = h.shape

        # 1. 提取基础层（全局结构估计）
        # 将各频率的隐特征平均为标量，得到 (B, F) → 线性投影到单元空间
        h_pool = h.mean(dim=-1)             # (B, F) 各频率隐空间均值
        base = self.base_extractor(h_pool)  # (B, n_elems) — 基础层

        # 2. 为每个频率计算校正
        h_corrected = []
        gate_values = []
        for f in range(F):
            h_f = h[:, f, :]  # (B, D)

            # 校正预测
            correction = self.correction_predictor(h_f)  # (B, D)
            freq_specific = self.freq_correction[f](h_f)  # (B, D)

            # 组合校正
            delta = correction + freq_specific  # (B, D)

            # 门控：多少校正被应用
            base_scalar = base.mean(dim=-1, keepdim=True)  # (B, 1) 全局结构强度
            gate_input = torch.cat([h_f, base_scalar], dim=-1)  # (B, D+1)
            gate = self.correction_gate(gate_input)   # (B, 1)
            gate_values.append(gate.squeeze(-1))       # (B,)

            # 残差校正
            h_f_corrected = h_f + delta * gate
            h_corrected.append(h_f_corrected)

        h_corrected = torch.stack(h_corrected, dim=1)   # (B, F, D)
        gate_values = torch.stack(gate_values, dim=1)   # (B, F)

        return h_corrected, base, gate_values
