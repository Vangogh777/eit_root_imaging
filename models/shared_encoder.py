"""
多频共享编码器 (Shared Encoder)
================================
对多频率的边界电压进行共享权重的特征提取。
核心思想：不同频率的电压使用同一组编码器参数，
使编码器学习"频率无关"的空间特征表示。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    """
    共享编码器

    输入: (B, n_freq, n_meas) — 多频边界电压
    输出: (B, n_freq, hidden_dim) — 每个频率的隐特征

    设计:
        - 每频率独立编码但共享权重
        - 1D 卷积堆叠 + 残差连接
        - 最终通过跨频率注意力融合
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512,
                 n_frequencies: int = 6, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_freq = n_frequencies
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 输入投影: 每个测量通道独立投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
        )

        # 共享权重的一维卷积层堆叠
        conv_layers = []
        in_ch = hidden_dim
        for i in range(n_layers):
            conv_layers.extend([
                nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_ch = hidden_dim
        self.shared_convs = nn.ModuleList([
            nn.Sequential(*conv_layers) for _ in range(n_frequencies)
        ])

        # 跨频率注意力 (Transformer-inspired)
        self.cross_freq_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, n_freq, n_meas)  # 批大小, 频率数, 测量通道数

        返回:
            h: (B, n_freq, hidden_dim)  # 每个频率的编码特征
        """
        B, F, M = x.shape
        assert F == self.n_freq, f"频率数不匹配: 输入{F} vs 期望{self.n_freq}"

        # 1. 输入投影 (每频率独立)
        h = self.input_proj(x)  # (B, F, hidden_dim)

        # 2. 共享卷积编码 (各频率独立但权重独立)
        h_freqs = []
        for f in range(F):
            h_f = h[:, f, :]                    # (B, hidden_dim)
            h_f = h_f.unsqueeze(-1)             # (B, hidden_dim, 1)
            h_f = self.shared_convs[f](h_f)      # (B, hidden_dim, 1)
            h_f = h_f.squeeze(-1)               # (B, hidden_dim)
            h_freqs.append(h_f)

        h = torch.stack(h_freqs, dim=1)         # (B, F, hidden_dim)

        # 3. 跨频率自注意力 (频率间的信息交互)
        h_attn, _ = self.cross_freq_attn(h, h, h)
        h = self.attn_norm(h + h_attn)           # 残差连接

        # 4. 输出投影
        h = self.output_proj(h)                  # (B, F, hidden_dim)

        return h


class FrequencyGate(nn.Module):
    """
    频率选择门控
    为每个输入样本动态选择最佳频率组合
    """

    def __init__(self, n_frequencies: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, n_frequencies),
            nn.Softmax(dim=-1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        参数:
            h: (B, F, hidden_dim)

        返回:
            gated: (B, F, hidden_dim) 频率加权的特征
            weights: (B, F) 频率重要性权重
        """
        # 每个频率的全局表示
        h_pool = h.mean(dim=-1)   # (B, F)
        weights = self.gate(h_pool)  # (B, F) 软选择

        # 加权
        gated = h * weights.unsqueeze(-1)
        return gated, weights
