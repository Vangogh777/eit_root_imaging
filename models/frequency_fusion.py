"""
频率融合解码器 (Frequency Fusion Decoder)
===========================================
将多频率编码特征融合并解码为电导率分布图。

核心操作:
  1. 跨频率融合: 将各频率的隐特征合并为统一的表示
  2. 上采样解码: 从隐空间映射到单元级电导率
  3. 细粒度增强: 保留高频细节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyFusion(nn.Module):
    """
    跨频率融合模块

    策略: 注意力加权 + 方差引导融合
    - 某些频率在某些位置信息量大
    - 用可学习的注意力权重做软选择
    """

    def __init__(self, hidden_dim: int = 512, n_frequencies: int = 6):
        super().__init__()

        # 每个频率的可学习查询向量
        self.freq_queries = nn.Parameter(
            torch.randn(1, n_frequencies, hidden_dim) * 0.02
        )

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * n_frequencies, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

        # 频率注意力
        self.freq_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        参数:
            h: (B, F, hidden_dim) 编码特征

        返回:
            fused: (B, hidden_dim) 融合后的统一表示
            attn_weights: (B, F) 各频率的注意力权重
        """
        B, F, D = h.shape

        # 1. 频率自注意力（频率间交互）
        h_attn, weights = self.freq_attn(h, h, h)
        weights = weights.mean(dim=1)  # (B, F)

        # 2. 频率级联
        h_concat = h_attn.reshape(B, F * D)  # (B, F*D)

        # 3. 融合投影
        fused = self.fusion(h_concat)  # (B, D)

        return fused, weights


class FrequencyFusionDecoder(nn.Module):
    """
    频率融合解码器

    输入: (B, F, hidden_dim) — 多频编码特征
    输出: (B, n_elems) — 单元级电导率分布
    """

    def __init__(self, hidden_dim: int = 512, n_frequencies: int = 6,
                 n_elems: int = 1500, n_res_blocks: int = 4,
                 dropout: float = 0.1, use_attention: bool = True):
        super().__init__()

        self.fusion = FrequencyFusion(hidden_dim, n_frequencies)

        # 解码 MLP (全连接)
        decoder_layers = []
        in_dim = hidden_dim
        for i in range(n_res_blocks):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # 中间层：带残差连接
        decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.decoder_base = nn.Sequential(*decoder_layers)

        # 残差校正头（双路径）
        self.main_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_elems),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_elems),
        )

        # 可选的注意力增强
        self.use_attention = use_attention
        if use_attention:
            from models.layers.attention_gate import CBAM
            self.cbam = CBAM(hidden_dim)

        # 频率门控: 每个频率对最终输出的贡献
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_frequencies, n_frequencies),
            nn.Softmax(dim=-1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        参数:
            h: (B, F, hidden_dim)

        返回:
            sigma: (B, n_elems) 预测的电导率分布
            freq_weights: (B, F) 各频率对输出的贡献
        """
        B, F, D = h.shape

        # 1. 频率门控（全局频率选择）
        h_pool = h.mean(dim=-1, keepdim=False)  # (B, F)
        fgate = self.freq_gate(h_pool.unsqueeze(-1)).squeeze(-1)  # (B, F)

        # 2. 跨频率融合
        h_fused, attn_weights = self.fusion(h)

        # 3. 解码基础路径
        h_decoded = self.decoder_base(h_fused)

        if self.use_attention:
            h_decoded = h_decoded.unsqueeze(-1)  # (B, D, 1)
            h_decoded = self.cbam(h_decoded)
            h_decoded = h_decoded.squeeze(-1)    # (B, D)

        # 4. 双路径输出
        main = self.main_head(h_decoded)          # (B, n_elems)
        residual = self.residual_head(h_decoded)  # (B, n_elems)

        # 5. 结合门控信息
        # 频率门控影响主路径和残差路径的平衡
        gate_bias = fgate.mean(dim=-1, keepdim=True)  # (B, 1)
        sigma = main + residual * torch.sigmoid(gate_bias)

        # 注：SFSBLC 的 forward() 中会统一做 sigmoid + 范围缩放
        return sigma, attn_weights
