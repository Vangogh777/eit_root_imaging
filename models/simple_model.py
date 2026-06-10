"""
简化版 SF-SBLC 模型 - 用于快速测试
=====================================
仅保留核心功能，便于调试和验证
"""

import torch
import torch.nn as nn


class SimpleSFSBLC(nn.Module):
    """
    简化版 EIT 重建网络

    输入: (B, n_freq, n_meas) 边界电压
    输出: (B, n_elems) 电导率分布
    """

    def __init__(self, input_dim: int = 208, hidden_dim: int = 256,
                 n_frequencies: int = 6, n_elems: int = 2824):
        super().__init__()

        self.n_freq = n_frequencies
        self.n_elems = n_elems

        # 1. 输入编码：将多频电压展平后编码
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * n_frequencies, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 2. 中间处理：残差块
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(4)]
        )

        # 3. 输出解码：映射到网格单元
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, n_elems),
        )

        # 输出范围：电导率约 0.01-0.05，使用 sigmoid 缩放到合理范围
        self.output_scale = nn.Parameter(torch.tensor([0.02]))  # 可学习的缩放因子

    def forward(self, voltages: torch.Tensor) -> dict:
        """
        参数:
            voltages: (B, n_freq, n_meas)

        返回:
            dict: {'sigma': (B, n_elems)}
        """
        B = voltages.shape[0]

        # 展平多频输入
        x = voltages.view(B, -1)  # (B, n_freq * n_meas)

        # 编码
        h = self.encoder(x)  # (B, hidden_dim)

        # 残差处理
        h = self.res_blocks(h)  # (B, hidden_dim)

        # 解码
        sigma_raw = self.decoder(h)  # (B, n_elems)

        # 使用sigmoid缩放到合理范围 [0.005, 0.1]
        # sigmoid输出[0,1]，缩放到[sigma_min, sigma_max]
        sigma_min, sigma_max = 0.005, 0.1
        sigma = torch.sigmoid(sigma_raw) * (sigma_max - sigma_min) + sigma_min

        return {'sigma': sigma}


class ResBlock(nn.Module):
    """简单的残差块"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


if __name__ == "__main__":
    # 快速测试
    model = SimpleSFSBLC(input_dim=208, hidden_dim=256, n_frequencies=6, n_elems=2824)
    x = torch.randn(4, 6, 208)
    out = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out['sigma'].shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ 简化模型测试通过")
