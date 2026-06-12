"""
EIT 重建模型 - GNN 增强版
========================
融合 MLP 和 GNN 物理编码器

架构:
    边界电压 → MLP编码器 → 特征
    网格结构 → GNN编码器 → 物理特征
    融合 → 解码器 → 电导率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple

from models.physics_gnn import (
    PhysicsGNN,
    build_element_adjacency,
)


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class EITModelGNN(nn.Module):
    """
    EIT 重建模型 - GNN 增强版

    输入:
        - 边界电压 V: (B, n_freq, n_meas)
        - 网格结构 (centers, elements)
        - 可选: Jacobian 矩阵

    输出:
        - 电导率 sigma: (B, n_elems)
    """

    def __init__(self,
                 input_dim: int = 208,       # 测量通道数
                 n_frequencies: int = 1,     # 频率数
                 n_elems: int = 4000,        # 网格单元数
                 hidden_dim: int = 512,      # 隐藏层维度
                 n_res_blocks: int = 6,      # 残差块数
                 gnn_layers: int = 4,        # GNN层数
                 gnn_heads: int = 4,         # GNN注意力头数（当use_attention=True时使用）
                 dropout: float = 0.1,
                 use_jacobian: bool = True,
                 use_attention: bool = False):  # 默认禁用注意力，节省显存
        super().__init__()

        self.n_elems = n_elems
        self.hidden_dim = hidden_dim
        self.use_jacobian = use_jacobian

        # ============ 1. 电压编码器 (MLP) ============
        self.voltage_encoder = nn.Sequential(
            nn.Linear(input_dim * n_frequencies, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ============ 2. 物理编码器 (GNN) ============
        # 计算节点特征维度
        jacobian_feat_dim = hidden_dim // 2 if use_jacobian else 0
        node_dim = 2 + jacobian_feat_dim  # (x, y) + jacobian_feat

        self.physics_gnn = PhysicsGNN(
            n_meas=input_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim,
            n_layers=gnn_layers,
            n_heads=gnn_heads,
            dropout=dropout,
            use_jacobian=use_jacobian,
            node_dim=node_dim,  # 指定正确的节点维度
            use_attention=use_attention,  # 控制是否使用注意力机制
        )

        # ============ 3. 特征融合 ============
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ============ 4. 残差处理 ============
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout) for _ in range(n_res_blocks)]
        )

        # ============ 5. 解码器 ============
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, n_elems),
        )

        # 输出范围
        self.sigma_min = 0.005
        self.sigma_max = 0.1

        # 缓存邻接矩阵
        self._adj_cache = None
        self._centers_cache = None

    def setup_mesh(self, centers: np.ndarray, elements: np.ndarray):
        """
        设置网格结构（训练前调用一次）

        参数:
            centers: (n_elems, 2) 单元中心坐标
            elements: (n_elems, 3) 单元节点索引
        """
        # 构建单元邻接矩阵
        adj = build_element_adjacency(elements, len(elements))
        self._adj_cache = torch.from_numpy(adj).float()
        self._centers_cache = torch.from_numpy(centers).float()

    def forward(self,
                voltages: torch.Tensor,
                jacobian: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            voltages: (B, n_freq, n_meas) 边界电压
            jacobian: (B, n_meas, n_elems) 或 (n_meas, n_elems) Jacobian矩阵

        返回:
            dict: {'sigma': (B, n_elems), ...}
        """
        B = voltages.shape[0]
        device = voltages.device

        # 获取缓存的网格数据
        if self._adj_cache is None:
            raise RuntimeError("请先调用 setup_mesh() 设置网格结构")

        adj = self._adj_cache.to(device)
        centers = self._centers_cache.to(device)

        # ============ 1. 电压编码 ============
        v_flat = voltages.view(B, -1)  # (B, n_freq * n_meas)
        v_feat = self.voltage_encoder(v_flat)  # (B, hidden_dim)

        # ============ 2. 物理编码 (GNN) ============
        # 扩展 centers 到 batch
        centers_batch = centers.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

        # GNN 编码
        if self.use_jacobian and jacobian is not None:
            physics_feat, node_feats = self.physics_gnn(
                centers_batch, adj, jacobian
            )
        else:
            physics_feat, node_feats = self.physics_gnn(
                centers_batch, adj, jacobian=None
            )

        # ============ 3. 特征融合 ============
        fused = torch.cat([v_feat, physics_feat], dim=-1)  # (B, hidden_dim * 2)
        h = self.fusion(fused)  # (B, hidden_dim)

        # ============ 4. 残差处理 ============
        h = self.res_blocks(h)  # (B, hidden_dim)

        # ============ 5. 解码 ============
        sigma_raw = self.decoder(h)  # (B, n_elems)

        # 缩放到合理范围
        sigma = torch.sigmoid(sigma_raw) * (self.sigma_max - self.sigma_min) + self.sigma_min

        return {
            'sigma': sigma,
            'voltage_feat': v_feat,
            'physics_feat': physics_feat,
        }


class EITModelSimple(nn.Module):
    """
    简化版模型（不使用GNN，用于对比）
    """

    def __init__(self,
                 input_dim: int = 208,
                 n_frequencies: int = 1,
                 n_elems: int = 4000,
                 hidden_dim: int = 512,
                 n_res_blocks: int = 6,
                 dropout: float = 0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * n_frequencies, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout) for _ in range(n_res_blocks)]
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, n_elems),
        )

        self.sigma_min = 0.005
        self.sigma_max = 0.1

    def forward(self, voltages: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = voltages.shape[0]

        x = voltages.view(B, -1)
        h = self.encoder(x)
        h = self.res_blocks(h)
        sigma_raw = self.decoder(h)

        sigma = torch.sigmoid(sigma_raw) * (self.sigma_max - self.sigma_min) + self.sigma_min

        return {'sigma': sigma}

    def setup_mesh(self, centers: np.ndarray, elements: np.ndarray):
        """兼容接口"""
        pass


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 60)
    print("测试 EIT GNN 模型")
    print("=" * 60)

    # 模拟网格
    n_elems = 4000
    centers = np.random.randn(n_elems, 2).astype(np.float32) * 0.1
    elements = np.random.randint(0, 1000, (n_elems, 3)).astype(np.int64)

    # 模拟输入
    B, n_freq, n_meas = 4, 1, 208
    voltages = torch.randn(B, n_freq, n_meas) * 1e-5
    jacobian = torch.randn(B, n_meas, n_elems) * 0.01

    # 测试 GNN 模型
    print("\n测试 EITModelGNN...")
    model_gnn = EITModelGNN(
        input_dim=n_meas,
        n_frequencies=n_freq,
        n_elems=n_elems,
        hidden_dim=512,
        use_jacobian=True,
    )
    model_gnn.setup_mesh(centers, elements)

    out = model_gnn(voltages, jacobian)
    print(f"  输入 voltages: {voltages.shape}")
    print(f"  输入 jacobian: {jacobian.shape}")
    print(f"  输出 sigma: {out['sigma'].shape}")
    print(f"  参数量: {sum(p.numel() for p in model_gnn.parameters()):,}")

    # 测试简化模型
    print("\n测试 EITModelSimple...")
    model_simple = EITModelSimple(
        input_dim=n_meas,
        n_frequencies=n_freq,
        n_elems=n_elems,
        hidden_dim=512,
    )

    out_simple = model_simple(voltages)
    print(f"  输出 sigma: {out_simple['sigma'].shape}")
    print(f"  参数量: {sum(p.numel() for p in model_simple.parameters()):,}")

    print("\n✅ 模型测试通过")
