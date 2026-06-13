"""
改进版 EIT GNN 模型 - 利用 Jacobian 物理信息
=============================================

核心改进：
1. 预计算 Jacobian 矩阵（测量敏感度）
2. GNN 输入：坐标 + Jacobian 特征 + 邻接关系
3. 物理感知的图卷积

流程：
    电压 → MLP → voltage_feat
    Jacobian → 编码 → 每个单元的敏感度特征
    网格 + 敏感度 → GNN → physics_feat
    融合 → 解码 → sigma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


class JacobianFeatureEncoder(nn.Module):
    """
    Jacobian 特征编码器

    将 Jacobian 矩阵编码为每个单元的特征
    输入: (B, n_meas, n_elems) 或 (n_meas, n_elems)
    输出: (B, n_elems, feat_dim)
    """

    def __init__(self, n_meas: int = 208, feat_dim: int = 128):
        super().__init__()

        # 每个单元的 Jacobian 向量 (n_meas,) 编码为特征
        self.encoder = nn.Sequential(
            nn.Linear(n_meas, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, jacobian: torch.Tensor) -> torch.Tensor:
        """
        参数:
            jacobian: (n_meas, n_elems) 或 (B, n_meas, n_elems)

        返回:
            (n_elems, feat_dim) 或 (B, n_elems, feat_dim)
        """
        if jacobian.dim() == 2:
            # (n_meas, n_elems) -> (n_elems, n_meas)
            J = jacobian.T.unsqueeze(0)  # (1, n_elems, n_meas)
        else:
            # (B, n_meas, n_elems) -> (B, n_elems, n_meas)
            J = jacobian.transpose(1, 2)

        return self.encoder(J)  # (B, n_elems, feat_dim)


class PhysicsGConvLayer(nn.Module):
    """
    物理感知图卷积层

    结合：
    - 邻接关系（空间连接）
    - Jacobian 信息（测量敏感度）
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()

        self.linear_self = nn.Linear(in_dim, out_dim)  # 自身特征
        self.linear_neigh = nn.Linear(in_dim, out_dim)  # 邻居特征
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, N, D) 节点特征
            adj: (N, N) 邻接矩阵

        返回:
            (B, N, out_dim)
        """
        B, N, D = x.shape

        # 自身特征
        h_self = self.linear_self(x)  # (B, N, out_dim)

        # 邻居聚合
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)  # (1, N, N)

        # 度归一化
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / degree

        # 图卷积
        h_neigh = torch.matmul(adj_norm, x)  # (B, N, D)
        h_neigh = self.linear_neigh(h_neigh)  # (B, N, out_dim)

        # 组合
        h = h_self + h_neigh
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)

        # 残差
        if D == h.shape[-1]:
            h = h + x

        return h


class ImprovedPhysicsGNN(nn.Module):
    """
    改进版物理 GNN

    输入：
    - 网格坐标 (x, y)
    - Jacobian 特征（每个单元的测量敏感度）
    - 邻接矩阵
    """

    def __init__(self,
                 n_meas: int = 208,
                 n_elems: int = 2824,
                 coord_dim: int = 2,
                 jacobian_feat_dim: int = 64,
                 hidden_dim: int = 256,
                 output_dim: int = 512,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.n_elems = n_elems

        # Jacobian 编码器
        self.jacobian_encoder = JacobianFeatureEncoder(n_meas, jacobian_feat_dim)

        # 节点特征维度：坐标(2) + Jacobian特征
        node_dim = coord_dim + jacobian_feat_dim

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 图卷积层
        self.gnn_layers = nn.ModuleList([
            PhysicsGConvLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self,
                centers: torch.Tensor,
                adj: torch.Tensor,
                jacobian: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            centers: (B, N, 2) 或 (N, 2) 单元中心坐标
            adj: (N, N) 邻接矩阵
            jacobian: (n_meas, n_elems) 或 (B, n_meas, n_elems) Jacobian矩阵

        返回:
            (B, output_dim) 全局特征
        """
        # 处理坐标维度
        if centers.dim() == 2:
            centers = centers.unsqueeze(0)
        B = centers.shape[0]
        N = centers.shape[1]
        device = centers.device

        # 构建 Jacobian 特征
        if jacobian is not None:
            if jacobian.dim() == 2:
                # 共享 Jacobian，扩展到 batch
                jacobian = jacobian.unsqueeze(0).expand(B, -1, -1)
            J_feat = self.jacobian_encoder(jacobian)  # (B, N, jacobian_feat_dim)
        else:
            # 没有 Jacobian，用零填充或随机初始化
            J_feat = torch.zeros(B, N, 64, device=device)

        # 组合节点特征
        node_feat = torch.cat([centers, J_feat], dim=-1)  # (B, N, node_dim)

        # 输入投影
        h = self.input_proj(node_feat)  # (B, N, hidden_dim)

        # 图卷积
        for layer in self.gnn_layers:
            h = layer(h, adj)

        # 输出投影
        h = self.output_proj(h)  # (B, N, output_dim)

        # 全局池化
        global_feat = h.mean(dim=1)  # (B, output_dim)

        return global_feat


class ImprovedEITModelGNN(nn.Module):
    """
    改进版 EIT GNN 模型

    整合：
    - MLP 电压编码器
    - 改进版物理 GNN（带 Jacobian）
    - 特征融合
    - 残差解码器
    """

    def __init__(self,
                 input_dim: int = 208,
                 n_frequencies: int = 1,
                 n_elems: int = 2824,
                 hidden_dim: int = 512,
                 n_res_blocks: int = 6,
                 gnn_layers: int = 4,
                 jacobian_feat_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()

        self.n_elems = n_elems
        self.hidden_dim = hidden_dim
        self.sigma_min = 0.005
        self.sigma_max = 0.1

        # ============ 1. 电压编码器 (MLP) ============
        self.voltage_encoder = nn.Sequential(
            nn.Linear(input_dim * n_frequencies, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ============ 2. 改进版物理 GNN ============
        self.physics_gnn = ImprovedPhysicsGNN(
            n_meas=input_dim,
            n_elems=n_elems,
            coord_dim=2,
            jacobian_feat_dim=jacobian_feat_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim,
            n_layers=gnn_layers,
            dropout=dropout,
        )

        # ============ 3. 特征融合 ============
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_elems),
        )

        # 缓存
        self._adj_cache = None
        self._centers_cache = None
        self._jacobian_cache = None

    def setup_mesh(self, centers: np.ndarray, elements: np.ndarray, jacobian: Optional[np.ndarray] = None):
        """
        设置网格结构和 Jacobian

        参数:
            centers: (n_elems, 2) 单元中心坐标
            elements: (n_elems, 3) 单元节点索引
            jacobian: (n_meas, n_elems) Jacobian矩阵（可选）
        """
        from models.physics_gnn import build_element_adjacency

        adj = build_element_adjacency(elements, len(elements))
        self._adj_cache = torch.from_numpy(adj).float()
        self._centers_cache = torch.from_numpy(centers).float()

        if jacobian is not None:
            self._jacobian_cache = torch.from_numpy(jacobian).float()

    def forward(self, voltages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        参数:
            voltages: (B, n_freq, n_meas) 边界电压

        返回:
            {'sigma': (B, n_elems)}
        """
        B = voltages.shape[0]
        device = voltages.device

        if self._adj_cache is None:
            raise RuntimeError("请先调用 setup_mesh()")

        adj = self._adj_cache.to(device)
        centers = self._centers_cache.to(device)

        # 扩展到 batch
        centers_batch = centers.unsqueeze(0).expand(B, -1, -1)

        # Jacobian
        jacobian = None
        if self._jacobian_cache is not None:
            jacobian = self._jacobian_cache.to(device)

        # ============ 1. 电压编码 ============
        v_flat = voltages.view(B, -1)
        v_feat = self.voltage_encoder(v_flat)

        # ============ 2. 物理 GNN 编码 ============
        physics_feat = self.physics_gnn(centers_batch, adj, jacobian)

        # ============ 3. 融合 ============
        fused = torch.cat([v_feat, physics_feat], dim=-1)
        h = self.fusion(fused)

        # ============ 4. 残差处理 ============
        h = self.res_blocks(h)

        # ============ 5. 解码 ============
        sigma_raw = self.decoder(h)

        # 限制范围
        sigma = torch.clamp(sigma_raw, self.sigma_min, self.sigma_max)

        return {'sigma': sigma}


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ============ 测试 ============
if __name__ == "__main__":
    print("=" * 60)
    print("测试改进版 EIT GNN 模型")
    print("=" * 60)

    n_elems = 2824
    n_meas = 208

    # 模拟网格
    centers = np.random.randn(n_elems, 2).astype(np.float32) * 0.1
    elements = np.random.randint(0, 1000, (n_elems, 3)).astype(np.int64)

    # 模拟 Jacobian（服务器上会预计算）
    jacobian = np.random.randn(n_meas, n_elems).astype(np.float32) * 0.01

    # 模拟输入
    B = 4
    voltages = torch.randn(B, 1, n_meas) * 1e-5

    # 测试模型
    print("\n测试 ImprovedEITModelGNN（带 Jacobian）...")
    model = ImprovedEITModelGNN(
        input_dim=n_meas,
        n_frequencies=1,
        n_elems=n_elems,
        hidden_dim=512,
    )
    model.setup_mesh(centers, elements, jacobian)

    out = model(voltages)
    print(f"  输入: {voltages.shape}")
    print(f"  输出: {out['sigma'].shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 不使用 Jacobian 的情况
    print("\n测试不带 Jacobian...")
    model2 = ImprovedEITModelGNN(
        input_dim=n_meas,
        n_frequencies=1,
        n_elems=n_elems,
        hidden_dim=512,
    )
    model2.setup_mesh(centers, elements, jacobian=None)

    out2 = model2(voltages)
    print(f"  输出: {out2['sigma'].shape}")

    print("\n✅ 测试通过")