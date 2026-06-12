"""
GNN 物理编码器
==============
使用图神经网络编码 EIT 网格的空间结构和物理约束。

核心思想：
- 网格单元作为图的节点
- 相邻单元作为图的边
- 编码空间关系和物理约束（Jacobian）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (GAT)
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features

        # 多头注意力
        self.W_q = nn.Linear(in_features, out_features * n_heads, bias=False)
        self.W_k = nn.Linear(in_features, out_features * n_heads, bias=False)
        self.W_v = nn.Linear(in_features, out_features * n_heads, bias=False)

        self.out_proj = nn.Linear(out_features * n_heads, out_features)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, N, D) 节点特征
            adj: (B, N, N) 或 (N, N) 邻接矩阵

        返回:
            (B, N, out_features)
        """
        B, N, D = x.shape

        # 多头投影
        Q = self.W_q(x).view(B, N, self.n_heads, self.out_features).transpose(1, 2)  # (B, H, N, d)
        K = self.W_k(x).view(B, N, self.n_heads, self.out_features).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.out_features).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.out_features)  # (B, H, N, N)

        # 应用邻接矩阵掩码（只关注邻居）
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # 掩码：非邻居位置的注意力设为 -inf
        mask = (adj == 0).unsqueeze(1)  # (B, 1, N, N)
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 聚合
        out = torch.matmul(attn, V)  # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, H*d)
        out = self.out_proj(out)

        # 残差 + LayerNorm
        if D == self.out_features:
            out = self.layer_norm(x + out)
        else:
            out = self.layer_norm(out)

        return out


class GraphConvLayer(nn.Module):
    """
    简化版图卷积层 (GCN)
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, N, D)
            adj: (N, N) 或 (B, N, N) 归一化邻接矩阵
        """
        # 归一化邻接矩阵
        if adj.dim() == 2:
            # D^{-1/2} A D^{-1/2}
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
            deg_inv_sqrt = deg.pow(-0.5)
            adj_norm = deg_inv_sqrt * adj * deg_inv_sqrt.t()
        else:
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
            deg_inv_sqrt = deg.pow(-0.5)
            adj_norm = deg_inv_sqrt * adj * deg_inv_sqrt.transpose(-1, -2)

        # 图卷积
        out = torch.matmul(adj_norm, x) if adj_norm.dim() == 2 else torch.matmul(adj_norm, x)
        out = self.linear(out)
        out = self.dropout(out)
        out = self.layer_norm(out)

        return out


class PhysicsEncoder(nn.Module):
    """
    物理编码器

    输入:
        - 节点特征 (坐标、初始电导率估计等)
        - 邻接矩阵
        - 可选: Jacobian 信息

    输出:
        - 物理感知的特征表示
    """

    def __init__(self,
                 node_dim: int = 3,          # 节点特征维度 (x, y, sigma_init)
                 hidden_dim: int = 256,
                 output_dim: int = 512,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 use_attention: bool = True):
        super().__init__()

        self.use_attention = use_attention

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 图神经网络层
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            if use_attention:
                self.gnn_layers.append(
                    GraphAttentionLayer(hidden_dim, hidden_dim, n_heads, dropout)
                )
            else:
                self.gnn_layers.append(
                    GraphConvLayer(hidden_dim, hidden_dim, dropout)
                )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self,
                node_features: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        参数:
            node_features: (B, N, node_dim) 节点特征
            adj: (N, N) 或 (B, N, N) 邻接矩阵

        返回:
            (B, N, output_dim) 编码后的节点特征
        """
        # 输入投影
        h = self.input_proj(node_features)  # (B, N, hidden_dim)

        # 图神经网络层
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, adj)

        # 输出投影
        out = self.output_proj(h)  # (B, N, output_dim)

        return out

    def get_global_feature(self,
                           node_features: torch.Tensor,
                           adj: torch.Tensor) -> torch.Tensor:
        """
        获取全局特征（用于后续融合）
        """
        node_feats = self.forward(node_features, adj)  # (B, N, output_dim)
        global_feat = node_feats.mean(dim=1)  # (B, output_dim)
        return global_feat


class JacobianEncoder(nn.Module):
    """
    Jacobian 矩阵编码器

    将 Jacobian 信息编码为图节点特征
    """

    def __init__(self,
                 n_meas: int = 208,
                 hidden_dim: int = 128,
                 output_dim: int = 256):
        super().__init__()

        # 每个单元的 Jacobian 向量编码
        self.encoder = nn.Sequential(
            nn.Linear(n_meas, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, jacobian: torch.Tensor) -> torch.Tensor:
        """
        参数:
            jacobian: (B, n_meas, n_elems) 或 (n_meas, n_elems)
                      每个单元对测量的敏感度

        返回:
            (B, n_elems, output_dim) 每个单元的编码特征
        """
        if jacobian.dim() == 2:
            jacobian = jacobian.unsqueeze(0)

        B, n_meas, n_elems = jacobian.shape

        # 转置: (B, n_elems, n_meas)
        J = jacobian.transpose(1, 2)

        # 编码每个单元
        out = self.encoder(J)  # (B, n_elems, output_dim)

        return out


class PhysicsGNN(nn.Module):
    """
    完整的物理 GNN 模块

    融合:
        - 节点坐标
        - 初始电导率估计
        - Jacobian 信息
        - 图结构
    """

    def __init__(self,
                 n_meas: int = 208,
                 hidden_dim: int = 256,
                 output_dim: int = 512,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 use_jacobian: bool = True):
        super().__init__()

        self.use_jacobian = use_jacobian
        self.output_dim = output_dim

        # Jacobian 编码器
        if use_jacobian:
            self.jacobian_encoder = JacobianEncoder(n_meas, hidden_dim // 2, hidden_dim)
            node_dim = 2 + hidden_dim  # (x, y) + jacobian_feat
        else:
            node_dim = 2  # 只有坐标

        # 物理图编码器
        self.physics_encoder = PhysicsEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            use_attention=True,
        )

    def forward(self,
                centers: torch.Tensor,
                adj: torch.Tensor,
                jacobian: Optional[torch.Tensor] = None,
                sigma_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            centers: (B, N, 2) 或 (N, 2) 单元中心坐标
            adj: (N, N) 邻接矩阵
            jacobian: (B, n_meas, N) 或 (n_meas, N) Jacobian矩阵
            sigma_init: (B, N) 初始电导率估计 (可选)

        返回:
            (B, output_dim) 全局物理特征
        """
        # 处理维度
        if centers.dim() == 2:
            centers = centers.unsqueeze(0)
        B = centers.shape[0]
        N = centers.shape[1]

        # 构建节点特征
        if self.use_jacobian and jacobian is not None:
            # 编码 Jacobian
            J_feat = self.jacobian_encoder(jacobian)  # (B, N, hidden_dim)
            node_features = torch.cat([centers, J_feat], dim=-1)  # (B, N, 2+hidden_dim)
        else:
            node_features = centers  # (B, N, 2)

        # 可选: 添加初始电导率估计
        if sigma_init is not None:
            if sigma_init.dim() == 1:
                sigma_init = sigma_init.unsqueeze(0)
            sigma_init = sigma_init.unsqueeze(-1)  # (B, N, 1)
            node_features = torch.cat([node_features, sigma_init], dim=-1)

        # 图神经网络编码
        node_feats = self.physics_encoder(node_features, adj)  # (B, N, output_dim)

        # 全局池化
        global_feat = node_feats.mean(dim=1)  # (B, output_dim)

        return global_feat, node_feats


def build_adjacency_from_elements(elements: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    从单元连接关系构建节点邻接矩阵

    参数:
        elements: (n_elems, 3) 三形单元节点索引
        n_nodes: 节点数量

    返回:
        (n_nodes, n_nodes) 邻接矩阵
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for elem in elements:
        # 三角形三个顶点两两相连
        for i in range(3):
            for j in range(i + 1, 3):
                n1, n2 = elem[i], elem[j]
                adj[n1, n2] = 1
                adj[n2, n1] = 1

    return adj


def build_element_adjacency(elements: np.ndarray, n_elems: int) -> np.ndarray:
    """
    构建单元邻接矩阵（共享边的单元相邻）

    参数:
        elements: (n_elems, 3) 三形单元节点索引
        n_elems: 单元数量

    返回:
        (n_elems, n_elems) 单元邻接矩阵
    """
    adj = np.zeros((n_elems, n_elems), dtype=np.float32)

    # 找到每个单元的边
    elem_edges = []
    for elem in elements:
        edges = set()
        for i in range(3):
            n1, n2 = sorted([elem[i], elem[(i + 1) % 3]])
            edges.add((n1, n2))
        elem_edges.append(edges)

    # 检查相邻
    for i in range(n_elems):
        for j in range(i + 1, n_elems):
            # 如果共享至少一条边
            shared = elem_edges[i] & elem_edges[j]
            if len(shared) > 0:
                adj[i, j] = 1
                adj[j, i] = 1

    # 添加自连接
    np.fill_diagonal(adj, 1)

    return adj


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 60)
    print("测试 GNN 物理编码器")
    print("=" * 60)

    # 模拟数据
    B, N, n_meas = 4, 4000, 208
    hidden_dim, output_dim = 256, 512

    # 节点坐标
    centers = torch.randn(B, N, 2)

    # 邻接矩阵（稀疏，模拟真实情况）
    adj = torch.zeros(N, N)
    for i in range(N):
        # 每个节点连接约6个邻居
        neighbors = torch.randint(0, N, (6,))
        adj[i, neighbors] = 1
        adj[neighbors, i] = 1
    adj.fill_diagonal_(1)

    # Jacobian
    jacobian = torch.randn(B, n_meas, N) * 0.01

    # 测试完整模型
    print("\n测试 PhysicsGNN...")
    model = PhysicsGNN(
        n_meas=n_meas,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=4,
        n_heads=4,
    )

    global_feat, node_feats = model(centers, adj, jacobian)
    print(f"  输入 centers: {centers.shape}")
    print(f"  输入 adj: {adj.shape}")
    print(f"  输入 jacobian: {jacobian.shape}")
    print(f"  输出 global_feat: {global_feat.shape}")
    print(f"  输出 node_feats: {node_feats.shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n✅ GNN 物理编码器测试通过")
