"""
Conv-Spatial EIT — 高精度通用 EIT 重建模型
=============================================
架构: Conv2D Encoder → Grid Sampling → GNN → Output Head

输入:  voltages (B, n_freq=6, 13, 16)
输出:  sigma    (B, n_elems)

特点:
  - 2D Conv 处理测量矩阵的空间相关性
  - Grid Sampling 桥接规则网格与不规则三角网格
  - GNN 在网格拓扑上精修
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ResBlock(nn.Module):
    """2D 残差块"""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


class ConvEncoder(nn.Module):
    """
    Conv2D 编码器
    输入 (B, 6, 13, 16) → 输出 (B, 128, 8, 8)

    设计原则：13×16 本身很小，不宜过度下采样。
    只用一次 stride=2，保持空间分辨率。
    """
    def __init__(self, in_channels: int = 6, base_ch: int = 64):
        super().__init__()

        # Stem: 保持分辨率
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64ch → 128ch, 不下采样
        self.stage1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            ResBlock(base_ch * 2),
            ResBlock(base_ch * 2),
        )

        # Stage 2: 128ch → 256ch, 一次下采样
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            ResBlock(base_ch * 4),
        )

        # 输出投影到 128 维
        self.out_proj = nn.Sequential(
            nn.Conv2d(base_ch * 4, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)       # (B, 64, 13, 16)
        x = self.stage1(x)     # (B, 128, 13, 16)
        x = self.stage2(x)     # (B, 256, 7, 8)
        x = self.out_proj(x)   # (B, 128, 7, 8)
        # 插值到统一尺寸
        x = F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)
        return x               # (B, 128, 8, 8)


class GridSampler(nn.Module):
    """
    网格采样层
    用双线性插值将规则特征图映射到三角网格节点

    输入:
        feat_map: (B, C, H, W)  规则特征图
        centers:  (n_elems, 2)  单元中心坐标

    输出:
        (B, n_elems, C)
    """
    def __init__(self):
        super().__init__()
        self._grid = None

    def setup_grid(self, centers: np.ndarray):
        """设置采样网格坐标（训练/推理前调用一次）"""
        n_elems = centers.shape[0]
        # 归一化到 [-1, 1]
        x = centers[:, 0]
        y = centers[:, 1]
        r = max(abs(x).max(), abs(y).max()) + 1e-8
        grid = np.stack([x / r, y / r], axis=-1).astype(np.float32)
        # grid_sample 需要 (1, n_elems, 2) 形状
        self._grid = torch.from_numpy(grid).view(1, 1, n_elems, 2)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        feat_map: (B, C, H, W)
        return:   (B, n_elems, C)
        """
        B, C, H, W = feat_map.shape
        device = feat_map.device

        if self._grid is None:
            raise RuntimeError("请先调用 setup_grid() 设置网格坐标")

        grid = self._grid.to(device)  # (1, 1, N, 2)

        # grid_sample 输入: (B, C, H, W), grid: (B, H_out, W_out, 2)
        grid = grid.expand(B, -1, -1, -1)  # (B, 1, N, 2)

        # 注意: grid_sample 的 grid 需要 (B, H_out, W_out, 2)
        grid = grid.squeeze(1).unsqueeze(2)  # (B, N, 1, 2)

        sampled = F.grid_sample(
            feat_map, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # (B, C, N, 1)

        return sampled.squeeze(-1).transpose(1, 2)  # (B, N, C)


class SimpleGNNLayer(nn.Module):
    """图卷积层（稀疏边列表 + MLP 更新，内存高效）"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor,
                edge_idx: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, N, D)
        edge_idx:    (2, n_edges)  [src, dst]
        edge_weight: (n_edges,)    归一化权重
        """
        B, N, D = x.shape
        n_edges = edge_idx.shape[1]
        device = x.device

        src, dst = edge_idx  # (n_edges,)
        w = edge_weight      # (n_edges,)

        # 边分块消息传递（避免逐样本 for 循环，同时控制显存）
        chunk_size = 20000
        x_agg = torch.zeros(B, N, D, device=device)
        for start in range(0, n_edges, chunk_size):
            end = min(start + chunk_size, n_edges)
            s, d, ww = src[start:end], dst[start:end], w[start:end]
            # gather 当前块的边特征 (B, chunk, D)
            x_src = x[:, s]  # (B, chunk, D)
            x_agg.scatter_add_(1,
                d.view(1, -1, 1).expand(B, -1, D),
                x_src * ww.view(1, -1, 1))

        # 拼接自身 + 聚合邻居
        h = torch.cat([x, x_agg], dim=-1)
        # MLP 更新
        h = self.mlp(h.view(B * N, -1)).view(B, N, -1)
        return h


class ConvSpatialEIT(nn.Module):
    """
    Conv-Spatial EIT 完整模型

    用法:
        model = ConvSpatialEIT(n_elems=11466)
        model.setup_mesh(centers, adj)
        out = model(voltages)  # {'sigma': (B, n_elems)}
    """

    def __init__(self,
                 n_frequencies: int = 6,
                 n_meas: int = 208,
                 n_elems: int = 11466,
                 hidden_dim: int = 256,
                 gnn_layers: int = 4,
                 gnn_hidden: int = 256,
                 dropout: float = 0.1,
                 sigma_min: float = 0.005,
                 sigma_max: float = 0.1):
        super().__init__()

        self.n_elems = n_elems
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # 1. Conv Encoder
        self.encoder = ConvEncoder(in_channels=n_frequencies, base_ch=48)

        # 2. Grid Sampler
        self.sampler = GridSampler()

        # 3. GNN layers
        gnn_in_dim = 128  # ConvEncoder 输出通道
        self.gnn_blocks = nn.ModuleList([
            SimpleGNNLayer(gnn_in_dim if i == 0 else gnn_hidden, gnn_hidden, dropout)
            for i in range(gnn_layers)
        ])

        # 4. Output head (MLP per node)
        self.output_head = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(gnn_hidden // 2, gnn_hidden // 4),
            nn.GELU(),
            nn.Linear(gnn_hidden // 4, 1),
        )

        self._edge_idx = None
        self._edge_weight = None

    def setup_mesh(self, centers: np.ndarray, elements: np.ndarray):
        """
        设置网格结构

        参数:
            centers:  (n_elems, 2) 单元中心坐标
            elements: (n_elems, 3) 单元节点索引（用于构建邻接矩阵）
        """
        # 设置 Grid Sampling 坐标
        self.sampler.setup_grid(centers)

        # 构建邻接矩阵 (基于共享边)
        n_elems = elements.shape[0]
        edges = set()
        # 用单元节点索引找共享边的单元对
        from collections import defaultdict
        node_to_elems = defaultdict(list)
        for i in range(n_elems):
            for node in elements[i]:
                node_to_elems[int(node)].append(i)

        for node, elems in node_to_elems.items():
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    edge = (min(elems[i], elems[j]), max(elems[i], elems[j]))
                    edges.add(edge)

        # 构建边列表 (N, 2) 用于稀疏消息传递
        edge_list = np.array(list(edges), dtype=np.int64).T  # (2, n_edges)

        # 计算度归一化权重
        deg = np.zeros(n_elems, dtype=np.float32)
        for i, j in edges:
            deg[i] += 1.0
            deg[j] += 1.0
        deg = np.sqrt(deg) + 1e-8

        # 归一化边权重
        edge_weight = np.ones(len(edges), dtype=np.float32)
        edge_weight /= deg[edge_list[0]] * deg[edge_list[1]]

        self._edge_idx = torch.from_numpy(edge_list).long()     # (2, n_edges)
        self._edge_weight = torch.from_numpy(edge_weight).float()  # (n_edges,)
        print(f"  [ConvSpatial] 网格: {n_elems} 单元, {len(edges)} 条边 (稀疏)")

    def forward(self, voltages: torch.Tensor) -> dict:
        """
        前向传播

        参数:
            voltages: (B, n_freq, n_meas) 或 (B, 6, 13, 16)

        返回:
            {'sigma': (B, n_elems)}
        """
        B = voltages.shape[0]
        device = voltages.device

        # Reshape 输入
        if voltages.dim() == 3:
            x = voltages.view(B, -1, 13, 16)  # (B, 6, 13, 16)
        else:
            x = voltages

        # 1. Conv 编码
        feat = self.encoder(x)  # (B, 128, 8, 8)

        # 2. Grid Sampling
        node_feat = self.sampler(feat)  # (B, n_elems, 128)

        # 3. GNN (稀疏消息传递)
        h = node_feat
        if self._edge_idx is not None:
            edge_idx = self._edge_idx.to(device)
            edge_weight = self._edge_weight.to(device)
            for gnn in self.gnn_blocks:
                h = gnn(h, edge_idx, edge_weight)

        # 4. Output
        sigma_raw = self.output_head(h).squeeze(-1)  # (B, n_elems)
        sigma = torch.sigmoid(sigma_raw) * (self.sigma_max - self.sigma_min) + self.sigma_min

        return {
            'sigma': sigma,
            'base_map': None,
            'freq_weights': None,
            'blc_gates': None,
        }

    def predict(self, voltages: torch.Tensor) -> torch.Tensor:
        """推理接口"""
        if voltages.dim() == 2:
            voltages = voltages.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(voltages)
            sigma = out['sigma']

        if voltages.size(0) == 1:
            sigma = sigma.squeeze(0)
        return sigma


if __name__ == "__main__":
    # 快速测试
    n_elems = 1000
    centers = np.random.randn(n_elems, 2).astype(np.float32) * 0.1
    elements = np.random.randint(0, 500, (n_elems, 3)).astype(np.int64)

    model = ConvSpatialEIT(n_frequencies=6, n_meas=208, n_elems=n_elems)
    model.setup_mesh(centers, elements)

    x = torch.randn(4, 6, 13, 16)
    out = model(x)
    print(f"输入: {x.shape}")
    print(f"输出 sigma: {out['sigma'].shape}")
    print(f"范围: [{out['sigma'].min().item():.4f}, {out['sigma'].max().item():.4f}]")
    print(f"参数: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ ConvSpatialEIT 测试通过")
