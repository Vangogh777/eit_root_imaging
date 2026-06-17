"""
Conv-Spatial EIT — 高精度通用 EIT 重建模型 v2
================================================
架构: Conv2D Encoder → Grid Sampling + 位置编码 → GNN → Output Head

输入:  voltages (B, 6, 208) → 取第1频率 (B, 1, 13, 16)
输出:  sigma    (B, n_elems)

v2 改进:
  - 单频输入（P0-1: 6频相同→退化为1频）
  - GridSampler 保持13×16原生分辨率（P1-4: 不下采样）
  - GNN 注入位置编码（P0-2: 半径+Fourier坐标编码）
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
    Conv2D 编码器 v2
    - 单频输入 (in_channels=1)
    - 不下采样，保持 13×16 原生分辨率
    - 输出 (B, 128, 13, 16)
    """
    def __init__(self, in_channels: int = 1, base_ch: int = 48):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        # Stage 1: base_ch → base_ch*2, 不下采样
        self.stage1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            ResBlock(base_ch * 2),
            ResBlock(base_ch * 2),
        )

        # Stage 2: base_ch*2 → base_ch*4, 不下采样（v2: 删除 stride=2）
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            ResBlock(base_ch * 4),
        )

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(base_ch * 4, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)       # (B, base_ch, 13, 16)
        x = self.stage1(x)     # (B, base_ch*2, 13, 16)
        x = self.stage2(x)     # (B, base_ch*4, 13, 16)
        x = self.out_proj(x)   # (B, 128, 13, 16)
        return x               # 保持 13×16 原生分辨率


class GridSampler(nn.Module):
    """
    网格采样层
    用双线性插值将规则特征图映射到三角网格节点
    """
    def __init__(self):
        super().__init__()
        self._grid = None

    def setup_grid(self, centers: np.ndarray):
        """设置采样网格坐标"""
        n_elems = centers.shape[0]
        x = centers[:, 0]
        y = centers[:, 1]
        r = max(abs(x).max(), abs(y).max()) + 1e-8
        grid = np.stack([x / r, y / r], axis=-1).astype(np.float32)
        self._grid = torch.from_numpy(grid).view(1, 1, n_elems, 2)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_map.shape
        device = feat_map.device
        if self._grid is None:
            raise RuntimeError("请先调用 setup_grid() 设置网格坐标")
        grid = self._grid.to(device)
        grid = grid.expand(B, -1, -1, -1)
        grid = grid.squeeze(1).unsqueeze(2)
        sampled = F.grid_sample(feat_map, grid,
                                mode='bilinear', padding_mode='border',
                                align_corners=False)
        return sampled.squeeze(-1).transpose(1, 2)


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

    def forward(self, x, edge_idx, edge_weight):
        B, N, D = x.shape
        n_edges = edge_idx.shape[1]
        device = x.device
        src, dst = edge_idx
        w = edge_weight

        chunk_size = 20000
        x_agg = torch.zeros(B, N, D, device=device)
        for start in range(0, n_edges, chunk_size):
            end = min(start + chunk_size, n_edges)
            s, d, ww = src[start:end], dst[start:end], w[start:end]
            x_src = x[:, s]
            x_agg.scatter_add_(1,
                d.view(1, -1, 1).expand(B, -1, D),
                x_src * ww.view(1, -1, 1))

        h = torch.cat([x, x_agg], dim=-1)
        h = self.mlp(h.view(B * N, -1)).view(B, N, -1)
        return h


class ConvSpatialEIT(nn.Module):
    """
    Conv-Spatial EIT 模型 v2

    用法:
        model = ConvSpatialEIT(n_elems=11466)
        model.setup_mesh(centers, elements)
        out = model(voltages)  # voltages: (B, 6, 208)
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

        # 0. 多频融合层（6频 → 1频，自动学习权重）
        self.freq_fusion = nn.Conv2d(in_channels=n_frequencies, out_channels=1,
                                      kernel_size=1, bias=False)

        # 1. Conv Encoder（单频输入）
        self.encoder = ConvEncoder(in_channels=1, base_ch=48)

        # 2. Grid Sampler
        self.sampler = GridSampler()

        # 3. GNN（第一层 in_dim 在 setup_mesh 中动态设置）
        self.gnn_hidden = gnn_hidden
        self.gnn_layers = gnn_layers
        self.gnn_dropout = dropout
        self.gnn_blocks = None
        self.pos_dim = 0  # 位置编码维度，setup_mesh 中计算

        # 4. Output head
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
        设置网格结构 + 位置编码

        参数:
            centers:  (n_elems, 2) 单元中心坐标
            elements: (n_elems, 3) 单元节点索引
        """
        # ── Grid Sampling 坐标 ──
        self.sampler.setup_grid(centers)

        # ── 邻接矩阵（基于共享节点）──
        n_elems = elements.shape[0]
        from collections import defaultdict
        node_to_elems = defaultdict(list)
        for i in range(n_elems):
            for node in elements[i]:
                node_to_elems[int(node)].append(i)

        edges = set()
        for elems in node_to_elems.values():
            for a in range(len(elems)):
                for b in range(a + 1, len(elems)):
                    edges.add((min(elems[a], elems[b]), max(elems[a], elems[b])))

        edge_list = np.array(list(edges), dtype=np.int64).T  # (2, n_edges)

        deg = np.zeros(n_elems, dtype=np.float32)
        for i, j in edges:
            deg[i] += 1.0
            deg[j] += 1.0
        deg = np.sqrt(deg) + 1e-8
        edge_weight = np.ones(len(edges), dtype=np.float32)
        edge_weight /= deg[edge_list[0]] * deg[edge_list[1]]

        self._edge_idx = torch.from_numpy(edge_list).long()
        self._edge_weight = torch.from_numpy(edge_weight).float()

        # ── 位置编码（P0-2）──
        c = centers[:, :2].astype(np.float32)
        r_max = np.abs(c).max() + 1e-8
        pos = c / r_max

        # 半径编码（区分中心/边缘，EIT 灵敏度差异巨大）
        radius = np.linalg.norm(c, axis=1, keepdims=True) / r_max

        # Fourier 位置编码
        def fourier(x, n_freq=8, scale=2.0):
            freqs = scale ** torch.arange(n_freq).float() * np.pi
            args = torch.from_numpy(x).float()[:, :, None] * freqs
            return torch.cat([torch.sin(args), torch.cos(args)], -1).reshape(len(x), -1)

        pe = torch.cat([
            torch.from_numpy(pos).float(),
            fourier(pos),
            torch.from_numpy(radius).float(),
        ], dim=-1)  # (n_elems, pos_dim)
        self.register_buffer('pos_encoding', pe)
        self.pos_dim = pe.shape[1]

        # ── 重建 GNN（第一层输入维度 = Conv通道 + 位置编码维度）──
        gnn_in_dim = 128 + self.pos_dim
        self.gnn_blocks = nn.ModuleList([
            SimpleGNNLayer(
                gnn_in_dim if i == 0 else self.gnn_hidden,
                self.gnn_hidden,
                self.gnn_dropout,
            )
            for i in range(self.gnn_layers)
        ])

        print(f"  [ConvSpatial] 网格: {n_elems} 单元, {len(edges)} 条边, "
              f"位置编码: {self.pos_dim}维")

    def forward(self, voltages: torch.Tensor) -> dict:
        """
        前向传播

        参数:
            voltages: (B, 6, 208) 或 (B, 6, 13, 16)

        返回:
            {'sigma': (B, n_elems)}
        """
        B = voltages.shape[0]
        device = voltages.device

        # ── 多频融合 ──
        if voltages.dim() == 3:
            x = voltages.view(B, 6, 13, 16)
        else:
            x = voltages  # (B, 6, 13, 16)
        x = self.freq_fusion(x)  # (B, 1, 13, 16)

        # ── 输入归一化（P1-8）──
        amax = x.flatten(1).abs().max(dim=1)[0].view(B, 1, 1, 1) + 1e-8
        x = x / amax

        # 1. Conv 编码
        feat = self.encoder(x)  # (B, 128, 13, 16)

        # 2. Grid Sampling
        node_feat = self.sampler(feat)  # (B, n_elems, 128)

        # 3. 拼接位置编码（P0-2）
        pe = self.pos_encoding.to(device).unsqueeze(0).expand(B, -1, -1)
        node_feat = torch.cat([node_feat, pe], dim=-1)  # (B, n_elems, 128+pos_dim)

        # 4. GNN
        h = node_feat
        if self._edge_idx is not None:
            edge_idx = self._edge_idx.to(device)
            edge_weight = self._edge_weight.to(device)
            for gnn in self.gnn_blocks:
                h = gnn(h, edge_idx, edge_weight)

        # 5. Output
        sigma_raw = self.output_head(h).squeeze(-1)
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
            sigma = self.forward(voltages)['sigma']
        if sigma.dim() == 2 and sigma.size(0) == 1:
            sigma = sigma.squeeze(0)
        return sigma


if __name__ == "__main__":
    n_elems = 5000
    np.random.seed(0)
    centers = np.random.randn(n_elems, 2).astype(np.float32) * 0.1
    elements = np.random.randint(0, 500, (n_elems, 3)).astype(np.int64)

    model = ConvSpatialEIT(n_frequencies=6, n_meas=208, n_elems=n_elems)
    model.setup_mesh(centers, elements)

    x = torch.randn(4, 6, 208)  # standard input shape
    out = model(x)
    print(f"输入: {x.shape}")
    print(f"输出 sigma: {out['sigma'].shape}")
    print(f"范围: [{out['sigma'].min().item():.4f}, {out['sigma'].max().item():.4f}]")
    print(f"参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"位置编码: {model.pos_dim}维")

    loss = out['sigma'].mean()
    loss.backward()
    print("✅ 前向+反向通过")
