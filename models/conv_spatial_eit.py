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


class FrequencyCrossAttention(nn.Module):
    """多频 Cross-Attention 融合层 — 替换 1×1 Conv，建模频率间非线性交互"""
    def __init__(self, n_freq: int = 6, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.proj_in = nn.Conv2d(n_freq, d_model, 1, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, d_model, 13, 16) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5
        self.proj_out = nn.Conv2d(d_model, 1, 1, bias=False)

    def forward(self, x):
        B = x.shape[0]
        h = self.proj_in(x) + self.pos_embed              # (B, d_model, 13, 16)
        h_flat = h.view(B, self.d_model, -1).transpose(1, 2)  # (B, 208, d_model)
        Q = self.q_proj(h_flat)                            # (B, 208, d_model)
        K = self.k_proj(h_flat)
        V = self.v_proj(h_flat)
        attn = (Q @ K.transpose(-2, -1)) * self.scale     # (B, 208, 208)
        attn = F.softmax(attn, dim=-1)
        out = attn @ V                                     # (B, 208, d_model)
        out = out.transpose(1, 2).view(B, self.d_model, 13, 16)
        out = self.proj_out(out)                           # (B, 1, 13, 16)
        return out


class SEModule(nn.Module):
    """SE 通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvEncoder(nn.Module):
    """
    Conv2D 编码器 v2
    - 单频输入 (in_channels=1)
    - 不下采样，保持 13×16 原生分辨率
    - SE 通道注意力
    - 跳跃连接输出 (f1: stage1, f2: stage2+SE)
    - 输出 (B, 256, 13, 16), [f1: (B, 192, 13, 16), f2: (B, 384, 13, 16)]
    """
    def __init__(self, in_channels: int = 1, base_ch: int = 96):
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

        # SE 通道注意力 (stage2 后)
        self.se = SEModule(base_ch * 4)

        # 输出投影
        self.out_channels = base_ch * 2 + 64  # = 256
        self.out_proj = nn.Sequential(
            nn.Conv2d(base_ch * 4, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

        # 跳跃连接通道数记录 (含主路径输出，用于 GNN 第一层维度)
        self.total_out_channels = self.out_channels + base_ch * 2 + base_ch * 4  # = 832

    def forward(self, x):
        x = self.stem(x)       # (B, base_ch, 13, 16)
        f1 = self.stage1(x)    # (B, base_ch*2, 13, 16)  跳跃连接 1
        x = self.stage2(f1)    # (B, base_ch*4, 13, 16)
        x = self.se(x)         # (B, base_ch*4, 13, 16)
        f2 = x                 # (B, base_ch*4, 13, 16)  跳跃连接 2
        x = self.out_proj(x)   # (B, out_channels, 13, 16)
        return x, [f1, f2]


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
    """图卷积层（稀疏边列表 + MLP 更新，内存高效）
    Phase 1 升级: 支持边特征 (edge_feat) 输入
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1,
                 edge_dim: int = 0):
        super().__init__()
        self.edge_dim = edge_dim
        if edge_dim > 0:
            self.edge_net = nn.Sequential(
                nn.Linear(edge_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
            )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, edge_idx, edge_weight, edge_feat=None):
        B, N, D = x.shape
        n_edges = edge_idx.shape[1]
        device = x.device
        src, dst = edge_idx
        w = edge_weight

        chunk_size = 20000
        x_agg = torch.zeros(B, N, D, device=device)
        if self.edge_dim > 0 and edge_feat is not None:
            # 边特征编码: (n_edges, edge_dim) → (n_edges,)
            e_score = self.edge_net(edge_feat).sum(dim=-1)  # (n_edges,)
            for start in range(0, n_edges, chunk_size):
                end = min(start + chunk_size, n_edges)
                s, d = src[start:end], dst[start:end]
                # 边特征调制权重: w * sigmoid(edge_score)
                ew = w[start:end] * torch.sigmoid(e_score[start:end])
                x_src = x[:, s]
                x_agg.scatter_add_(1,
                    d.view(1, -1, 1).expand(B, -1, D),
                    x_src * ew.view(1, -1, 1))
        else:
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


class GATv2Layer(nn.Module):
    """
    GATv2 图注意力层 — 多头动态注意力
    - 注意力系数同时依赖 sender 和 receiver 节点特征（比 GAT 更动态）
    - 边特征作为注意力偏置
    - 输出维度必须能被 n_heads 整除
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1,
                 n_heads: int = 4, edge_dim: int = 0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0, \
            f"out_dim({out_dim}) 必须能被 n_heads({n_heads}) 整除"

        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)

        attn_in = self.head_dim * 2 + edge_dim
        self.a = nn.Sequential(
            nn.Linear(attn_in, self.head_dim, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(self.head_dim, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_idx, edge_weight, edge_feat=None):
        B, N, D = x.shape
        device = x.device
        src, dst = edge_idx

        # 线性投影 + 多头拆分
        Q = self.W_q(x).view(B, N, self.n_heads, self.head_dim)
        K = self.W_k(x).view(B, N, self.n_heads, self.head_dim)
        V = self.W_v(x).view(B, N, self.n_heads, self.head_dim)

        # 每条边的 sender Q 和 receiver K
        Q_src = Q[:, src]  # (B, E, n_heads, head_dim)
        K_dst = K[:, dst]  # (B, E, n_heads, head_dim)

        # 注意力分数计算 (GATv2: concat Q||K + leaky_relu + linear)
        attn_input = torch.cat([Q_src, K_dst], dim=-1)
        if edge_feat is not None:
            ef = edge_feat.unsqueeze(0).unsqueeze(2).expand(B, -1, self.n_heads, -1)
            attn_input = torch.cat([attn_input, ef], dim=-1)

        e = self.a(attn_input).squeeze(-1)                     # (B, E, n_heads)
        e = e + torch.log(edge_weight.clamp_min(1e-12)).view(1, -1, 1)

        # 对每个目标节点的入边分别做 softmax，而不是对全图所有边归一化。
        dst_index = dst.view(1, -1, 1).expand(B, -1, self.n_heads)
        max_per_dst = torch.full((B, N, self.n_heads), -float('inf'),
                                 device=device, dtype=e.dtype)
        max_per_dst.scatter_reduce_(1, dst_index, e, reduce='amax',
                                    include_self=True)
        e_max = max_per_dst.gather(1, dst_index)
        exp_e = torch.exp(e - e_max)
        denom = torch.zeros(B, N, self.n_heads, device=device, dtype=e.dtype)
        denom.scatter_add_(1, dst_index, exp_e)
        alpha = exp_e / (denom.gather(1, dst_index) + 1e-12)   # (B, E, n_heads)
        alpha = self.dropout(alpha)

        # 消息聚合
        msg = V[:, src] * alpha.unsqueeze(-1)                  # (B, E, n_heads, head_dim)
        x_agg = torch.zeros(B, N, self.n_heads, self.head_dim,
                            device=device, dtype=msg.dtype)
        x_agg.scatter_add_(1, dst.view(1, -1, 1, 1).expand(B, -1, self.n_heads, self.head_dim), msg)
        x_agg = x_agg.contiguous().view(B, N, -1)             # (B, N, out_dim)
        return x_agg


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
                 sigma_max: float = 0.1,
                 use_gat: bool = True,
                 n_heads: int = 4):
        super().__init__()

        self.n_elems = n_elems
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.use_gat = use_gat
        self.n_heads = n_heads

        # 0. 多频融合层（6频 → 1频，自动学习权重）
        self.freq_fusion = FrequencyCrossAttention(n_freq=n_frequencies, d_model=64)

        # 1. Conv Encoder（单频输入）
        self.encoder = ConvEncoder(in_channels=1, base_ch=96)

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

    def setup_mesh(self, centers: np.ndarray, elements: np.ndarray,
                   jacobian: Optional[np.ndarray] = None,
                   sigma_ref: float = 0.01):
        """
        设置网格结构 + 位置编码 + Jacobian (Phase 1)

        参数:
            centers:  (n_elems, 2) 单元中心坐标
            elements: (n_elems, 3) 单元节点索引
            jacobian: (n_meas, n_elems) 灵敏度矩阵 (可选)
            sigma_ref: 参考电导率 (用于线性化)
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

        undirected_edges = set()
        for elems in node_to_elems.values():
            for a in range(len(elems)):
                for b in range(a + 1, len(elems)):
                    i, j = int(elems[a]), int(elems[b])
                    if i != j:
                        undirected_edges.add((min(i, j), max(i, j)))

        directed_edges = []
        for i, j in sorted(undirected_edges):
            directed_edges.append((i, j))
            directed_edges.append((j, i))
        directed_edges.extend((i, i) for i in range(n_elems))
        edge_list = np.array(directed_edges, dtype=np.int64).T  # (2, n_edges)

        deg = np.ones(n_elems, dtype=np.float32)  # self-loop
        for i, j in undirected_edges:
            deg[i] += 1.0
            deg[j] += 1.0
        deg = np.sqrt(deg) + 1e-8
        edge_weight = np.ones(edge_list.shape[1], dtype=np.float32)
        edge_weight /= deg[edge_list[0]] * deg[edge_list[1]]

        self._edge_idx = torch.from_numpy(edge_list).long()
        self._edge_weight = torch.from_numpy(edge_weight).float()

        # ── 边特征 (Phase 1新增) ──
        n_edges = edge_list.shape[1]
        edge_feat = np.zeros((n_edges, 4), dtype=np.float32)
        # 特征0: 单元中心距离 (归一化到 [0,1])
        max_dist = np.max(np.abs(centers[:, :2])) * 2 + 1e-8
        element_nodes = [set(map(int, tri)) for tri in elements]
        for e_idx, (i, j) in enumerate(zip(edge_list[0], edge_list[1])):
            ci, cj = centers[i, :2], centers[j, :2]
            dist = np.linalg.norm(ci - cj)
            edge_feat[e_idx, 0] = dist / (dist + 0.002)  # 归一化到~[0,1]
            # 特征1: 共享节点数 / 3
            shared = len(element_nodes[i] & element_nodes[j])
            edge_feat[e_idx, 1] = shared / 3.0
            # 特征2: 单元中心相对方向 (单位向量的 dot product ≈ 面积比)
            ri = np.linalg.norm(ci) / max_dist
            rj = np.linalg.norm(cj) / max_dist
            edge_feat[e_idx, 2] = min(ri, rj) / (max(ri, rj) + 1e-8)
            # 特征3: 方向角度的余弦相似性
            if dist > 1e-8:
                dot_ij = (ci * cj).sum() / ((np.linalg.norm(ci) + 1e-8) * (np.linalg.norm(cj) + 1e-8))
            else:
                dot_ij = 1.0
            edge_feat[e_idx, 3] = (dot_ij + 1) / 2.0  # → [0,1]
        self.register_buffer('_edge_feat', torch.from_numpy(edge_feat).float())
        edge_dim = 4

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
        gnn_in_dim = self.encoder.total_out_channels + self.pos_dim
        if self.use_gat:
            # GATv2 多头注意力图卷积
            self.gnn_blocks = nn.ModuleList([
                GATv2Layer(
                    gnn_in_dim if i == 0 else self.gnn_hidden,
                    self.gnn_hidden,
                    dropout=self.gnn_dropout,
                    n_heads=self.n_heads,
                    edge_dim=edge_dim,
                )
                for i in range(self.gnn_layers)
            ])
        else:
            # SimpleGNN 回退（消融实验用）
            self.gnn_blocks = nn.ModuleList([
                SimpleGNNLayer(
                    gnn_in_dim if i == 0 else self.gnn_hidden,
                    self.gnn_hidden,
                    self.gnn_dropout,
                    edge_dim=edge_dim,
                )
                for i in range(self.gnn_layers)
            ])

        # ── Jacobian 残差反投影 (Phase 1新增) ──
        self.sigma_ref = sigma_ref
        self._has_jacobian = False
        if jacobian is not None:
            # jacobian: (n_meas, n_elems)
            J = torch.from_numpy(jacobian).float()
            self.register_buffer('J', J)                      # (n_meas, n_elems)
            self.register_buffer('J_T', J.T.contiguous())     # (n_elems, n_meas)
            # 残差校正头 (输入: gnn_hidden + 1, 输出: 1)
            self.correction_head = nn.Sequential(
                nn.Linear(self.gnn_hidden + 1, self.gnn_hidden // 2),
                nn.GELU(),
                nn.Dropout(self.gnn_dropout * 0.5),
                nn.Linear(self.gnn_hidden // 2, 1),
            )
            self._has_jacobian = True
        else:
            self.register_buffer('J', torch.zeros(1, 1))
            self.register_buffer('J_T', torch.zeros(1, 1))
            self.correction_head = nn.Identity()

        print(f"  [ConvSpatial] 网格: {n_elems} 单元, {edge_list.shape[1]} 条有向边, "
              f"位置编码: {self.pos_dim}维"
              + (f", Jᵀr: 已启用" if self._has_jacobian else ""))

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
        feat, skip_feats = self.encoder(x)  # (B, 256, 13, 16), [skip1, skip2]

        # 2. Grid Sampling
        node_feat = self.sampler(feat)  # (B, n_elems, 256)

        # 2b. 跳跃连接采样
        skip_sampled = []
        for sf in skip_feats:
            ss = self.sampler(sf)
            skip_sampled.append(ss)

        # 3. 拼接位置编码（P0-2）
        pe = self.pos_encoding.to(device).unsqueeze(0).expand(B, -1, -1)
        node_feat = torch.cat([node_feat] + skip_sampled, dim=-1)  # (B, n_elems, 832)
        node_feat = torch.cat([node_feat, pe], dim=-1)  # (B, n_elems, 832+pos_dim)

        # 4. GNN
        h = node_feat
        if self._edge_idx is not None:
            edge_idx = self._edge_idx.to(device)
            edge_weight = self._edge_weight.to(device)
            edge_feat = self._edge_feat.to(device) if hasattr(self, '_edge_feat') else None
            for gnn in self.gnn_blocks:
                h = gnn(h, edge_idx, edge_weight, edge_feat=edge_feat)

        # 5a. 初始预测 σ₀（主路径）
        sigma_raw_0 = self.output_head(h).squeeze(-1)        # (B, n_elems)

        # 5b. Jᵀr 残差反投影 (Phase 1新增)
        delta = 0.0
        if self._has_jacobian:
            # 用物理范围内的 σ₀ 做线性化，避免未约束 logits 经 J 放大后溢出。
            sigma_0_phys = torch.sigmoid(sigma_raw_0) * (self.sigma_max - self.sigma_min) + self.sigma_min
            sigma_delta = sigma_0_phys - self.sigma_ref       # (B, n_elems)
            V_lin = (self.J.unsqueeze(0) @ sigma_delta.float().unsqueeze(-1)
                     ).squeeze(-1).to(sigma_raw_0.dtype)      # (B, n_meas)
            # 电压残差: 用融合后的单频电压，并按样本尺度归一化。
            V_meas = x.squeeze(1).view(B, -1)                 # (B, 208)
            v_scale = V_meas.detach().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            r = (V_meas - V_lin) / v_scale                    # (B, n_meas)
            # 反投影到网格后标准化，避免 correction_head 输入尺度失控。
            g = (self.J_T.unsqueeze(0) @ r.float().unsqueeze(-1)
                 ).squeeze(-1).to(sigma_raw_0.dtype)          # (B, n_elems)
            g = (g - g.mean(dim=-1, keepdim=True)) / (g.std(dim=-1, keepdim=True) + 1e-6)
            # 拼接到 GNN 隐藏层 → 残差校正
            h_corr = torch.cat([h, g.unsqueeze(-1)], dim=-1)  # (B, n_elems, gnn_hidden+1)
            delta = self.correction_head(h_corr).squeeze(-1)  # (B, n_elems)

        # 5c. 最终输出: σ = Sigmoid(σ₀ + Δσ) → [σ_min, σ_max]
        sigma_raw = sigma_raw_0 + delta
        sigma = torch.sigmoid(sigma_raw) * (self.sigma_max - self.sigma_min) + self.sigma_min

        return {
            'sigma': sigma,
            'sigma_0': torch.sigmoid(sigma_raw_0) * (self.sigma_max - self.sigma_min) + self.sigma_min,
            'delta': delta if isinstance(delta, torch.Tensor) else torch.zeros_like(sigma),
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
    print(f"输出 sigma_0: {out['sigma_0'].shape}")
    print(f"范围: [{out['sigma'].min().item():.4f}, {out['sigma'].max().item():.4f}]")
    print(f"参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"位置编码: {model.pos_dim}维")
    print(f"Jᵀr: {'启用' if model._has_jacobian else '禁用'}")

    loss = out['sigma'].mean()
    loss.backward()
    print("✅ 前向+反向通过")
