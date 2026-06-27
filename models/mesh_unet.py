"""
Hierarchical Mesh U-Net for diffusion denoising (v3 — FiLM + per-level conditioning).
======================================================================
GNN-based encoder-decoder on multi-scale graph hierarchy.
As the denoising network ε_θ(σ_t, t, V) → σ_0 for DiffEIT.

v3 improvements:
  - FiLM (Feature-wise Linear Modulation) for time/voltage conditioning at EVERY level
  - Optional extra features (J^T·V, J_energy) passed as additional node input channels
  - Bottleneck Cross-Attention to voltage latent
  - Per-level conditioning replaces single L0-only injection
"""
import torch
import torch.nn as nn
import numpy as np
from models.conv_spatial_eit import SimpleGNNLayer


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: γ·x + β, 逐层注入条件"""
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(cond_dim, feat_dim),
            nn.Tanh(),
        )
        self.beta = nn.Sequential(
            nn.Linear(cond_dim, feat_dim),
            nn.Tanh(),
        )
        # 初始化: γ≈1, β≈0 → 条件初始不影响特征
        nn.init.zeros_(self.gamma[0].weight)
        nn.init.constant_(self.gamma[0].bias, 2.0)   # Tanh(2) ≈ 0.96 ≈ 1
        nn.init.zeros_(self.beta[0].weight)
        nn.init.constant_(self.beta[0].bias, 0.0)    # Tanh(0) = 0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        cond: (B, C)
        """
        gamma = self.gamma(cond).unsqueeze(1)  # (B, 1, D)
        beta = self.beta(cond).unsqueeze(1)    # (B, 1, D)
        return gamma * x + beta


class MeshConvBlock(nn.Module):
    """GNN + LayerNorm + GELU + residual"""
    def __init__(self, in_dim, out_dim, edge_dim=4, dropout=0.1):
        super().__init__()
        self.gnn = SimpleGNNLayer(in_dim, out_dim, dropout=dropout, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_idx, edge_weight, edge_feat):
        return self.act(self.norm(self.gnn(x, edge_idx, edge_weight, edge_feat=edge_feat) + self.skip(x)))


class MeshUNet(nn.Module):
    """
    Multi-scale GNN U-Net (v3: FiLM conditioning at all levels).

    Level 0: n_elems nodes,  dim = hidden
    Level 1: ~n/4 nodes,      dim = hidden*2
    Level 2: ~n/16 nodes,     dim = hidden*2
    Bottleneck: ~n/64 nodes,  dim = hidden*2 + Cross-Attention

    Input:  σ_t (B, n_elems, 1), t_emb (B, Tdim), v_emb (B, Vdim)
            extra_feat (B, n_elems, F)  [optional — J^T·V, J_energy, etc.]
    Output: σ_0_pred (B, n_elems, 1)  [x₀-prediction mode]
    """

    def __init__(self, n_elems=4424, hidden_dim=384, time_dim=256,
                 voltage_dim=512, pos_dim=35, extra_dim=0, dropout=0.1):
        super().__init__()
        h = hidden_dim
        self.hidden_dim = h
        self.extra_dim = extra_dim

        # Input projection: σ_t + V_broadcast + PE + radius + extra_feat
        in_dim = 1 + voltage_dim + pos_dim + 1 + extra_dim
        self.node_proj = nn.Linear(in_dim, h)

        # ---- Per-level FiLM conditioning ----
        # Encoder: FiLM applied BEFORE conv blocks → use INPUT dimension at each level
        #   L0: input h (from node_proj), L1: input h (from pool L0),
        #   L2: input h*2 (from pool L1), BN: input h*2 (from pool L2)
        dims_enc_in = [h, h, h*2, h*2]
        self.enc_time_films = nn.ModuleList([FiLM(time_dim, d) for d in dims_enc_in])
        self.enc_volt_films = nn.ModuleList([FiLM(voltage_dim, d) for d in dims_enc_in])

        # Decoder: FiLM applied AFTER conv blocks → use OUTPUT dimension at each level
        #   BN→2 output: h*2, 2→1 output: h*2, 1→0 output: h, final refine output: h//2
        dims_dec_out = [h*2, h*2, h, h//2]
        self.dec_time_films = nn.ModuleList([FiLM(time_dim, d) for d in dims_dec_out])
        self.dec_volt_films = nn.ModuleList([FiLM(voltage_dim, d) for d in dims_dec_out])

        # ---- Encoder ----
        self.enc0 = nn.ModuleList([MeshConvBlock(h, h, dropout=dropout),
                                   MeshConvBlock(h, h, dropout=dropout)])
        self.enc1 = nn.ModuleList([MeshConvBlock(h, h*2, dropout=dropout),
                                   MeshConvBlock(h*2, h*2, dropout=dropout)])
        self.enc2 = nn.ModuleList([MeshConvBlock(h*2, h*2, dropout=dropout),
                                   MeshConvBlock(h*2, h*2, dropout=dropout)])
        self.enc_bn = nn.ModuleList([MeshConvBlock(h*2, h*2, dropout=dropout),
                                     MeshConvBlock(h*2, h*2, dropout=dropout)])

        # Bottleneck: Self-Attention + Cross-Attention to voltage
        self.bn_self_attn = nn.MultiheadAttention(h*2, num_heads=4, dropout=dropout, batch_first=True)
        # Cross-attn: mesh nodes attend to learnable voltage-conditioned latent tokens
        self.v_latent_proj = nn.Linear(voltage_dim, h*2)
        self.v_latent_tokens = nn.Parameter(torch.randn(4, h*2) * 0.02)  # learnable query bases
        self.bn_cross_attn = nn.MultiheadAttention(h*2, num_heads=4, dropout=dropout, batch_first=True)

        # ---- Decoder ----
        self.dec_bn = nn.ModuleList([MeshConvBlock(h*4, h*2, dropout=dropout),
                                     MeshConvBlock(h*2, h*2, dropout=dropout)])
        self.dec2 = nn.ModuleList([MeshConvBlock(h*4, h*2, dropout=dropout),
                                   MeshConvBlock(h*2, h*2, dropout=dropout)])
        self.dec1 = nn.ModuleList([MeshConvBlock(h*3, h, dropout=dropout),
                                   MeshConvBlock(h, h, dropout=dropout)])
        self.dec0 = nn.ModuleList([MeshConvBlock(h, h, dropout=dropout),
                                   MeshConvBlock(h, h//2, dropout=dropout)])

        # ---- Output head (x₀-prediction: predict clean σ in N(0,1) space) ----
        # RankGauss ensures data is N(0,1); Linear output allows full ±3σ range
        self.out_head = nn.Sequential(
            nn.Linear(h//2, h//4), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(h//4, 1),
        )

        self.hierarchy = None
        self.pos_encoding = None
        self.pool = None
        self.unpool = None

    def setup_mesh(self, hierarchy, pos_encoding):
        from models.mesh_pooling import GraphPool, GraphUnpool
        self.hierarchy = hierarchy
        self.pos_encoding = pos_encoding
        self.pool = GraphPool()
        self.unpool = GraphUnpool()

    def _gather_edges(self, level_idx, device):
        lv = self.hierarchy[level_idx]
        return (lv['edges'].to(device), lv['edge_weight'].to(device),
                lv['edge_feat'].to(device))

    def forward(self, sigma_t, t_emb, v_emb, extra_feat=None):
        """
        sigma_t:   (B, N0, 1)  noisy conductivity
        t_emb:     (B, Tdim)   time embedding
        v_emb:     (B, Vdim)   voltage condition
        extra_feat: (B, N0, F) optional extra node features (J^T·V, J_energy, etc.)

        Returns: σ_0_pred (B, N0, 1) — clean conductivity prediction
        """
        if self.hierarchy is None:
            raise RuntimeError("Call setup_mesh() first.")

        B, N0, _ = sigma_t.shape
        device = sigma_t.device
        L = self.hierarchy

        # ---- Build node features ----
        pe = self.pos_encoding.to(device).unsqueeze(0).expand(B, -1, -1)   # (B, N0, P)
        radius = pe[:, :, :2].norm(dim=-1, keepdim=True)                     # (B, N0, 1)
        v_node = v_emb.unsqueeze(1).expand(-1, N0, -1)                       # (B, N0, V)

        feat_parts = [sigma_t, v_node, pe, radius]
        if extra_feat is not None:
            feat_parts.append(extra_feat)
        elif self.extra_dim > 0:
            # unconditional mode: pad zeros to keep input dim consistent
            feat_parts.append(torch.zeros(B, N0, self.extra_dim, device=device))
        x = torch.cat(feat_parts, dim=-1)
        x = self.node_proj(x)                                                # (B, N0, h)

        # ====== ENCODER with per-level FiLM ======
        # L0
        x = self.enc_time_films[0](x, t_emb) + self.enc_volt_films[0](x, v_emb)
        for blk in self.enc0:
            x = blk(x, *self._gather_edges(0, device))
        s0 = x  # (B, N0, h)

        # Pool 0→1
        x = self.pool(x, L[1]['cluster'].to(device), L[1]['nodes'])
        x = self.enc_time_films[1](x, t_emb) + self.enc_volt_films[1](x, v_emb)
        for blk in self.enc1:
            x = blk(x, *self._gather_edges(1, device))
        s1 = x  # (B, N1, h*2)

        # Pool 1→2
        x = self.pool(x, L[2]['cluster'].to(device), L[2]['nodes'])
        x = self.enc_time_films[2](x, t_emb) + self.enc_volt_films[2](x, v_emb)
        for blk in self.enc2:
            x = blk(x, *self._gather_edges(2, device))
        s2 = x  # (B, N2, h*2)

        # Pool 2→BN
        x = self.pool(x, L[3]['cluster'].to(device), L[3]['nodes'])
        x = self.enc_time_films[3](x, t_emb) + self.enc_volt_films[3](x, v_emb)
        for blk in self.enc_bn:
            x = blk(x, *self._gather_edges(3, device))

        # Bottleneck attention
        x_self, _ = self.bn_self_attn(x, x, x)
        x = x + x_self  # Self-attn residual

        # Cross-attention to voltage: mesh nodes attend to learnable latent tokens
        v_base = self.v_latent_proj(v_emb).unsqueeze(1)          # (B, 1, h*2)
        v_latent = v_base + self.v_latent_tokens.unsqueeze(0)    # (B, 4, h*2) — 4 distinct tokens
        x_cross, _ = self.bn_cross_attn(x, v_latent, v_latent)
        x = x + x_cross  # Cross-attn residual

        # ====== DECODER with per-level FiLM (applied AFTER blocks) ======
        # Unpool BN→2
        x = self.unpool(x, L[3]['cluster'].to(device), s2.shape[1])
        x = torch.cat([x, s2], dim=-1)  # h*2 + h*2 = h*4
        for blk in self.dec_bn:
            x = blk(x, *self._gather_edges(2, device))
        x = self.dec_time_films[0](x, t_emb) + self.dec_volt_films[0](x, v_emb)  # dim h*2

        # Unpool 2→1
        x = self.unpool(x, L[2]['cluster'].to(device), s1.shape[1])
        x = torch.cat([x, s1], dim=-1)  # h*2 + h*2 = h*4
        for blk in self.dec2:
            x = blk(x, *self._gather_edges(1, device))
        x = self.dec_time_films[1](x, t_emb) + self.dec_volt_films[1](x, v_emb)  # dim h*2

        # Unpool 1→0
        x = self.unpool(x, L[1]['cluster'].to(device), s0.shape[1])
        x = torch.cat([x, s0], dim=-1)  # h*2 + h = h*3
        for blk in self.dec1:
            x = blk(x, *self._gather_edges(0, device))
        x = self.dec_time_films[2](x, t_emb) + self.dec_volt_films[2](x, v_emb)  # dim h

        # Final refine
        for blk in self.dec0:
            x = blk(x, *self._gather_edges(0, device))
        x = self.dec_time_films[3](x, t_emb) + self.dec_volt_films[3](x, v_emb)  # dim h//2

        return self.out_head(x)  # (B, N0, 1) in [0, 1]
