"""
Hierarchical Mesh U-Net for diffusion denoising (v2 - simplified dim flow).
======================================================================
GNN-based encoder-decoder on multi-scale graph hierarchy.
As the denoising network ε_θ(σ_t, t, V) for DiffEIT.
"""
import torch
import torch.nn as nn
import numpy as np
from models.conv_spatial_eit import SimpleGNNLayer


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
    Multi-scale GNN U-Net.

    Level 0: n_elems nodes,  dim = hidden
    Level 1: ~n/4 nodes,      dim = hidden*2
    Level 2: ~n/16 nodes,     dim = hidden*2
    Bottleneck: ~n/64 nodes,  dim = hidden*2 + self-attention

    Input:  σ_t (B, n_elems, 1), t_emb (B, Tdim), v_emb (B, Vdim)
    Output: ε_pred (B, n_elems, 1)
    """

    def __init__(self, n_elems=4424, hidden_dim=384, time_dim=256,
                 voltage_dim=512, pos_dim=35, dropout=0.1):
        super().__init__()
        h = hidden_dim
        self.hidden_dim = h

        # Input projection
        in_dim = 1 + voltage_dim + pos_dim + 1  # σ_t, v_enc, PE, radius
        self.node_proj = nn.Linear(in_dim, h)

        # Time/voltage modulation (added to node features at each level)
        self.time_mod = nn.Linear(time_dim, h)
        self.volt_mod = nn.Linear(voltage_dim, h)

        # ---- Encoder ----
        self.enc0 = nn.ModuleList([MeshConvBlock(h, h, dropout=dropout),
                                   MeshConvBlock(h, h, dropout=dropout)])
        self.enc1 = nn.ModuleList([MeshConvBlock(h, h*2, dropout=dropout),
                                   MeshConvBlock(h*2, h*2, dropout=dropout)])
        self.enc2 = nn.ModuleList([MeshConvBlock(h*2, h*2, dropout=dropout),
                                   MeshConvBlock(h*2, h*2, dropout=dropout)])
        self.enc_bn = nn.ModuleList([MeshConvBlock(h*2, h*2, dropout=dropout),
                                     MeshConvBlock(h*2, h*2, dropout=dropout)])

        # Bottleneck self-attention
        self.bn_attn = nn.MultiheadAttention(h*2, num_heads=4, dropout=dropout, batch_first=True)

        # ---- Decoder ----
        self.dec_bn = nn.ModuleList([MeshConvBlock(h*4, h*2, dropout=dropout),
                                     MeshConvBlock(h*2, h*2, dropout=dropout)])
        self.dec2 = nn.ModuleList([MeshConvBlock(h*4, h*2, dropout=dropout),
                                   MeshConvBlock(h*2, h*2, dropout=dropout)])
        self.dec1 = nn.ModuleList([MeshConvBlock(h*3, h, dropout=dropout),
                                   MeshConvBlock(h, h, dropout=dropout)])
        self.dec0 = nn.ModuleList([MeshConvBlock(h, h, dropout=dropout),
                                   MeshConvBlock(h, h//2, dropout=dropout)])

        # Output head
        self.out_head = nn.Sequential(
            nn.Linear(h//2, h//4), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(h//4, 1),
        )

        self.hierarchy = None
        self.pos_encoding = None
        self.pool = None
        self.unpool = None

    def setup_mesh(self, hierarchy, pos_encoding):
        from models.mesh_pooling import GraphPool, GraphUnpool
        self.hierarchy = hierarchy
        # pos_encoding is set as instance attr (not buffer since MeshUNet init already has it as None)
        self.pos_encoding = pos_encoding
        self.pool = GraphPool()
        self.unpool = GraphUnpool()

    def _gather_edges(self, level_idx, device):
        """Helper: gather edge info from hierarchy onto device."""
        lv = self.hierarchy[level_idx]
        return (lv['edges'].to(device), lv['edge_weight'].to(device),
                lv['edge_feat'].to(device))

    def forward(self, sigma_t, t_emb, v_emb):
        if self.hierarchy is None:
            raise RuntimeError("Call setup_mesh() first.")

        B, N0, _ = sigma_t.shape
        device = sigma_t.device
        L = self.hierarchy

        # ---- Node features ----
        pe = self.pos_encoding.to(device).unsqueeze(0).expand(B, -1, -1)       # (B, N0, P)
        radius = pe[:, :, :2].norm(dim=-1, keepdim=True)            # (B, N0, 1)
        v_node = v_emb.unsqueeze(1).expand(-1, N0, -1)              # (B, N0, V)

        x = torch.cat([sigma_t, v_node, pe, radius], dim=-1)        # (B, N0, D_in)
        x = self.node_proj(x)                                        # (B, N0, h)

        t_mod = self.time_mod(t_emb).unsqueeze(1)                    # (B, 1, h)
        v_mod = self.volt_mod(v_emb).unsqueeze(1)                    # (B, 1, h)

        # ====== ENCODER ======
        # L0
        x = x + t_mod + v_mod
        for blk in self.enc0:
            x = blk(x, *self._gather_edges(0, device))
        s0 = x  # (B, N0, h)

        # Pool 0→1
        x = self.pool(x, L[1]['cluster'].to(device), L[1]['nodes'])
        for blk in self.enc1:
            x = blk(x, *self._gather_edges(1, device))
        s1 = x  # (B, N1, h*2)

        # Pool 1→2
        x = self.pool(x, L[2]['cluster'].to(device), L[2]['nodes'])
        for blk in self.enc2:
            x = blk(x, *self._gather_edges(2, device))
        s2 = x  # (B, N2, h*2)

        # Pool 2→BN
        x = self.pool(x, L[3]['cluster'].to(device), L[3]['nodes'])
        for blk in self.enc_bn:
            x = blk(x, *self._gather_edges(3, device))
        x_attn, _ = self.bn_attn(x, x, x)
        x = x + x_attn  # (B, N3, h*2)

        # ====== DECODER ======
        # Unpool BN→2
        x = self.unpool(x, L[3]['cluster'].to(device), s2.shape[1])
        x = torch.cat([x, s2], dim=-1)  # h*2 + h*2 = h*4
        for blk in self.dec_bn:
            x = blk(x, *self._gather_edges(2, device))

        # Unpool 2→1
        x = self.unpool(x, L[2]['cluster'].to(device), s1.shape[1])
        x = torch.cat([x, s1], dim=-1)  # h*2 + h*2 = h*4
        for blk in self.dec2:
            x = blk(x, *self._gather_edges(1, device))

        # Unpool 1→0
        x = self.unpool(x, L[1]['cluster'].to(device), s0.shape[1])
        x = torch.cat([x, s0], dim=-1)  # h*2 + h = h*3
        for blk in self.dec1:
            x = blk(x, *self._gather_edges(0, device))

        # Final refine
        for blk in self.dec0:
            x = blk(x, *self._gather_edges(0, device))

        return self.out_head(x)  # (B, N0, 1)
