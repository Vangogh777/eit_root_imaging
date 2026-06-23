"""Mesh GNN blocks for residual EIT reconstruction."""

import torch
import torch.nn as nn
from typing import Optional

from models.conv_spatial_eit import GATv2Layer, SimpleGNNLayer


class ResidualMeshGNN(nn.Module):
    """Predict a bounded conductivity residual on FEM elements."""

    def __init__(
        self,
        node_dim: int,
        global_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_gat: bool = True,
        n_heads: int = 4,
        edge_dim: int = 0,
        delta_scale: float = 0.02,
    ):
        super().__init__()
        self.delta_scale = delta_scale
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        layer_cls = GATv2Layer if use_gat else SimpleGNNLayer
        self.layers = nn.ModuleList([
            layer_cls(
                hidden_dim,
                hidden_dim,
                dropout=dropout,
                n_heads=n_heads,
                edge_dim=edge_dim,
            ) if use_gat else layer_cls(
                hidden_dim,
                hidden_dim,
                dropout=dropout,
                edge_dim=edge_dim,
            )
            for _ in range(n_layers)
        ])

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        z_v: torch.Tensor,
        edge_idx: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = node_feat.shape
        z_expand = z_v.unsqueeze(1).expand(B, N, -1)
        h = self.node_proj(torch.cat([node_feat, z_expand], dim=-1))

        for layer in self.layers:
            h = h + layer(h, edge_idx, edge_weight, edge_feat=edge_feat)

        raw_delta = self.delta_head(h).squeeze(-1)
        return self.delta_scale * torch.tanh(raw_delta)
