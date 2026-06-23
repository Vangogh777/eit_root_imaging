"""Residual EIT model: traditional coarse reconstruction plus neural correction."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from models.residual_mesh_gnn import ResidualMeshGNN
from models.voltage_encoder import VoltageGlobalEncoder


def _as_float_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert numpy arrays even when torch's NumPy bridge is unavailable."""
    try:
        return torch.from_numpy(array).float()
    except RuntimeError:
        return torch.tensor(array.tolist(), dtype=torch.float32)


def _as_long_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert numpy integer arrays even when torch's NumPy bridge is unavailable."""
    try:
        return torch.from_numpy(array).long()
    except RuntimeError:
        return torch.tensor(array.tolist(), dtype=torch.long)


class ResidualComputer(nn.Module):
    """
    Compute residual physics features.

    Assumes voltages are differential voltages and J maps
    (sigma - sigma_ref) to differential voltage.
    """

    def __init__(self, jacobian: Union[np.ndarray, torch.Tensor], sigma_ref: Union[float, np.ndarray] = 0.01):
        super().__init__()
        if isinstance(jacobian, np.ndarray):
            jacobian = _as_float_tensor(jacobian)
        if jacobian.dim() == 3:
            jacobian = jacobian[0]
        self.register_buffer("J", jacobian.float())
        self.register_buffer("J_T", jacobian.float().T.contiguous())

        if isinstance(sigma_ref, np.ndarray):
            sigma_ref_t = _as_float_tensor(sigma_ref)
        else:
            sigma_ref_t = torch.full((jacobian.shape[-1],), float(sigma_ref))
        self.register_buffer("sigma_ref", sigma_ref_t)

    def forward(self, voltages: torch.Tensor, sigma_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if voltages.dim() == 3:
            V = voltages[:, 0, :]
        else:
            V = voltages

        delta0 = sigma_0 - self.sigma_ref.unsqueeze(0)
        V0 = (self.J.unsqueeze(0) @ delta0.float().unsqueeze(-1)).squeeze(-1).to(V.dtype)
        residual = V - V0
        g = (self.J_T.unsqueeze(0) @ residual.float().unsqueeze(-1)).squeeze(-1).to(V.dtype)
        g = (g - g.mean(dim=-1, keepdim=True)) / (g.std(dim=-1, keepdim=True) + 1e-6)
        return g, residual


class ResidualEIT(nn.Module):
    """
    Route B model.

    The model expects precomputed sigma_0/J^T r during training. If g/residual
    are omitted and a Jacobian was provided, they are computed on the fly.
    """

    def __init__(
        self,
        n_frequencies: int = 6,
        n_meas: int = 208,
        n_elems: int = 11466,
        hidden_dim: int = 256,
        gnn_layers: int = 4,
        dropout: float = 0.1,
        sigma_min: float = 0.005,
        sigma_max: float = 0.1,
        sigma_ref: float = 0.01,
        jacobian: Optional[Union[np.ndarray, torch.Tensor]] = None,
        delta_scale: float = 0.02,
        use_gat: bool = True,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_meas = n_meas
        self.n_elems = n_elems
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_ref_value = sigma_ref

        self.voltage_encoder = VoltageGlobalEncoder(
            n_frequencies=n_frequencies,
            n_meas=n_meas,
            hidden_dim=hidden_dim,
        )

        self.residual_computer = (
            ResidualComputer(jacobian, sigma_ref=sigma_ref) if jacobian is not None else None
        )

        self.hidden_dim = hidden_dim
        self.gnn_layers = gnn_layers
        self.dropout = dropout
        self.delta_scale = delta_scale
        self.use_gat = use_gat
        self.n_heads = n_heads

        self.mesh_gnn: Optional[ResidualMeshGNN] = None
        self.register_buffer("pos_encoding", torch.empty(0), persistent=False)
        self.register_buffer("_edge_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_edge_weight", torch.empty(0), persistent=False)
        self.register_buffer("_edge_feat", torch.empty(0), persistent=False)

    def setup_mesh(self, centers: np.ndarray, elements: np.ndarray):
        centers = centers[:, :2].astype(np.float32)
        n_elems = elements.shape[0]
        if n_elems != self.n_elems:
            self.n_elems = n_elems

        edge_idx, edge_weight, edge_feat = self._build_graph(centers, elements)
        self._edge_idx = edge_idx
        self._edge_weight = edge_weight
        self._edge_feat = edge_feat

        pe = self._build_position_encoding(centers)
        self.pos_encoding = pe

        node_dim = 3 + pe.shape[1]  # sigma_0, g, |g|, PE
        self.mesh_gnn = ResidualMeshGNN(
            node_dim=node_dim,
            global_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.gnn_layers,
            dropout=self.dropout,
            use_gat=self.use_gat,
            n_heads=self.n_heads,
            edge_dim=edge_feat.shape[1],
            delta_scale=self.delta_scale,
        )

    def forward(
        self,
        voltages: torch.Tensor,
        sigma_0: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> dict:
        if self.mesh_gnn is None or self.pos_encoding.numel() == 0:
            raise RuntimeError("Call setup_mesh(centers, elements) before forward().")

        B = voltages.shape[0]
        device = voltages.device

        if sigma_0 is None:
            sigma_0 = torch.full(
                (B, self.n_elems),
                self.sigma_ref_value,
                dtype=voltages.dtype,
                device=device,
            )
        else:
            sigma_0 = sigma_0.to(device).detach()

        if g is None or residual is None:
            if self.residual_computer is None:
                raise ValueError("g/residual must be provided when no Jacobian is attached.")
            g_calc, residual_calc = self.residual_computer(voltages, sigma_0)
            g = g_calc if g is None else g.to(device).detach()
            residual = residual_calc if residual is None else residual.to(device).detach()
        else:
            g = g.to(device).detach()
            residual = residual.to(device).detach()

        z_v = self.voltage_encoder(voltages)
        pe = self.pos_encoding.to(device).unsqueeze(0).expand(B, -1, -1)
        node_feat = torch.cat([
            sigma_0.unsqueeze(-1),
            g.unsqueeze(-1),
            g.abs().unsqueeze(-1),
            pe,
        ], dim=-1)

        delta_sigma = self.mesh_gnn(
            node_feat=node_feat,
            z_v=z_v,
            edge_idx=self._edge_idx.to(device),
            edge_weight=self._edge_weight.to(device),
            edge_feat=self._edge_feat.to(device),
        )
        sigma = torch.clamp(sigma_0 + delta_sigma, self.sigma_min, self.sigma_max)

        return {
            "sigma": sigma,
            "sigma_0": sigma_0,
            "delta_sigma": delta_sigma,
            "g": g,
            "residual": residual,
            "z_v": z_v,
        }

    @staticmethod
    def _build_position_encoding(centers: np.ndarray, n_freq: int = 8, scale: float = 2.0) -> torch.Tensor:
        r_max = np.abs(centers).max() + 1e-8
        pos = centers / r_max
        radius = np.linalg.norm(centers, axis=1, keepdims=True) / r_max

        freqs = (scale ** torch.arange(n_freq).float()) * np.pi
        args = _as_float_tensor(pos)[:, :, None] * freqs
        fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).reshape(len(pos), -1)
        return torch.cat([
            _as_float_tensor(pos),
            _as_float_tensor(radius),
            fourier,
        ], dim=-1)

    @staticmethod
    def _build_graph(centers: np.ndarray, elements: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_elems = elements.shape[0]
        node_to_elems = defaultdict(list)
        for i, tri in enumerate(elements):
            for node in tri:
                node_to_elems[int(node)].append(i)

        undirected = set()
        for elems in node_to_elems.values():
            for a in range(len(elems)):
                for b in range(a + 1, len(elems)):
                    i, j = int(elems[a]), int(elems[b])
                    if i != j:
                        undirected.add((min(i, j), max(i, j)))

        directed = []
        for i, j in sorted(undirected):
            directed.append((i, j))
            directed.append((j, i))
        directed.extend((i, i) for i in range(n_elems))

        edge_idx_np = np.asarray(directed, dtype=np.int64).T
        deg = np.ones(n_elems, dtype=np.float32)
        for i, j in undirected:
            deg[i] += 1.0
            deg[j] += 1.0
        deg = np.sqrt(deg) + 1e-8
        edge_weight = np.ones(edge_idx_np.shape[1], dtype=np.float32)
        edge_weight /= deg[edge_idx_np[0]] * deg[edge_idx_np[1]]

        edge_feat = np.zeros((edge_idx_np.shape[1], 4), dtype=np.float32)
        element_nodes = [set(map(int, tri)) for tri in elements]
        max_dist = np.max(np.abs(centers)) * 2 + 1e-8
        for e, (i, j) in enumerate(zip(edge_idx_np[0], edge_idx_np[1])):
            ci, cj = centers[i], centers[j]
            dist = np.linalg.norm(ci - cj)
            edge_feat[e, 0] = dist / (dist + 0.002)
            edge_feat[e, 1] = len(element_nodes[i] & element_nodes[j]) / 3.0
            ri = np.linalg.norm(ci) / max_dist
            rj = np.linalg.norm(cj) / max_dist
            edge_feat[e, 2] = min(ri, rj) / (max(ri, rj) + 1e-8)
            if dist > 1e-8:
                dot = (ci * cj).sum() / ((np.linalg.norm(ci) + 1e-8) * (np.linalg.norm(cj) + 1e-8))
            else:
                dot = 1.0
            edge_feat[e, 3] = (dot + 1.0) / 2.0

        return (
            _as_long_tensor(edge_idx_np),
            _as_float_tensor(edge_weight),
            _as_float_tensor(edge_feat),
        )
