"""Losses for the residual EIT route."""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_float_tensor(array: np.ndarray) -> torch.Tensor:
    try:
        return torch.from_numpy(array).float()
    except RuntimeError:
        return torch.tensor(array.tolist(), dtype=torch.float32)


class ResidualMeasurementConsistencyLoss(nn.Module):
    """
    Enforce J * delta_sigma ~= residual.

    residual is the voltage residual after the traditional coarse
    reconstruction:

        r = V_measured - J(sigma_0 - sigma_ref)
    """

    def __init__(self, jacobian: Union[np.ndarray, torch.Tensor], normalize: bool = True):
        super().__init__()
        if isinstance(jacobian, np.ndarray):
            jacobian = _as_float_tensor(jacobian)
        if jacobian.dim() == 3:
            jacobian = jacobian[0]
        self.register_buffer("J", jacobian.float())
        self.normalize = normalize

    def forward(self, delta_sigma: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if residual.dim() == 3:
            residual = residual[:, 0, :]
        V_delta = (self.J.unsqueeze(0) @ delta_sigma.float().unsqueeze(-1)).squeeze(-1)
        V_delta = V_delta.to(residual.dtype)

        if self.normalize:
            scale = residual.detach().norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return F.mse_loss(V_delta / scale, residual / scale)
        return F.mse_loss(V_delta, residual)


class ResidualSparsityLoss(nn.Module):
    """Encourage the neural correction to be the smallest necessary change."""

    def forward(self, delta_sigma: torch.Tensor) -> torch.Tensor:
        return delta_sigma.abs().mean()


class ResidualSmoothnessLoss(nn.Module):
    """Smoothness regularization on delta_sigma over mesh neighbor edges."""

    def __init__(self, edge_idx: Optional[torch.Tensor] = None):
        super().__init__()
        if edge_idx is not None:
            self.register_buffer("edge_idx", edge_idx.long())
        else:
            self.edge_idx = None

    def forward(self, delta_sigma: torch.Tensor, edge_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        edges = edge_idx if edge_idx is not None else self.edge_idx
        if edges is None or edges.numel() == 0:
            return torch.tensor(0.0, device=delta_sigma.device, dtype=delta_sigma.dtype)
        edges = edges.to(delta_sigma.device)
        diff = delta_sigma[:, edges[0]] - delta_sigma[:, edges[1]]
        return diff.pow(2).mean()


class RelativeMSELoss(nn.Module):
    """Supervised loss with optional mask weighting to balance inclusion vs background."""

    def __init__(self, inclusion_weight: float = 10.0):
        super().__init__()
        self.inclusion_weight = inclusion_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = (pred - target) ** 2
        if masks is not None:
            # 内含物区域放大 inclusion_weight 倍
            weight = 1.0 + (self.inclusion_weight - 1.0) * masks
            return (weight * diff).mean()
        return diff.mean()


def weighted_residual_loss(losses: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    total = None
    for name, loss in losses.items():
        if name not in weights:
            continue
        term = weights[name] * loss
        total = term if total is None else total + term
    if total is None:
        raise ValueError("No matching loss names found for the provided weights.")
    return total
