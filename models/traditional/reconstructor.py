"""
Traditional EIT reconstructors used by the residual pipeline.

The first implementation intentionally wraps the existing pyEIT BP/JAC path.
SBL/BSBL can be added behind the same interface after the residual pipeline is
validated on BP/JAC coarse reconstructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np

from models.two_stage_model import TraditionalReconstructor as _PyEITReconstructor


@dataclass
class ReconstructionInfo:
    method: str
    residual_norm: float = float("nan")
    failed: bool = False

    def as_dict(self) -> Dict[str, Union[float, str, bool]]:
        return {
            "method": self.method,
            "residual_norm": self.residual_norm,
            "failed": self.failed,
        }


class BaseReconstructor:
    """Common interface for traditional coarse EIT reconstruction."""

    method: str

    def reconstruct(self, voltage: np.ndarray) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError

    def batch_reconstruct(self, voltages: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        sigmas = []
        failed = []
        residual_norm = []
        for i in range(voltages.shape[0]):
            sigma_i, info_i = self.reconstruct(voltages[i])
            sigmas.append(sigma_i)
            failed.append(bool(info_i.get("failed", False)))
            residual_norm.append(float(info_i.get("residual_norm", np.nan)))

        return np.stack(sigmas).astype(np.float32), {
            "failed": np.asarray(failed, dtype=np.bool_),
            "residual_norm": np.asarray(residual_norm, dtype=np.float32),
        }


class PyEITTraditionalReconstructor(BaseReconstructor):
    """
    BP/JAC reconstructor wrapper.

    `voltage` must be absolute boundary voltage. If the dataset stores
    differential voltages, add `solver.V_uniform` before calling this class.
    """

    def __init__(self, solver, method: str = "bp"):
        if method not in {"bp", "jac"}:
            raise ValueError(f"Unsupported pyEIT method: {method}")
        self.inner = _PyEITReconstructor(solver, method=method)
        self.solver = solver
        self.method = self.inner.method
        self.sigma_ref = np.full(
            solver.n_elems,
            solver.gt_cfg.get("conductivity_soil", 0.01),
            dtype=np.float32,
        )

    def reconstruct(self, voltage: np.ndarray) -> Tuple[np.ndarray, Dict]:
        info = ReconstructionInfo(method=self.method)
        try:
            sigma = self.inner.reconstruct(voltage).astype(np.float32)
            if np.isnan(sigma).any() or np.isinf(sigma).any():
                raise FloatingPointError("Traditional reconstruction produced non-finite values")
        except Exception:
            sigma = self.sigma_ref.copy()
            info.failed = True

        return sigma, info.as_dict()


def build_reconstructor(solver, method: str = "bp", **kwargs) -> BaseReconstructor:
    """Factory kept small until non-pyEIT methods are added."""
    method = method.lower()
    if method in {"bp", "jac"}:
        return PyEITTraditionalReconstructor(solver, method=method)
    raise ValueError(
        f"Unsupported reconstructor '{method}'. Implement it in models/traditional first."
    )
