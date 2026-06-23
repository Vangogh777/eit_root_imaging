"""
Minimal ResidualEIT smoke test.

This test intentionally avoids HDF5 and pyEIT so it can validate the neural
route-B modules even when the full scientific Python stack is not available.
"""

import numpy as np
import torch

from models.residual_eit import ResidualEIT
from training.residual_loss import (
    RelativeMSELoss,
    ResidualMeasurementConsistencyLoss,
    ResidualSparsityLoss,
    ResidualSmoothnessLoss,
)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    batch_size = 2
    n_freq = 6
    n_meas = 208
    n_elems = 32
    n_nodes = 24

    centers = rng.normal(size=(n_elems, 2)).astype("float32") * 0.05
    elements = np.stack([
        np.arange(0, n_elems) % n_nodes,
        np.arange(1, n_elems + 1) % n_nodes,
        np.arange(2, n_elems + 2) % n_nodes,
    ], axis=1).astype("int64")

    # Use torch tensors for J in the smoke test so it does not depend on
    # torch's NumPy bridge in environments with NumPy ABI mismatches.
    J = torch.randn(n_meas, n_elems) * 1e-3

    model = ResidualEIT(
        n_frequencies=n_freq,
        n_meas=n_meas,
        n_elems=n_elems,
        hidden_dim=32,
        gnn_layers=1,
        dropout=0.0,
        jacobian=J,
        delta_scale=0.01,
        n_heads=4,
    )
    model.setup_mesh(centers, elements)
    model.eval()

    voltages = torch.randn(batch_size, n_freq, n_meas) * 1e-3
    sigma_0 = torch.full((batch_size, n_elems), 0.01)

    out = model(voltages, sigma_0=sigma_0)
    assert out["sigma"].shape == (batch_size, n_elems)
    assert out["delta_sigma"].shape == (batch_size, n_elems)
    assert out["g"].shape == (batch_size, n_elems)
    assert out["residual"].shape == (batch_size, n_meas)
    assert torch.isfinite(out["sigma"]).all()
    assert torch.isfinite(out["delta_sigma"]).all()

    target = sigma_0 + 0.002 * torch.randn_like(sigma_0)
    losses = {
        "supervised": RelativeMSELoss()(out["sigma"], target),
        "residual_measurement": ResidualMeasurementConsistencyLoss(J)(
            out["delta_sigma"], out["residual"]),
        "delta_l1": ResidualSparsityLoss()(out["delta_sigma"]),
        "delta_smooth": ResidualSmoothnessLoss(model._edge_idx)(out["delta_sigma"]),
    }
    for name, loss in losses.items():
        assert loss.ndim == 0, name
        assert torch.isfinite(loss), name

    print("ResidualEIT minimal smoke test passed")
    print("sigma", tuple(out["sigma"].shape), "delta", tuple(out["delta_sigma"].shape))
    print({k: float(v.detach()) for k, v in losses.items()})


if __name__ == "__main__":
    main()
