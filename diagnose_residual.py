"""Diagnose why residual EIT model is not learning."""

import os
import sys
import torch
import numpy as np
import h5py

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.datasets.eit_dataset import EITDataModule
from models.residual_eit import ResidualEIT

def diagnose():
    print("=" * 60)
    print("Residual EIT Model Diagnosis")
    print("=" * 60)

    # Load data
    dm = EITDataModule(
        h5_path="data/generated/mixed_dataset.h5",
        batch_size=8,
        load_residual_features=True,
        voltage_mask_ratio=0.0,
    )
    val_loader = dm.val_dataloader()

    # Get a batch
    batch = next(iter(val_loader))
    voltages = batch["voltages"]
    sigma_0 = batch["sigma_0"]
    g = batch["physics_g"]
    residual = batch["voltage_residual"]
    target = batch["sigmas"]

    print("\n[Input Shapes]")
    print(f"  voltages: {voltages.shape}, dtype={voltages.dtype}")
    print(f"  sigma_0: {sigma_0.shape}, dtype={sigma_0.dtype}")
    print(f"  g (J^T r): {g.shape}, dtype={g.dtype}")
    print(f"  residual: {residual.shape}, dtype={residual.dtype}")
    print(f"  target: {target.shape}, dtype={target.dtype}")

    # Check value ranges
    print("\n[Value Ranges]")
    print(f"  voltages: min={voltages.min():.6f}, max={voltages.max():.6f}, mean={voltages.mean():.6f}")
    print(f"  sigma_0: min={sigma_0.min():.6f}, max={sigma_0.max():.6f}, mean={sigma_0.mean():.6f}")
    print(f"  g: min={g.min():.6f}, max={g.max():.6f}, mean={g.mean():.6f}")
    print(f"  residual: min={residual.min():.6f}, max={residual.max():.6f}, mean={residual.mean():.6f}")
    print(f"  target: min={target.min():.6f}, max={target.max():.6f}, mean={target.mean():.6f}")

    # Check if sigma_0 == target (coarse reconstruction already perfect?)
    diff = (sigma_0 - target).abs()
    print("\n[sigma_0 vs target]")
    print(f"  max abs diff: {diff.max():.6f}")
    print(f"  mean abs diff: {diff.mean():.6f}")

    # Load Jacobian
    J = np.load("data/generated/jacobian.npy").astype(np.float32)
    if J.ndim == 3:
        J = J[0]
    print(f"\n[Jacobian]")
    print(f"  shape: {J.shape}")
    print(f"  value range: min={J.min():.6e}, max={J.max():.6e}")

    # Check if sigma_0 values are all the same (stuck at sigma_ref?)
    print("\n[sigma_0 Analysis]")
    print(f"  unique values in batch: {len(torch.unique(sigma_0))}")
    print(f"  std per sample: {sigma_0.std(dim=1).mean():.6f}")

    # Check g (J^T r) - should have variation if coarse reconstruction is meaningful
    print("\n[g (J^T r) Analysis]")
    print(f"  unique values in batch: {len(torch.unique(g))}")
    print(f"  std per sample: {g.std(dim=1).mean():.6f}")

    # Check if residual is near zero (coarse reconstruction already matches measurements?)
    print("\n[residual Analysis]")
    print(f"  L2 norm per sample: {residual.norm(dim=1).mean():.6f}")
    print(f"  relative to voltage: {(residual.norm(dim=1) / voltages[:, 0, :].norm(dim=1)).mean():.6f}")

    # Build model and check delta_sigma range
    ds = dm.val_dataset
    centers = np.mean(ds.mesh_nodes[ds.mesh_elements], axis=1)
    elements = ds.mesh_elements

    model = ResidualEIT(
        n_frequencies=ds.n_freq,
        n_meas=ds.n_meas,
        n_elems=ds.n_elems,
        hidden_dim=256,
        gnn_layers=4,
        delta_scale=0.02,  # This is the key parameter!
        jacobian=J,
    )
    model.setup_mesh(centers, elements)
    model.eval()

    with torch.no_grad():
        out = model(voltages, sigma_0, g, residual)

    delta_sigma = out["delta_sigma"]
    sigma_pred = out["sigma"]

    print("\n[Model Output]")
    print(f"  delta_sigma range: min={delta_sigma.min():.6f}, max={delta_sigma.max():.6f}")
    print(f"  delta_sigma std: {delta_sigma.std():.6f}")
    print(f"  delta_scale (config): {model.delta_scale}")
    print(f"  max possible delta_sigma: ±{model.delta_scale}")

    print(f"\n  sigma_pred range: min={sigma_pred.min():.6f}, max={sigma_pred.max():.6f}")
    print(f"  sigma_pred - sigma_0 max abs: {(sigma_pred - sigma_0).abs().max():.6f}")

    # Calculate RE
    re_coarse = (sigma_0 - target).norm(dim=1) / (target.norm(dim=1) + 1e-8)
    re_pred = (sigma_pred - target).norm(dim=1) / (target.norm(dim=1) + 1e-8)

    print("\n[Relative Error]")
    print(f"  RE(sigma_0): mean={re_coarse.mean():.4f}, std={re_coarse.std():.4f}")
    print(f"  RE(sigma_pred): mean={re_pred.mean():.4f}, std={re_pred.std():.4f}")
    print(f"  Improvement: {(re_coarse.mean() - re_pred.mean()) / re_coarse.mean() * 100:.2f}%")

    # The key diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    issues = []

    # Check 1: delta_scale too small
    if model.delta_scale < 0.01:
        issues.append(f"delta_scale={model.delta_scale} is very small, limiting model's correction capacity")

    # Check 2: sigma_0 already good
    if re_coarse.mean() < 0.2:
        issues.append(f"coarse RE={re_coarse.mean():.4f} is already quite good, may not need much correction")

    # Check 3: sigma_0 stuck at reference
    if sigma_0.std() < 1e-4:
        issues.append(f"sigma_0 std={sigma_0.std():.6f} is nearly zero - coarse reconstruction may be failing")

    # Check 4: g has no information
    if g.std() < 0.01:
        issues.append(f"g std={g.std():.6f} is very small - J^T r feature may not be informative")

    # Check 5: residual near zero
    if residual.norm() < 1e-4:
        issues.append("residual is near zero - coarse reconstruction already matches measurements perfectly")

    # Check 6: delta_sigma not changing
    if delta_sigma.std() < 1e-6:
        issues.append(f"delta_sigma std={delta_sigma.std():.6f} is near zero - model is not producing corrections")

    if issues:
        print("ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No obvious issues found. Check model architecture and training loop.")

    # Additional check: compare with expected correction magnitude
    expected_correction = (target - sigma_0).abs().mean()
    print(f"\nExpected correction magnitude: {expected_correction:.6f}")
    print(f"Max possible correction (delta_scale): {model.delta_scale}")
    print(f"Ratio: {expected_correction / model.delta_scale:.2f}x")

    if expected_correction > model.delta_scale * 2:
        print("  WARNING: Expected correction is much larger than delta_scale allows!")
        print("  Consider increasing delta_scale in config.")

if __name__ == "__main__":
    diagnose()
