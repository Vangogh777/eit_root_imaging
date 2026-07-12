"""
Complete analysis of why the residual EIT model is not learning properly.
"""

import os
import sys
import torch
import numpy as np
import h5py

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.datasets.eit_dataset import EITDataModule
from models.residual_eit import ResidualEIT, ResidualComputer
from training.residual_loss import ResidualMeasurementConsistencyLoss

def analyze_all():
    print("=" * 70)
    print("DETAILED ANALYSIS: Why Residual EIT is Not Learning")
    print("=" * 70)

    # Load data
    dm = EITDataModule(
        h5_path="data/generated/mixed_dataset.h5",
        batch_size=8,
        load_residual_features=True,
        voltage_mask_ratio=0.0,
    )
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))

    voltages = batch["voltages"]
    sigma_0 = batch["sigma_0"]
    g = batch["physics_g"]
    residual = batch["voltage_residual"]
    target = batch["sigmas"]

    # Load Jacobian
    J = np.load("data/generated/jacobian.npy").astype(np.float32)
    if J.ndim == 3:
        J = J[0]

    # Build model
    ds = dm.val_dataset
    centers = np.mean(ds.mesh_nodes[ds.mesh_elements], axis=1)
    elements = ds.mesh_elements

    model = ResidualEIT(
        n_frequencies=ds.n_freq,
        n_meas=ds.n_meas,
        n_elems=ds.n_elems,
        hidden_dim=256,
        gnn_layers=4,
        delta_scale=0.02,
        jacobian=J,
    )
    model.setup_mesh(centers, elements)

    # ===========================================
    # PROBLEM 1: Check if sigma_0 is the problem
    # ===========================================
    print("\n" + "=" * 70)
    print("PROBLEM 1: Quality of Traditional Reconstruction (sigma_0)")
    print("=" * 70)

    re_sigma_0 = (sigma_0 - target).norm(dim=-1) / (target.norm(dim=-1) + 1e-8)
    print(f"RE(sigma_0): mean={re_sigma_0.mean():.4f}, min={re_sigma_0.min():.4f}, max={re_sigma_0.max():.4f}")

    # Check if sigma_0 has meaningful structure
    sigma_0_std_per_sample = sigma_0.std(dim=1)
    target_std_per_sample = target.std(dim=1)
    print(f"sigma_0 std per sample: mean={sigma_0_std_per_sample.mean():.6f}")
    print(f"target std per sample: mean={target_std_per_sample.mean():.6f}")

    # ===========================================
    # PROBLEM 2: Check residual physics correctness
    # ===========================================
    print("\n" + "=" * 70)
    print("PROBLEM 2: Residual Physics Verification")
    print("=" * 70)

    sigma_ref = 0.01
    residual_computer = ResidualComputer(J, sigma_ref=sigma_ref)

    # Verify: r = V_diff - J @ (sigma_0 - sigma_ref)
    with torch.no_grad():
        g_recomputed, residual_recomputed = residual_computer(voltages, sigma_0)

    print(f"Original g vs recomputed g: max diff = {(g - g_recomputed).abs().max():.6f}")
    print(f"Original r vs recomputed r: max diff = {(residual - residual_recomputed).abs().max():.6f}")

    # Check residual magnitude
    print(f"\nResidual magnitude: mean L2 norm = {residual.norm(dim=-1).mean():.4f}")
    print(f"Relative residual (||r||/||V||): {(residual.norm(dim=-1) / voltages[:,0,:].norm(dim=-1)).mean():.4f}")

    # ===========================================
    # PROBLEM 3: Check the residual loss
    # ===========================================
    print("\n" + "=" * 70)
    print("PROBLEM 3: Residual Measurement Consistency Loss")
    print("=" * 70)

    # The loss is: ||J @ delta_sigma - r||^2
    # We want to understand what delta_sigma would minimize this loss

    # Minimum-norm solution: delta_sigma = J^T @ (J @ J^T)^{-1} @ r
    # But for now, let's just check the relationship between delta_sigma and r

    # The residual loss is normalized:
    loss_fn = ResidualMeasurementConsistencyLoss(J, normalize=True)

    with torch.no_grad():
        out = model(voltages, sigma_0, g, residual)

    delta_sigma = out["delta_sigma"]
    loss_value = loss_fn(delta_sigma, residual)
    print(f"Initial residual loss: {loss_value:.6f}")

    # What if delta_sigma = 0?
    loss_zero = loss_fn(torch.zeros_like(delta_sigma), residual)
    print(f"Loss with delta_sigma=0: {loss_zero:.6f}")

    # ===========================================
    # PROBLEM 4: Analyze delta_scale constraint
    # ===========================================
    print("\n" + "=" * 70)
    print("PROBLEM 4: delta_scale Constraint Analysis")
    print("=" * 70)

    # delta_sigma = delta_scale * tanh(raw_delta)
    # max possible |delta_sigma| = delta_scale = 0.02

    print(f"delta_scale = {model.delta_scale}")
    print(f"Max possible correction per element = ±{model.delta_scale}")

    # Required correction to go from sigma_0 to target
    required_correction = target - sigma_0
    required_correction_abs = required_correction.abs()

    print(f"\nRequired correction statistics:")
    print(f"  mean: {required_correction_abs.mean():.6f}")
    print(f"  max: {required_correction_abs.max():.6f}")
    print(f"  std: {required_correction.std():.6f}")

    # Check how many elements need correction > delta_scale
    over_limit = (required_correction_abs > model.delta_scale).float().mean()
    print(f"\nFraction of elements needing correction > delta_scale: {over_limit:.2%}")

    # ===========================================
    # PROBLEM 5: Supervised loss magnitude
    # ===========================================
    print("\n" + "=" * 70)
    print("PROBLEM 5: Supervised Loss Analysis")
    print("=" * 70)

    sigma_pred = out["sigma"]

    # RelativeMSELoss: ((pred - target)^2 / (target^2 + 1e-6)).mean()
    supervised_loss = ((sigma_pred - target).pow(2) / (target.pow(2) + 1e-6)).mean()
    print(f"Supervised loss (initial): {supervised_loss:.6f}")

    # What if sigma_pred = sigma_0?
    supervised_loss_sigma0 = ((sigma_0 - target).pow(2) / (target.pow(2) + 1e-6)).mean()
    print(f"Supervised loss (sigma_0 only): {supervised_loss_sigma0:.6f}")

    # MSE loss
    mse_loss = (sigma_pred - target).pow(2).mean()
    print(f"MSE loss (initial): {mse_loss:.6f}")

    # ===========================================
    # PROBLEM 6: Gradient flow analysis
    # ===========================================
    print("\n" + "=" * 70)
    print("PROBLEM 6: Gradient Flow Analysis")
    print("=" * 70)

    # Re-run with gradients
    model.train()
    out = model(voltages, sigma_0, g, residual)

    # Compute individual losses
    from training.residual_loss import RelativeMSELoss, ResidualSparsityLoss, ResidualSmoothnessLoss
    from training.loss import TVRegularizationLoss

    supervised_fn = RelativeMSELoss()
    residual_fn = ResidualMeasurementConsistencyLoss(J, normalize=True)
    sparsity_fn = ResidualSparsityLoss()
    smoothness_fn = ResidualSmoothnessLoss(model._edge_idx)

    L_sup = supervised_fn(out["sigma"], target)
    L_res = residual_fn(out["delta_sigma"], residual)
    L_sparse = sparsity_fn(out["delta_sigma"])
    L_smooth = smoothness_fn(out["delta_sigma"])

    print(f"L_sup (supervised): {L_sup:.6f}")
    print(f"L_res (residual meas): {L_res:.6f}")
    print(f"L_sparse: {L_sparse:.6f}")
    print(f"L_smooth: {L_smooth:.6f}")

    # Check gradients
    model.zero_grad()
    L_sup.backward(retain_graph=True)
    grad_norm_sup = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
    print(f"\nGradient norm from L_sup: {grad_norm_sup:.6f}")

    model.zero_grad()
    L_res.backward(retain_graph=True)
    grad_norm_res = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
    print(f"Gradient norm from L_res: {grad_norm_res:.6f}")

    # ===========================================
    # DIAGNOSIS SUMMARY
    # ===========================================
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []

    # Issue 1: delta_scale too restrictive
    if over_limit > 0.5:
        issues.append(f"CRITICAL: {over_limit:.0%} of elements need correction > delta_scale ({model.delta_scale})")

    # Issue 2: sigma_0 very poor
    if re_sigma_0.mean() > 1.5:
        issues.append(f"CRITICAL: Traditional reconstruction is very poor (RE={re_sigma_0.mean():.2f})")

    # Issue 3: Residual loss dominates
    if L_res > L_sup * 10:
        issues.append(f"WARNING: Residual loss ({L_res:.2f}) is much larger than supervised loss ({L_sup:.2f})")

    # Issue 4: Gradient too small
    if grad_norm_sup < 1e-4:
        issues.append(f"WARNING: Gradient from supervised loss is very small ({grad_norm_sup:.6f})")

    if issues:
        print("ISSUES IDENTIFIED:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No obvious issues identified.")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Why val_RE doesn't change")
    print("=" * 70)

    print("""
The validation RE (2.0957) is constant because:

1. The model's delta_sigma is bounded by tanh * delta_scale = ±0.02
   - This is a very small correction range

2. The required correction (sigma_0 → target) is much larger:
   - Mean required correction: {:.4f}
   - Max required correction: {:.4f}

3. The sigma_0 from BP reconstruction is very poor:
   - RE(sigma_0) = {:.2f} (should be < 0.5 for useful coarse reconstruction)

4. Even with perfect delta_sigma within the allowed range,
   the model can only achieve a tiny improvement.

5. The validation set is small (32 samples), and the model
   produces nearly identical outputs regardless of input.

RECOMMENDATIONS:
1. Increase delta_scale to 0.05-0.10 (allow larger corrections)
2. Improve traditional reconstruction (use JAC instead of BP)
3. Use larger validation set
4. Consider removing tanh bound, using simple clamping instead
5. Check if BP reconstruction is failing on most samples
""".format(
        required_correction_abs.mean(),
        required_correction_abs.max(),
        re_sigma_0.mean()
    ))

if __name__ == "__main__":
    analyze_all()
