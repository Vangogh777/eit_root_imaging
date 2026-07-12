"""
Test BP vs JAC traditional reconstruction quality.

This script compares the quality of coarse reconstruction (sigma_0)
produced by Back-Projection (BP) and Gauss-Newton Jacobian (JAC) methods.
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.eit_forward import EITForwardSolver
from data.datasets.eit_dataset import EITDataset
from models.traditional import build_reconstructor


def test_traditional_methods(n_samples: int = 100):
    """Test BP vs JAC reconstruction quality on validation set."""
    print("=" * 70)
    print("Testing Traditional Reconstruction Methods: BP vs JAC")
    print("=" * 70)

    # Load solver and dataset
    solver = EITForwardSolver("config/mesh_config.yaml")

    # Use validation set for testing
    dataset = EITDataset(
        "data/generated/mixed_dataset.h5",
        split="val",
        load_sigmas=True,
    )

    # Get reference values
    sigma_ref = solver.gt_cfg.get("conductivity_soil", 0.01)
    v_ref_abs = solver.V_uniform
    if np.isnan(v_ref_abs).any():
        print("  [WARN] V_uniform has NaN, using zero vector")
        v_ref_abs = np.zeros(solver.n_measurements, dtype=np.float32)
    v_ref_abs = v_ref_abs.astype(np.float32)

    # Build reconstructors
    print("\n[1] Initializing BP reconstructor...")
    bp_recon = build_reconstructor(solver, method="bp")

    print("\n[2] Initializing JAC reconstructor...")
    jac_recon = build_reconstructor(solver, method="jac")
    actual_jac_method = jac_recon.method  # May fallback to BP if JAC fails

    # Test on samples
    print(f"\n[3] Testing on {n_samples} samples from validation set...")

    bp_errors = []
    jac_errors = []
    bp_failed = 0
    jac_failed = 0

    # Sample indices
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for i in tqdm(indices, desc="Testing"):
        sample = dataset[i]
        V_diff = sample["voltages"][0].numpy()  # First frequency
        target = sample["sigmas"].numpy()

        # Convert to absolute voltage
        V_abs = V_diff + v_ref_abs

        # BP reconstruction
        try:
            sigma_bp, info_bp = bp_recon.reconstruct(V_abs)
            if info_bp.get("failed", False):
                bp_failed += 1
                sigma_bp = np.full_like(target, sigma_ref)
            re_bp = np.linalg.norm(sigma_bp - target) / (np.linalg.norm(target) + 1e-8)
            bp_errors.append(re_bp)
        except Exception as e:
            bp_failed += 1
            bp_errors.append(float("inf"))

        # JAC reconstruction
        try:
            sigma_jac, info_jac = jac_recon.reconstruct(V_abs)
            if info_jac.get("failed", False):
                jac_failed += 1
                sigma_jac = np.full_like(target, sigma_ref)
            re_jac = np.linalg.norm(sigma_jac - target) / (np.linalg.norm(target) + 1e-8)
            jac_errors.append(re_jac)
        except Exception as e:
            jac_failed += 1
            jac_errors.append(float("inf"))

    # Compute statistics
    bp_errors = np.array(bp_errors)
    jac_errors = np.array(jac_errors)

    # Remove inf values for mean calculation
    bp_mean = np.mean(bp_errors[bp_errors < float("inf")])
    jac_mean = np.mean(jac_errors[jac_errors < float("inf")])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nBP Method:")
    print(f"  Mean RE: {bp_mean:.4f}")
    print(f"  Std RE:  {np.std(bp_errors[bp_errors < float('inf')]):.4f}")
    print(f"  Min RE:  {np.min(bp_errors):.4f}")
    print(f"  Max RE:  {np.max(bp_errors[bp_errors < float('inf')]):.4f}")
    print(f"  Failed:  {bp_failed}/{n_samples} ({bp_failed/n_samples*100:.1f}%)")

    print(f"\nJAC Method (actual: {actual_jac_method}):")
    print(f"  Mean RE: {jac_mean:.4f}")
    print(f"  Std RE:  {np.std(jac_errors[jac_errors < float('inf')]):.4f}")
    print(f"  Min RE:  {np.min(jac_errors):.4f}")
    print(f"  Max RE:  {np.max(jac_errors[jac_errors < float('inf')]):.4f}")
    print(f"  Failed:  {jac_failed}/{n_samples} ({jac_failed/n_samples*100:.1f}%)")

    # Compare
    print("\n" + "-" * 50)
    print("COMPARISON")
    print("-" * 50)

    improvement = (bp_mean - jac_mean) / bp_mean * 100
    if jac_mean < bp_mean:
        print(f"JAC is BETTER: RE reduced by {improvement:.1f}%")
        print(f"  BP RE: {bp_mean:.4f} → JAC RE: {jac_mean:.4f}")
        recommended_method = "jac"
    else:
        print(f"BP is BETTER (or JAC failed)")
        print(f"  BP RE: {bp_mean:.4f}, JAC RE: {jac_mean:.4f}")
        recommended_method = "bp"

    print("\n" + "=" * 70)
    print(f"RECOMMENDED METHOD: {recommended_method}")
    print("=" * 70)

    return recommended_method, {
        "bp_mean_re": bp_mean,
        "jac_mean_re": jac_mean,
        "bp_failed_rate": bp_failed / n_samples,
        "jac_failed_rate": jac_failed / n_samples,
    }


def main():
    recommended_method, stats = test_traditional_methods(n_samples=200)

    # Print recommendation for precompute script
    print("\n" + "=" * 70)
    print("NEXT STEP")
    print("=" * 70)
    print(f"\nRun the following command to recompute residual features:")
    print(f"  python data/precompute_residual_features.py --method {recommended_method} --force")
    print("\nThen train:")
    print(f"  python train_residual_eit.py")


if __name__ == "__main__":
    main()