"""
Complete fix and retrain script for residual EIT.

This script:
1. Checks current sigma_0 quality
2. Tests if BP or JAC is better
3. Recomputes residual features with the better method
4. Runs training with improved config

Usage:
    python fix_and_retrain.py [--skip-precompute]
"""

import os
import sys
import argparse
import subprocess

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def check_current_quality():
    """Check current sigma_0 quality."""
    import h5py
    import numpy as np

    print("\n" + "=" * 70)
    print("Step 1: Checking Current sigma_0 Quality")
    print("=" * 70)

    with h5py.File("data/generated/mixed_dataset.h5", 'r') as f:
        val = f['val']
        sigma_0 = val['sigma_0'][:]
        target = val['sigmas'][:]

        re = np.linalg.norm(sigma_0 - target, axis=1) / (np.linalg.norm(target, axis=1) + 1e-8)

        print(f"  Samples: {len(sigma_0)}")
        print(f"  RE(sigma_0): {re.mean():.4f} ± {re.std():.4f}")
        print(f"  RE range: [{re.min():.4f}, {re.max():.4f}]")

        # Check correction needed
        diff = target - sigma_0
        print(f"\n  Correction needed:")
        print(f"    Mean: {diff.abs().mean():.4f}")
        print(f"    Max: {diff.abs().max():.4f}")

        # Check how many need large correction
        over_0_02 = (diff.abs() > 0.02).mean()
        over_0_05 = (diff.abs() > 0.05).mean()
        print(f"    > 0.02: {over_0_02*100:.1f}%")
        print(f"    > 0.05: {over_0_05*100:.1f}%")

        return re.mean()


def test_reconstruction_methods():
    """Test BP vs JAC quality."""
    print("\n" + "=" * 70)
    print("Step 2: Testing BP vs JAC Reconstruction Quality")
    print("=" * 70)

    try:
        from data.eit_forward import EITForwardSolver
        from models.traditional import build_reconstructor
        from data.datasets.eit_dataset import EITDataset
        import numpy as np

        solver = EITForwardSolver("config/mesh_config.yaml")
        dataset = EITDataset("data/generated/mixed_dataset.h5", split="val", load_sigmas=True)

        v_ref = solver.V_uniform.astype(np.float32)
        if np.isnan(v_ref).any():
            v_ref = np.zeros_like(v_ref)

        # Test BP
        bp_recon = build_reconstructor(solver, method="bp")
        jac_recon = build_reconstructor(solver, method="jac")

        bp_res = []
        jac_res = []

        n_test = min(100, len(dataset))
        print(f"  Testing on {n_test} samples...")

        for i in range(n_test):
            sample = dataset[i]
            V_diff = sample["voltages"][0].numpy()
            target = sample["sigmas"].numpy()
            V_abs = V_diff + v_ref

            # BP
            sigma_bp, _ = bp_recon.reconstruct(V_abs)
            re_bp = np.linalg.norm(sigma_bp - target) / (np.linalg.norm(target) + 1e-8)
            bp_res.append(re_bp)

            # JAC
            sigma_jac, _ = jac_recon.reconstruct(V_abs)
            re_jac = np.linalg.norm(sigma_jac - target) / (np.linalg.norm(target) + 1e-8)
            jac_res.append(re_jac)

        bp_mean = np.mean(bp_res)
        jac_mean = np.mean(jac_res)

        print(f"\n  BP RE:  {bp_mean:.4f}")
        print(f"  JAC RE: {jac_mean:.4f}")

        if jac_mean < bp_mean:
            print(f"  → JAC is better by {(bp_mean - jac_mean) / bp_mean * 100:.1f}%")
            return "jac"
        else:
            print(f"  → BP is better by {(jac_mean - bp_mean) / jac_mean * 100:.1f}%")
            return "bp"

    except Exception as e:
        print(f"  Error: {e}")
        print("  → Defaulting to BP")
        return "bp"


def recompute_features(method="bp"):
    """Recompute residual features."""
    print("\n" + "=" * 70)
    print(f"Step 3: Recomputing Residual Features (method={method})")
    print("=" * 70)

    cmd = [
        "python", "data/precompute_residual_features.py",
        "--h5", "data/generated/mixed_dataset.h5",
        "--jacobian", "data/generated/jacobian.npy",
        "--method", method,
        "--force",
    ]

    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def train():
    """Run training."""
    print("\n" + "=" * 70)
    print("Step 4: Training ResidualEIT Model")
    print("=" * 70)

    cmd = ["python", "train_residual_eit.py"]
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Fix and retrain residual EIT")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip recomputing features")
    parser.add_argument("--method", default=None, choices=["bp", "jac", "auto"], help="Traditional reconstruction method")
    args = parser.parse_args()

    print("=" * 70)
    print("RESIDUAL EIT FIX AND RETRAIN")
    print("=" * 70)
    print("\nConfiguration changes:")
    print("  - delta_scale: 0.02 → 0.10")
    print("  - sigma_max: 0.1 → 0.15")
    print("  - batch_size: 8 → 16")
    print("  - learning_rate: 3e-4 → 5e-4")
    print("  - residual_measurement weight: 1.0 → 0.5")

    # Step 1: Check current quality
    current_re = check_current_quality()

    # Step 2: Determine best method
    if args.method:
        method = args.method
        print(f"\n  Using specified method: {method}")
    else:
        method = test_reconstruction_methods()

    # Step 3: Recompute features
    if not args.skip_precompute:
        success = recompute_features(method)
        if not success:
            print("\n  ERROR: Failed to recompute features")
            return

    # Step 4: Train
    success = train()

    if success:
        print("\n" + "=" * 70)
        print("COMPLETE!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("FAILED!")
        print("=" * 70)


if __name__ == "__main__":
    main()
