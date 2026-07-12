"""
Quick script to recompute residual features with JAC method.
Bypasses the test script and directly uses JAC.
"""

import os
import sys
import subprocess

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

def main():
    print("=" * 70)
    print("Recomputing Residual Features with JAC Method")
    print("=" * 70)

    # First, let's check if JAC is available
    print("\n[1] Checking if JAC method is available...")

    try:
        from data.eit_forward import EITForwardSolver
        from models.traditional import build_reconstructor

        solver = EITForwardSolver("config/mesh_config.yaml")
        recon = build_reconstructor(solver, method="jac")

        if recon.method == "jac":
            print("  ✓ JAC method is available and working")
            use_jac = True
        else:
            print(f"  ⚠ JAC initialization failed, fell back to {recon.method}")
            print("  Will use BP method instead")
            use_jac = False
    except Exception as e:
        print(f"  ⚠ Error testing JAC: {e}")
        print("  Will use BP method instead")
        use_jac = False

    # Decide which method to use
    method = "jac" if use_jac else "bp"
    print(f"\n[2] Will use method: {method}")

    # Run precompute script
    print("\n[3] Running precompute script...")
    cmd = [
        "python", "data/precompute_residual_features.py",
        "--h5", "data/generated/mixed_dataset.h5",
        "--jacobian", "data/generated/jacobian.npy",
        "--method", method,
        "--force",
    ]

    print(f"  Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n" + "=" * 70)
        print("SUCCESS: Residual features recomputed!")
        print("=" * 70)
        print("\nNext step: Run training")
        print("  python train_residual_eit.py")
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 70)
        print("ERROR: Precompute failed")
        print("=" * 70)
        print(f"\nReturn code: {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
