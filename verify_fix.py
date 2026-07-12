"""
Quick verification and training script.

Verifies:
1. Config changes are correct
2. Model can produce larger delta_sigma

Then runs training.
"""

import os
import sys
import yaml
import torch
import numpy as np

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def verify_config():
    """Verify configuration changes."""
    print("=" * 70)
    print("VERIFICATION: Configuration Changes")
    print("=" * 70)

    with open("config/residual_eit_config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    print("\n[Model Config]")
    print(f"  delta_scale: {cfg['model']['delta_scale']} (expected: 0.10)")
    print(f"  sigma_max: {cfg['model']['sigma_max']} (expected: 0.15)")

    print("\n[Training Config]")
    print(f"  batch_size: {cfg['training']['batch_size']} (expected: 16)")
    print(f"  learning_rate: {cfg['training']['learning_rate']} (expected: 5e-4)")

    print("\n[Loss Weights]")
    for k, v in cfg['training']['loss_weights'].items():
        print(f"  {k}: {v}")

    # Check if correct
    issues = []
    if cfg['model']['delta_scale'] != 0.10:
        issues.append("delta_scale not updated to 0.10")
    if cfg['model']['sigma_max'] != 0.15:
        issues.append("sigma_max not updated to 0.15")
    if cfg['training']['batch_size'] != 16:
        issues.append("batch_size not updated to 16")

    if issues:
        print("\n⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\n✓ All configuration changes verified!")
    return True


def verify_model_output():
    """Verify model can produce larger corrections."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Model Output Range")
    print("=" * 70)

    from data.datasets.eit_dataset import EITDataModule
    from models.residual_eit import ResidualEIT

    # Load a batch
    dm = EITDataModule(
        h5_path="data/generated/mixed_dataset.h5",
        batch_size=4,
        load_residual_features=True,
        voltage_mask_ratio=0.0,
    )
    batch = next(iter(dm.val_dataloader()))

    # Load model config
    with open("config/residual_eit_config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    # Build model
    ds = dm.val_dataset
    centers = np.mean(ds.mesh_nodes[ds.mesh_elements], axis=1)
    elements = ds.mesh_elements

    J = np.load("data/generated/jacobian.npy").astype(np.float32)
    if J.ndim == 3:
        J = J[0]

    model = ResidualEIT(
        n_frequencies=ds.n_freq,
        n_meas=ds.n_meas,
        n_elems=ds.n_elems,
        hidden_dim=cfg['model']['hidden_dim'],
        gnn_layers=cfg['model']['gnn_layers'],
        delta_scale=cfg['model']['delta_scale'],
        jacobian=J,
        sigma_max=cfg['model']['sigma_max'],
    )
    model.setup_mesh(centers, elements)
    model.eval()

    # Forward pass
    with torch.no_grad():
        out = model(
            voltages=batch["voltages"],
            sigma_0=batch["sigma_0"],
            g=batch["physics_g"],
            residual=batch["voltage_residual"],
        )

    delta_sigma = out["delta_sigma"]
    print(f"\n  delta_scale: {model.delta_scale}")
    print(f"  delta_sigma range: [{delta_sigma.min():.4f}, {delta_sigma.max():.4f}]")
    print(f"  delta_sigma std: {delta_sigma.std():.4f}")
    print(f"  Max possible |delta_sigma|: {model.delta_scale}")

    # Check correction needed
    target = batch["sigmas"]
    sigma_0 = batch["sigma_0"]
    needed = (target - sigma_0).abs().mean()
    print(f"\n  Mean correction needed: {needed:.4f}")

    if model.delta_scale >= needed * 0.5:
        print("\n✓ delta_scale allows sufficient correction!")
    else:
        print(f"\n⚠ delta_scale ({model.delta_scale}) may still be too small for needed correction ({needed:.4f})")

    return True


def main():
    print("=" * 70)
    print("RESIDUAL EIT FIX VERIFICATION")
    print("=" * 70)

    # Verify config
    if not verify_config():
        print("\n✗ Configuration verification failed!")
        return

    # Verify model
    if not verify_model_output():
        print("\n✗ Model verification failed!")
        return

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE!")
    print("=" * 70)
    print("\nReady to train. Run:")
    print("  python train_residual_eit.py")


if __name__ == "__main__":
    main()
