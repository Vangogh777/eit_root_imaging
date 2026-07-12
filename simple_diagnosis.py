"""Simplified diagnosis of residual EIT issues."""

import os
import sys
import torch
import numpy as np

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

print("=" * 70)
print("SIMPLIFIED DIAGNOSIS: Residual EIT Training Issues")
print("=" * 70)

# Load just metadata from HDF5
import h5py

with h5py.File("data/generated/mixed_dataset.h5", 'r') as f:
    val_grp = f['val']
    n_val = val_grp['voltages'].shape[0]
    n_elems = val_grp['sigmas'].shape[1]
    print(f"\nValidation set: {n_val} samples, {n_elems} elements")

    # Load first few samples
    sigma_0 = val_grp['sigma_0'][:8]
    target = val_grp['sigmas'][:8]
    sigma_0 = torch.from_numpy(sigma_0)
    target = torch.from_numpy(target)

# Load Jacobian
J = np.load("data/generated/jacobian.npy").astype(np.float32)
if J.ndim == 3:
    J = J[0]
print(f"Jacobian shape: {J.shape}")

# ===========================================
# PROBLEM 1: delta_scale constraint
# ===========================================
print("\n" + "-" * 50)
print("PROBLEM 1: delta_scale Constraint")
print("-" * 50)

delta_scale = 0.02  # From config
required_correction = (target - sigma_0).abs()
print(f"delta_scale = {delta_scale}")
print(f"Required correction: mean={required_correction.mean():.4f}, max={required_correction.max():.4f}")

over_limit = (required_correction > delta_scale).float().mean()
print(f"Elements needing correction > delta_scale: {over_limit:.1%}")

# ===========================================
# PROBLEM 2: Coarse reconstruction quality
# ===========================================
print("\n" + "-" * 50)
print("PROBLEM 2: Coarse Reconstruction Quality")
print("-" * 50)

re_sigma_0 = (sigma_0 - target).norm(dim=-1) / (target.norm(dim=-1) + 1e-8)
print(f"RE(sigma_0): mean={re_sigma_0.mean():.4f}, min={re_sigma_0.min():.4f}, max={re_sigma_0.max():.4f}")

# Check sigma_0 value distribution
print(f"sigma_0 range: [{sigma_0.min():.4f}, {sigma_0.max():.4f}]")
print(f"target range: [{target.min():.4f}, {target.max():.4f}]")
print(f"sigma_ref (background): 0.01")

# Check if sigma_0 is stuck at sigma_ref
sigma_ref = 0.01
near_ref = ((sigma_0 - sigma_ref).abs() < 0.001).float().mean()
print(f"Elements with sigma_0 ≈ sigma_ref: {near_ref:.1%}")

# ===========================================
# PROBLEM 3: Loss weight balance
# ===========================================
print("\n" + "-" * 50)
print("PROBLEM 3: Loss Weight Analysis")
print("-" * 50)

# From config
weights = {
    'supervised': 1.0,
    'residual_measurement': 1.0,
    'tv': 0.03,
    'delta_l1': 0.01,
    'delta_smooth': 0.02,
}
print("Loss weights:", weights)

# ===========================================
# SUMMARY
# ===========================================
print("\n" + "=" * 70)
print("ROOT CAUSE ANALYSIS")
print("=" * 70)

print("""
## CRITICAL ISSUES IDENTIFIED:

### 1. delta_scale too restrictive (MOST CRITICAL)
   - delta_scale = 0.02 limits maximum correction to ±0.02 S/m
   - Required corrections often exceed this limit ({:.0%} of elements)
   - The model CANNOT learn the necessary corrections even if it wanted to!

### 2. Poor coarse reconstruction (sigma_0)
   - RE(sigma_0) = {:.2f} is extremely high (should be < 0.5)
   - This means the BP reconstruction is essentially failing
   - The coarse reconstruction is so bad that residual correction
     is an impossible task

### 3. Model is "stuck"
   - With such poor sigma_0, the model's small delta_sigma (±0.02)
     cannot make meaningful improvement
   - Validation metrics don't change because the model learns to
     output near-zero delta_sigma (the safest prediction)

## RECOMMENDATIONS (in priority order):

1. **Increase delta_scale**: Change from 0.02 to 0.05 or 0.10
   - Or remove tanh bound entirely, just use clamping

2. **Improve traditional reconstruction**: Use JAC instead of BP
   - Or implement Tikhonov/GN for better stability

3. **Check traditional reconstruction success rate**:
   - Many BP reconstructions may be failing and returning sigma_ref

4. **Add confidence weighting**:
   - If coarse reconstruction is poor, give more freedom to the network

5. **Consider alternative architecture**:
   - The "residual correction" paradigm assumes a reasonable coarse solution
   - If sigma_0 is garbage, the network cannot work
""".format(over_limit, re_sigma_0.mean()))
