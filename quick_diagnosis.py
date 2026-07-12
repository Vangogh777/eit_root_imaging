"""
Quick diagnosis for validation RE issue.
Run this to understand why val RE doesn't change.
"""

import torch
import numpy as np

# Quick check
print("=== Quick Diagnosis ===")

# Check if val_loss in log is identical
with open('residual_training_v2.log', 'r') as f:
    lines = [l for l in f if 'val_loss=' in l and 'Epoch' in l]

print(f"Found {len(lines)} val_loss entries")

# Extract values
import re
vals = re.findall(r'val_loss=([0-9.]+)', ''.join(lines))
if vals:
    print(f"Unique val_loss values: {len(set(vals))}")
    print(f"All same? {len(set(vals)) == 1}")

# Key insight: val_RE is already much better than before
re_vals = re.findall(r'val_RE=([0-9.]+)', ''.join(lines))
if re_vals:
    print(f"\nVal RE values: {re_vals}")
    print(f"\nCompare with previous training:")
    print(f"  Before fix: val_RE = 2.0957 (constant)")
    print(f"  After fix:  val_RE = {re_vals[-1]} (much better!)")

    if len(set(re_vals)) == 1:
        print("\n⚠ Issue: Val RE still constant, but at a much lower value")
        print("  This suggests model converged quickly to a local optimum")
        print("  May need more epochs or different learning rate schedule")
    else:
        print("\n✓ Val RE is changing")