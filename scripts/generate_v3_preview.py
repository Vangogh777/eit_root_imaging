"""Generate preview images for the enhanced v3 dataset.
Shows each shape type with: GT sigma (FEM mesh) + boundary voltage (1×208).
"""
import os, sys, h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

H5_PATH = "data/generated/mixed_dataset.h5"
OUT_DIR = "results/dataset_preview_v3"
os.makedirs(OUT_DIR, exist_ok=True)

f = h5py.File(H5_PATH, 'r')
nodes = f['metadata/mesh_nodes'][:, :2]
elems = f['metadata/mesh_elements'][:]
triang = Triangulation(nodes[:, 0], nodes[:, 1], elems)

# Extract shape metadata from val set (500 samples — check each sample's sigma pattern)
val_sigmas = f['val/sigmas'][:]
val_masks = f['val/masks'][:]
centers = np.mean(nodes[elems], axis=1)

# Classify each val sample into shape type by analyzing its mask
def classify_shape(mask, centers, sigma):
    """Heuristic shape classification based on mask geometry."""
    inc = centers[mask > 0.5]
    if len(inc) < 5:
        return "unknown"
    cx, cy = inc.mean(axis=0)
    dx, dy = inc[:, 0] - cx, inc[:, 1] - cy
    # Check ring: hole in middle
    dist_from_center = np.sqrt(dx**2 + dy**2)
    # Multiple clusters?
    # Aspect ratio
    _, s, _ = np.linalg.svd(np.cov(np.stack([dx, dy])))
    aspect = s[0] / (s[1] + 1e-8)
    edge_dist = 0.10 - np.sqrt(cx**2 + cy**2)
    
    # Check if it's a ring: count elements near center (within inner radius)
    inner_radius = dist_from_center.max() * 0.4
    inner_count = (dist_from_center < inner_radius).sum()
    total_count = len(inc)
    ring_ratio = inner_count / total_count if total_count > 0 else 0
    
    if ring_ratio < 0.15 and total_count > 100:
        return "ring"
    elif edge_dist < 0.025:
        return "near_boundary"
    elif aspect > 1.6:
        return "ellipse"
    elif total_count > 400:
        # Double circle tends to have more elements
        return "double_circle"
    else:
        return "circle"

# Map shape names from mask analysis (simpler: just sample evenly from val set)
# Actually easier: sample specific indices from each group
# We know the val set has ~500 samples with diverse shapes
# Let's look for specific shapes by analyzing the mask

print("Classifying val samples...")
shape_samples = {}
for i in range(min(500, len(val_sigmas))):
    shape = classify_shape(val_masks[i], centers, val_sigmas[i])
    if shape not in shape_samples:
        shape_samples[shape] = []
    shape_samples[shape].append(i)

print(f"Found shapes: { {k: len(v) for k, v in shape_samples.items()} }")

# Pick 2 samples from each shape category
SHAPE_LABELS = {
    'circle': 'Circle',
    'ellipse': 'Ellipse', 
    'double_circle': 'Double Circle',
    'ring': 'Ring',
    'near_boundary': 'Near-Boundary',
}

vmin, vmax = 0.01, 0.10  # BG=0.01, inclusion up to 0.10 (10x)
cmap_sigma = 'viridis'

for shape, label in SHAPE_LABELS.items():
    indices = shape_samples.get(shape, [])
    if len(indices) < 2:
        # Use first sample and duplicate
        indices = indices * 2 if indices else [0, 1]
    
    n_show = min(2, len(indices))
    fig, axes = plt.subplots(n_show, 2, figsize=(10, 4 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices[:n_show]):
        sigma = val_sigmas[idx]
        voltage = f['val/voltages'][idx, 0, :]  # (208,)
        
        # Left: GT sigma on FEM mesh
        ax_s = axes[row, 0]
        im = ax_s.tripcolor(triang, facecolors=sigma, cmap=cmap_sigma,
                            vmin=vmin, vmax=vmax, shading='flat')
        ax_s.set_aspect('equal')
        ax_s.set_xlim(-0.105, 0.105)
        ax_s.set_ylim(-0.105, 0.105)
        ax_s.set_title(f'{label} #{row+1} — σ Map', fontsize=11)
        ax_s.axis('off')
        plt.colorbar(im, ax=ax_s, fraction=0.08, pad=0.04,
                     label='Conductivity (S/m)')
        
        # Right: boundary voltage (1×208)
        ax_v = axes[row, 1]
        ax_v.plot(voltage, 'b-', linewidth=1.0, alpha=0.8)
        # Add vertical lines for excitation boundaries
        for e in range(1, 16):
            ax_v.axvline(e * 13, color='gray', linestyle='--', linewidth=0.4, alpha=0.4)
        ax_v.set_xlim(0, 208)
        ax_v.set_title(f'{label} #{row+1} — Boundary Voltage (208 ch)', fontsize=11)
        ax_v.set_xlabel('Measurement Channel', fontsize=9)
        ax_v.set_ylabel('Differential Voltage (V)', fontsize=9)
        ax_v.grid(True, alpha=0.2)
        ax_v.tick_params(labelsize=8)
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{shape}_preview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# Summary grid: one sample of each shape
n_shapes = len(SHAPE_LABELS)
fig, axes = plt.subplots(n_shapes, 2, figsize=(10, 3 * n_shapes))
for row, (shape, label) in enumerate(SHAPE_LABELS.items()):
    indices = shape_samples.get(shape, [0])
    idx = indices[0]
    sigma = val_sigmas[idx]
    voltage = f['val/voltages'][idx, 0, :]
    
    ax_s = axes[row, 0]
    im = ax_s.tripcolor(triang, facecolors=sigma, cmap=cmap_sigma,
                        vmin=vmin, vmax=vmax, shading='flat')
    ax_s.set_aspect('equal')
    ax_s.set_xlim(-0.105, 0.105)
    ax_s.set_ylim(-0.105, 0.105)
    ax_s.set_title(label, fontsize=10, fontweight='bold')
    ax_s.axis('off')
    if row == 0:
        plt.colorbar(im, ax=ax_s, fraction=0.08, pad=0.04, label='σ (S/m)')
    
    ax_v = axes[row, 1]
    ax_v.plot(voltage, 'b-', linewidth=0.8, alpha=0.7)
    for e in range(1, 16):
        ax_v.axvline(e * 13, color='gray', linestyle='--', linewidth=0.3, alpha=0.3)
    ax_v.set_xlim(0, 208)
    ax_v.set_ylabel('V', fontsize=8)
    ax_v.tick_params(labelsize=7)
    ax_v.grid(True, alpha=0.15)
    if row == 0:
        ax_v.set_title('Boundary Voltage (208 ch)', fontsize=10)

axes[0, 0].set_ylabel('GT σ', fontsize=10, fontweight='bold')
plt.tight_layout()
grid_path = os.path.join(OUT_DIR, 'all_shapes_grid.png')
fig.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {grid_path}")

# Also generate near-boundary + high-noise comparison
for noise_label, noise_split in [('low_noise_-30dB', 'test_low_noise'),
                                   ('high_noise_-15dB', 'test_high_noise'),
                                   ('near_boundary_-25dB', 'test_near_boundary')]:
    if noise_split not in f:
        continue
    sigmas = f[f'{noise_split}/sigmas'][:3]
    voltages = f[f'{noise_split}/voltages'][:, 0, :]
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for i in range(3):
        ax_s = axes[i, 0]
        im = ax_s.tripcolor(triang, facecolors=sigmas[i], cmap=cmap_sigma,
                            vmin=vmin, vmax=vmax, shading='flat')
        ax_s.set_aspect('equal')
        ax_s.set_xlim(-0.105, 0.105)
        ax_s.set_ylim(-0.105, 0.105)
        ax_s.axis('off')
        if i == 0:
            ax_s.set_title(f'{noise_label} — σ Map', fontsize=10)
        
        ax_v = axes[i, 1]
        ax_v.plot(voltages[i], 'b-', linewidth=0.8, alpha=0.7)
        for e in range(1, 16):
            ax_v.axvline(e * 13, color='gray', linestyle='--', linewidth=0.3, alpha=0.3)
        ax_v.set_xlim(0, 208)
        ax_v.tick_params(labelsize=7)
        ax_v.grid(True, alpha=0.15)
        if i == 0:
            ax_v.set_title(f'{noise_label} — Voltage (208 ch)', fontsize=10)
    
    plt.tight_layout()
    noise_path = os.path.join(OUT_DIR, f'{noise_split}_preview.png')
    fig.savefig(noise_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {noise_path}")

f.close()
print(f"\nAll previews saved to {OUT_DIR}/")
