"""
Per-Shape EIT Dataset Generator (11466-element fine mesh)
=========================================================
Generate separate HDF5 dataset per shape type for high-resolution training.

Shape types: circle, ellipse, double_circle, square, ring, near_boundary

Usage:
  # Preview mode: 5 samples per shape → visualization images
  python data/generate_per_shape_datasets.py --preview

  # Full generation: 20000 train + 500 val + 500 test per shape
  python data/generate_per_shape_datasets.py --shape circle
  python data/generate_per_shape_datasets.py --shape ellipse
  python data/generate_per_shape_datasets.py --all

Features:
  - 11466-element fine mesh (mesh_resolution=0.0025)
  - Uniform background (0.01 S/m)
  - Random position, size, contrast (3x~10x, same as mixed dataset)
  - Edge/center balanced sampling (50/50)
  - Multi-frequency voltages (6 frequencies × 208 measurements)
  - Multi-worker parallel generation
"""

import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.eit_forward import EITForwardSolver

# ─── Constants ────────────────────────────────────────────
DOMAIN_RADIUS = 0.10
BG_SIGMA = 0.01
CONFIG_PATH = "config/mesh_11466_config.yaml"
OUTPUT_DIR = "data/generated"

SHAPE_TYPES = ['circle', 'ellipse', 'double_circle', 'square', 'near_boundary']

# ─── Position Sampling (from generate_mixed_dataset.py) ───

def _sample_position(rng, max_r, edge_ratio=0.5, edge_threshold=0.05):
    """Sample random position within domain, edge/center balanced."""
    while True:
        angle = rng.uniform(0, 2 * np.pi)
        max_dist = DOMAIN_RADIUS - max_r - 0.003
        lo = min(edge_threshold, max_dist * 0.95)
        if rng.random() < edge_ratio:
            dist = rng.uniform(lo, max_dist)  # edge
        else:
            dist = rng.uniform(0, min(0.04, max_dist))  # center
        cx, cy = dist * np.cos(angle), dist * np.sin(angle)
        if np.sqrt(cx**2 + cy**2) + max_r < DOMAIN_RADIUS - 0.003:
            return cx, cy


def _sample_position_near_boundary(rng, max_r):
    """Forced near-boundary sampling (edge_dist < 0.025m)."""
    for _ in range(50):
        angle = rng.uniform(0, 2 * np.pi)
        max_allowed = DOMAIN_RADIUS - max_r - 0.003
        min_allowed = max(0.0, DOMAIN_RADIUS - max_r - 0.025)
        if min_allowed >= max_allowed:
            min_allowed = max_allowed * 0.7
        dist = rng.uniform(min_allowed, max_allowed)
        cx, cy = dist * np.cos(angle), dist * np.sin(angle)
        edge_dist = DOMAIN_RADIUS - np.sqrt(cx**2 + cy**2) - max_r
        if 0.003 < edge_dist < 0.025:
            return cx, cy
    return _sample_position(rng, max_r)


# ─── Shape Generators ────────────────────────────────────

def _gen_circle(rng, centers_np):
    r = rng.uniform(0.008, 0.030)
    cx, cy = _sample_position(rng, r)
    dist = np.sqrt((centers_np[:, 0] - cx)**2 + (centers_np[:, 1] - cy)**2)
    return dist < r, cx, cy, r, 'circle'


def _gen_ellipse(rng, centers_np):
    a = rng.uniform(0.015, 0.040)
    b = rng.uniform(0.008, 0.020)
    max_r = max(a, b)
    cx, cy = _sample_position(rng, max_r)
    angle = rng.uniform(0, np.pi)
    dx = centers_np[:, 0] - cx
    dy = centers_np[:, 1] - cy
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    return (rx / a)**2 + (ry / b)**2 <= 1.0, cx, cy, max_r, 'ellipse'


def _gen_double_circle(rng, centers_np):
    mask = np.zeros(len(centers_np), dtype=bool)
    positions = []
    for _ in range(2):
        for _2 in range(20):
            r = rng.uniform(0.012, 0.028)
            cx, cy = _sample_position(rng, r)
            ok = True
            for px, py, pr in positions:
                if np.sqrt((cx - px)**2 + (cy - py)**2) < pr + r + 0.002:
                    ok = False
                    break
            if ok:
                break
        positions.append((cx, cy, r))
        dist = np.sqrt((centers_np[:, 0] - cx)**2 + (centers_np[:, 1] - cy)**2)
        mask |= dist < r
    return mask, cx, cy, max(r for _, _, r in positions), 'double_circle'


def _gen_square(rng, centers_np):
    half = rng.uniform(0.01, 0.035)
    max_r = half * np.sqrt(2)
    cx, cy = _sample_position(rng, max_r)
    angle = rng.uniform(0, np.pi / 4)
    dx = centers_np[:, 0] - cx
    dy = centers_np[:, 1] - cy
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    return (np.abs(rx) <= half) & (np.abs(ry) <= half), cx, cy, max_r, 'square'


def _gen_ring(rng, centers_np):
    outer_r = rng.uniform(0.020, 0.040)
    inner_r = outer_r * rng.uniform(0.3, 0.6)
    cx, cy = _sample_position(rng, outer_r)
    dist = np.sqrt((centers_np[:, 0] - cx)**2 + (centers_np[:, 1] - cy)**2)
    return (dist < outer_r) & (dist > inner_r), cx, cy, outer_r, 'ring'


def _gen_near_boundary(rng, centers_np):
    if rng.random() < 0.5:
        r = rng.uniform(0.008, 0.025)
        cx, cy = _sample_position_near_boundary(rng, r)
        dist = np.sqrt((centers_np[:, 0] - cx)**2 + (centers_np[:, 1] - cy)**2)
        mask = dist < r
        return mask, cx, cy, r, 'circle_near_boundary'
    else:
        a = rng.uniform(0.015, 0.035)
        b = rng.uniform(0.008, 0.018)
        max_r = max(a, b)
        cx, cy = _sample_position_near_boundary(rng, max_r)
        angle = rng.uniform(0, np.pi)
        dx = centers_np[:, 0] - cx
        dy = centers_np[:, 1] - cy
        rx = dx * np.cos(angle) + dy * np.sin(angle)
        ry = -dx * np.sin(angle) + dy * np.cos(angle)
        mask = (rx / a)**2 + (ry / b)**2 <= 1.0
        return mask, cx, cy, max_r, 'ellipse_near_boundary'


SHAPE_GENERATORS = {
    'circle': _gen_circle,
    'ellipse': _gen_ellipse,
    'double_circle': _gen_double_circle,
    'square': _gen_square,
    'ring': _gen_ring,
    'near_boundary': _gen_near_boundary,
}

# ─── Workers ─────────────────────────────────────────────

_solver = None


def _worker_init(config_path):
    global _solver
    _solver = EITForwardSolver(config_path)


def _generate_one(seed, target_shape, fixed_contrast=None, use_absolute=False):
    """Generate one sample of the specified shape type.

    Args:
        seed: random seed
        target_shape: shape type name
        fixed_contrast: if set (e.g. 5.0), use fixed contrast ratio instead of random 3x~10x
        use_absolute: if True, return absolute voltage (for preview); default differential (for training)
    """
    global _solver
    rng = np.random.RandomState(seed)
    centers = _solver.element_centers
    n_elems = _solver.n_elems

    gen_fn = SHAPE_GENERATORS[target_shape]
    mask, cx, cy, max_r, shape_name = gen_fn(rng, centers)

    # Contrast: fixed or random 3x~10x
    if fixed_contrast is not None:
        contrast = float(fixed_contrast)
    else:
        contrast = rng.uniform(3.0, 10.0)
    inc_sigma = BG_SIGMA * contrast

    sigma = np.full(n_elems, BG_SIGMA, dtype=np.float32)
    sigma[mask] = inc_sigma

    # Full FEM forward solve — differential voltage for training, absolute for preview
    if use_absolute:
        V_abs = _solver.solve_current(sigma)
        V = np.tile(V_abs, (len(_solver.frequencies), 1)).astype(np.float32)
    else:
        V = _solver.solve_multi_frequency(sigma)  # differential: v - V_uniform
    noise_db = rng.uniform(-40, -20)
    V_noisy = _solver.add_noise(V, noise_db)

    return {
        'sigma': sigma,
        'mask': mask.astype(np.float32),
        'voltage': V_noisy.astype(np.float32),
        'noise_db': noise_db,
        'shape': shape_name,
        'contrast': contrast,
        'cx': float(cx), 'cy': float(cy),
    }


def _generate_split(name, n, start_seed, target_shape, workers, config_path, fixed_contrast=None, use_absolute=False):
    """Generate n samples of target_shape for a data split."""
    print(f"  {name}: {n} samples...")
    gen_args = [(start_seed + i, target_shape, fixed_contrast, use_absolute) for i in range(n)]

    if workers > 1:
        n_proc = min(workers, cpu_count(), n)
        with Pool(n_proc, initializer=_worker_init, initargs=(config_path,)) as pool:
            results = list(tqdm(pool.starmap(_generate_one, gen_args), total=n, desc=f"    {name}"))
    else:
        _worker_init(config_path)
        results = [_generate_one(s, target_shape, fixed_contrast, use_absolute)
                   for s in tqdm(range(start_seed, start_seed + n), desc=f"    {name}")]

    sigmas = np.stack([r['sigma'] for r in results])
    masks = np.stack([r['mask'] for r in results])
    voltages = np.stack([r['voltage'] for r in results])
    noise_dbs = np.array([r['noise_db'] for r in results], dtype=np.float32)
    return sigmas, masks, voltages, noise_dbs, results


# ─── Preview Generation ─────────────────────────────────

def generate_preview(samples_per_shape=5, seed=42, contrast=5.0):
    """Generate preview images: sigma + voltage side-by-side, fixed contrast, save to results/dataset_preview_v4/."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from matplotlib.patches import Circle as MplCircle

    preview_dir = "results/dataset_preview_v4"
    os.makedirs(preview_dir, exist_ok=True)

    solver = EITForwardSolver(CONFIG_PATH)
    mesh_nodes = solver.mesh.node
    mesh_elements = solver.mesh.element
    centers = solver.element_centers[:, :2]
    n_elems = solver.n_elems
    n_meas = solver.n_measurements
    freqs = solver.frequencies

    # Build Matplotlib Triangulation once (no edge lines = smooth appearance)
    triang = Triangulation(mesh_nodes[:, 0], mesh_nodes[:, 1], mesh_elements)
    print(f"Preview mode: {samples_per_shape} samples × {len(SHAPE_TYPES)} shapes | contrast={contrast}x")
    print(f"Mesh: {n_elems} elements, {solver.mesh.node.shape[0]} nodes")

    _worker_init(CONFIG_PATH)

    all_results = {}

    for shape_name in SHAPE_TYPES:
        print(f"\nGenerating {shape_name} previews...")
        results = []
        for i in range(samples_per_shape):
            r = _generate_one(seed + 1000 * SHAPE_TYPES.index(shape_name) + i,
                            shape_name, fixed_contrast=contrast, use_absolute=True)
            results.append(r)
        all_results[shape_name] = results

        # Per-sample grid: 2 rows (sigma, voltage) × N columns
        fig, axes = plt.subplots(2, samples_per_shape,
                                 figsize=(samples_per_shape * 2.5, 5.5))
        if samples_per_shape == 1:
            axes = axes.reshape(2, 1)

        for i, r in enumerate(results):
            # Row 0: Sigma (conductivity) — proper Triangulation, no edge lines
            ax_s = axes[0, i]
            sigma = r['sigma']
            im = ax_s.tripcolor(triang, facecolors=sigma,
                                cmap='viridis', vmin=0.0, vmax=0.06,
                                shading='flat', edgecolors='none')
            boundary = MplCircle((0, 0), DOMAIN_RADIUS, fill=False, color='white', linewidth=0.8)
            ax_s.add_patch(boundary)
            ax_s.set_xlim(-0.11, 0.11); ax_s.set_ylim(-0.11, 0.11)
            ax_s.set_aspect('equal')
            ax_s.set_title(f'{r["shape"]} | contrast={r["contrast"]:.1f}x', fontsize=8)
            ax_s.axis('off')

            # Row 1: Voltage (first 2 frequencies)
            ax_v = axes[1, i]
            V = r['voltage']  # (n_freq, n_meas)
            for fi in range(min(2, V.shape[0])):
                ax_v.plot(V[fi], linewidth=0.5, alpha=0.8,
                         label=f'{freqs[fi]} Hz' if freqs else f'freq{fi}')
            ax_v.set_xlabel('channel', fontsize=7)
            ax_v.set_ylabel('V', fontsize=7)
            ax_v.set_title(f'noise={r["noise_db"]:.0f}dB', fontsize=8)
            ax_v.tick_params(labelsize=6)
            if V.shape[0] > 1:
                ax_v.legend(fontsize=5, loc='upper right')

        plt.suptitle(f'{shape_name} — {n_elems} elements, {contrast}x contrast',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()

        img_path = os.path.join(preview_dir, f'{shape_name}_preview.png')
        fig.savefig(img_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        print(f"  Saved: {img_path}")

    # Summary grid: one sample per shape, sigma only
    fig, axes = plt.subplots(1, len(SHAPE_TYPES), figsize=(len(SHAPE_TYPES) * 3.0, 3.2))
    for i, shape_name in enumerate(SHAPE_TYPES):
        r = all_results[shape_name][0]
        sigma = r['sigma']
        ax = axes[i]
        im = ax.tripcolor(triang, facecolors=sigma,
                          cmap='viridis', vmin=0.0, vmax=0.06,
                          shading='flat', edgecolors='none')
        boundary = MplCircle((0, 0), DOMAIN_RADIUS, fill=False, color='white', linewidth=1.0)
        ax.add_patch(boundary)
        ax.set_xlim(-0.11, 0.11); ax.set_ylim(-0.11, 0.11)
        ax.set_aspect('equal')
        ax.set_title(f'{shape_name}', fontsize=12)
        ax.axis('off')

    plt.suptitle(f'EIT Per-Shape Dataset — {n_elems} elements, {contrast}x contrast, {samples_per_shape} samples/shape',
                 fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()

    summary_path = os.path.join(preview_dir, 'all_shapes_summary.png')
    fig.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"\nSummary grid saved: {summary_path}")

    # Also generate a voltage comparison: all shapes, one sample each
    fig, axes = plt.subplots(1, len(SHAPE_TYPES), figsize=(len(SHAPE_TYPES) * 3.0, 3.0))
    if len(SHAPE_TYPES) == 1:
        axes = [axes]
    for i, shape_name in enumerate(SHAPE_TYPES):
        r = all_results[shape_name][0]
        V = r['voltage']
        ax = axes[i]
        for fi in range(min(2, V.shape[0])):
            ax.plot(V[fi], linewidth=0.6, alpha=0.8,
                   label=f'{freqs[fi]} Hz' if freqs else f'freq{fi}')
        ax.set_xlabel('channel', fontsize=8)
        ax.set_title(f'{shape_name} voltages', fontsize=11)
        ax.tick_params(labelsize=7)
        if V.shape[0] > 1:
            ax.legend(fontsize=6)
    plt.suptitle(f'Voltage Comparison — {n_elems} elements, {contrast}x contrast',
                 fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    volt_path = os.path.join(preview_dir, 'all_voltages.png')
    fig.savefig(volt_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"Voltage comparison saved: {volt_path}")

    # Save meta info
    meta_info = {
        'n_elems': n_elems,
        'n_nodes': solver.mesh.node.shape[0],
        'mesh_resolution': 0.0025,
        'samples_per_shape': samples_per_shape,
        'shapes': SHAPE_TYPES,
        'bg_sigma': BG_SIGMA,
        'contrast': f'{contrast}x (fixed)',
        'noise_range_db': '-40~-20',
        'frequencies': list(freqs),
    }
    import json
    with open(os.path.join(preview_dir, 'meta.json'), 'w') as f:
        json.dump(meta_info, f, indent=2)

    print(f"\nPreview complete! {samples_per_shape * len(SHAPE_TYPES)} shapes in {preview_dir}/")
    print(f"View at: http://117.50.185.165/datasets/")
    return all_results


# ─── Full Dataset Generation ─────────────────────────────

def generate_shape_dataset(target_shape, n_train=20000, n_val=500, n_test=500,
                           workers=0, seed=42, fixed_contrast=None, use_absolute=False):
    """Generate full HDF5 dataset for a single shape type."""
    import h5py

    # Validate
    solver = EITForwardSolver(CONFIG_PATH)
    n_elems = solver.n_elems
    n_freq = len(solver.frequencies)
    n_meas = solver.n_measurements

    filename = f"{target_shape}_dataset_11466.h5"
    output_path = os.path.join(OUTPUT_DIR, filename)

    print(f"{'='*60}")
    print(f"Generating: {target_shape} dataset")
    print(f"Mesh: {n_elems} elements | Output: {output_path}")
    print(f"Split: train={n_train} / val={n_val} / test={n_test}")
    print(f"{'='*60}")

    train_s, train_m, train_v, train_n, train_meta = _generate_split(
        "train", n_train, seed, target_shape, workers, CONFIG_PATH, fixed_contrast, use_absolute)
    val_s, val_m, val_v, val_n, val_meta = _generate_split(
        "val", n_val, seed + n_train + 1000, target_shape, workers, CONFIG_PATH, fixed_contrast, use_absolute)
    test_s, test_m, test_v, test_n, test_meta = _generate_split(
        "test", n_test, seed + n_train + n_val + 2000, target_shape, workers, CONFIG_PATH, fixed_contrast, use_absolute)

    print(f"Saving to {output_path} ...")
    with h5py.File(output_path, 'w') as f:
        for name, sigmas, masks, voltages, noise_dbs in [
            ("train", train_s, train_m, train_v, train_n),
            ("val", val_s, val_m, val_v, val_n),
            ("test", test_s, test_m, test_v, test_n),
        ]:
            grp = f.create_group(name)
            grp.create_dataset('voltages', data=voltages, compression='gzip')
            grp.create_dataset('sigmas', data=sigmas, compression='gzip')
            grp.create_dataset('masks', data=masks, compression='gzip')
            grp.create_dataset('noise_db', data=noise_dbs)

        meta = f.create_group('metadata')
        meta.create_dataset('mesh_nodes', data=solver.mesh.node)
        meta.create_dataset('mesh_elements', data=solver.mesh.element)
        meta.create_dataset('frequencies', data=np.array(solver.frequencies))

    # Stats
    contrasts = [r['contrast'] for r in test_meta]
    shapes = [r['shape'] for r in test_meta]
    shape_dist = Counter(shapes)

    total_samples = n_train + n_val + n_test
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n✅ {target_shape} dataset complete!")
    print(f"   Total samples: {total_samples}")
    print(f"   File size: {file_size_mb:.0f} MB")
    print(f"   Test contrast range: [{min(contrasts):.1f}x, {max(contrasts):.1f}x]")
    print(f"   Output: {output_path}")
    return output_path


# ─── CLI ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-Shape EIT Dataset Generator (11466 elements)")
    parser.add_argument("--shape", type=str, default=None,
                        choices=['circle', 'ellipse', 'double_circle', 'square', 'near_boundary'],
                        help="Generate single shape dataset")
    parser.add_argument("--all", action="store_true",
                        help="Generate all 6 shape datasets")
    parser.add_argument("--preview", action="store_true",
                        help="Generate preview images only (5 per shape)")
    parser.add_argument("--preview_samples", type=int, default=5,
                        help="Number of preview samples per shape")
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--contrast", type=float, default=None,
                        help="Fixed contrast ratio (e.g. 5.0). Default: random 3x~10x")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0=sequential)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.preview:
        generate_preview(samples_per_shape=args.preview_samples, seed=args.seed,
                        contrast=args.contrast if args.contrast else 5.0)

    elif args.all:
        for shape in SHAPE_TYPES:
            print(f"\n{'='*60}")
            print(f"Shape [{SHAPE_TYPES.index(shape)+1}/{len(SHAPE_TYPES)}]: {shape}")
            print(f"{'='*60}")
            generate_shape_dataset(
                target_shape=shape,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
                workers=args.workers,
                seed=args.seed,
                fixed_contrast=args.contrast,
                use_absolute=False,  # training: differential voltage
            )

    elif args.shape:
        generate_shape_dataset(
            target_shape=args.shape,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            workers=args.workers,
            seed=args.seed,
            fixed_contrast=args.contrast,
            use_absolute=False,
        )

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python data/generate_per_shape_datasets.py --preview")
        print("  python data/generate_per_shape_datasets.py --shape circle")
        print("  python data/generate_per_shape_datasets.py --all --workers 8")
