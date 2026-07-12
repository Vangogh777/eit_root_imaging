#!/usr/bin/env python3
"""
DiffEIT v4 评估 + 可视化脚本
=============================
加载训练好的 DiffEIT v4 扩散模型, 在测试集上评估 RE/CC/SSIM,
并生成对比图像保存到 results/ 目录供 serve_results.py 展示.

用法:
  python evaluate_diff_eit.py
  python evaluate_diff_eit.py --checkpoint checkpoints/<run_id>/best.pt
  python evaluate_diff_eit.py --n_samples 20 --n_steps 50
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.diff_eit import DiffEIT
from models.mesh_pooling import build_hierarchy
from data.datasets.eit_dataset import MemoryEITDataset
from data.eit_forward import EITForwardSolver


# ============ Metrics ============
def relative_error(pred, target):
    B = pred.shape[0]
    err = torch.norm(pred.view(B, -1) - target.view(B, -1), dim=1)
    norm = torch.norm(target.view(B, -1), dim=1)
    return (err / (norm + 1e-8)).cpu().numpy()

def correlation_coefficient(pred, target):
    B = pred.shape[0]
    pf = pred.view(B, -1)
    tf = target.view(B, -1)
    pc = pf - pf.mean(dim=1, keepdim=True)
    tc = tf - tf.mean(dim=1, keepdim=True)
    num = (pc * tc).sum(dim=1)
    den = torch.sqrt((pc ** 2).sum(dim=1) * (tc ** 2).sum(dim=1))
    return (num / (den + 1e-8)).cpu().numpy()

def compute_ssim_grid(pred_np, target_np, centers, grid_size=128):
    """插值到规则网格后计算 SSIM"""
    from scipy.interpolate import griddata
    from skimage.metrics import structural_similarity
    x = np.linspace(-0.1, 0.1, grid_size)
    y = np.linspace(-0.1, 0.1, grid_size)
    X, Y = np.meshgrid(x, y)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    inside = np.sqrt(grid_pts[:, 0]**2 + grid_pts[:, 1]**2) < 0.098
    ssim_vals = []
    for i in range(len(pred_np)):
        pred_grid = griddata(centers, pred_np[i], grid_pts, method='linear', fill_value=0.01)
        target_grid = griddata(centers, target_np[i], grid_pts, method='linear', fill_value=0.01)
        p_img = pred_grid.reshape(grid_size, grid_size)
        t_img = target_grid.reshape(grid_size, grid_size)
        lo, hi = min(p_img[inside].min(), t_img[inside].min()), max(p_img[inside].max(), t_img[inside].max())
        if hi - lo > 1e-8:
            p_img = (p_img - lo) / (hi - lo)
            t_img = (t_img - lo) / (hi - lo)
        try:
            s = structural_similarity(p_img, t_img, data_range=1.0, win_size=7)
            ssim_vals.append(s)
        except Exception:
            ssim_vals.append(0.0)
    return np.array(ssim_vals)


# ============ Visualization ============
def plot_comparison(centers, elements, target, pred, save_path, idx=0,
                    sigma_min=0.005, sigma_max=0.1):
    """绘制 GT / Pred / Error 三联对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ['Ground Truth σ', 'DiffEIT Pred σ', '|Pred - GT|']
    data_list = [target, pred, np.abs(pred - target)]
    cmaps = ['inferno', 'inferno', 'hot']

    for ax, title, data, cmap in zip(axes, titles, data_list, cmaps):
        # 插值到网格
        try:
            from scipy.interpolate import griddata
            x = np.linspace(-0.1, 0.1, 200)
            y = np.linspace(-0.1, 0.1, 200)
            X, Y = np.meshgrid(x, y)
            grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
            grid_vals = griddata(centers, data, grid_pts, method='linear', fill_value=0.01)
            img = grid_vals.reshape(200, 200)
            # 圆形 mask
            mask = np.sqrt(X**2 + Y**2) > 0.098
            img_masked = np.ma.masked_where(mask, img)
            im = ax.pcolormesh(X, Y, img_masked, cmap=cmap, shading='auto')
            # 电极位置
            angles = np.linspace(0, 2*np.pi, 17)[:16]
            ex, ey = 0.098 * np.cos(angles), 0.098 * np.sin(angles)
            ax.scatter(ex, ey, c='cyan', s=8, marker='o', alpha=0.6)
            ax.contour(X, Y, mask, levels=[0.5], colors='white', linewidths=0.5)
            plt.colorbar(im, ax=ax, fraction=0.046)
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot error: {e}", transform=ax.transAxes, ha='center')
        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.suptitle(f'DiffEIT v4 Reconstruction — Sample #{idx}', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_metric_distributions(re_list, cc_list, ssim_list, save_path):
    """绘制指标分布直方图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(re_list, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(np.median(re_list), color='red', linestyle='--', label=f'Median={np.median(re_list):.4f}')
    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Count')
    ax.set_title(f'RE Distribution (mean={np.mean(re_list):.4f})')
    ax.legend()

    ax = axes[1]
    ax.hist(cc_list, bins=30, color='darkgreen', edgecolor='white', alpha=0.85)
    ax.axvline(np.median(cc_list), color='red', linestyle='--', label=f'Median={np.median(cc_list):.4f}')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title(f'CC Distribution (mean={np.mean(cc_list):.4f})')
    ax.legend()

    ax = axes[2]
    ax.hist(ssim_list, bins=30, color='darkorange', edgecolor='white', alpha=0.85)
    ax.axvline(np.median(ssim_list), color='red', linestyle='--', label=f'Median={np.median(ssim_list):.4f}')
    ax.set_xlabel('SSIM')
    ax.set_title(f'SSIM Distribution (mean={np.mean(ssim_list):.4f})')
    ax.legend()

    plt.suptitle('DiffEIT v4 — Test Set Metrics Distribution', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_comparison(re_list, cc_list, save_path):
    """RE vs CC 散点图"""
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(re_list, cc_list, c=np.arange(len(re_list)), cmap='viridis',
                    alpha=0.6, edgecolors='none')
    ax.set_xlabel('Relative Error', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title('DiffEIT v4: RE vs CC (per sample)', fontsize=14)
    # 最佳区域 (左上)
    ax.axhline(np.median(cc_list), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(np.median(re_list), color='gray', linestyle='--', alpha=0.5)
    ax.text(0.05, 0.95, f'n={len(re_list)}\nmedian RE={np.median(re_list):.4f}\nmedian CC={np.median(cc_list):.4f}',
            transform=ax.transAxes, va='top', fontsize=10, family='monospace')
    plt.colorbar(sc, ax=ax, label='Sample Index')
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='DiffEIT v4 Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to DiffEIT v4 checkpoint (best.pt)')
    parser.add_argument('--data', type=str, default='data/generated/mixed_dataset.h5')
    parser.add_argument('--jacobian', type=str, default='data/generated/jacobian.npy')
    parser.add_argument('--mesh_config', type=str, default='config/mesh_config.yaml')
    parser.add_argument('--n_samples', type=int, default=50, help='Test samples to evaluate')
    parser.add_argument('--n_steps', type=int, default=50, help='DDIM sampling steps')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto: results/<run_id>_eval/)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch (1=one at a time for sampling)')
    parser.add_argument('--sigma_min', type=float, default=0.0, help='Sigma min for normalization')
    parser.add_argument('--sigma_max', type=float, default=0.12, help='Sigma max for normalization')
    parser.add_argument('--sigma_mean', type=float, default=0.5, help='Sigma mean for standardization')
    parser.add_argument('--sigma_std', type=float, default=0.25, help='Sigma std for standardization')
    args = parser.parse_args()

    # Auto-detect device
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"🔧 Device: {device}")

    # Set output directory
    if args.output_dir is None:
        run_id = os.path.basename(os.path.dirname(args.checkpoint))
        args.output_dir = f'results/{run_id}_eval'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output: {args.output_dir}")

    # ---- Load Mesh ----
    print("📐 Loading mesh...")
    import yaml
    with open(args.mesh_config) as f:
        mesh_cfg = yaml.safe_load(f)['mesh']
    solver = EITForwardSolver(args.mesh_config)
    centers = solver.element_centers  # (n_elems, 2/3)
    elements = solver.mesh.element     # (n_elems, 3) triangles

    n_elems = centers.shape[0]
    print(f"   n_elems={n_elems}, centers={centers.shape}")

    # ---- Load Jacobian ----
    print("📐 Loading Jacobian...")
    jacobian = np.load(args.jacobian)  # (n_freq, 208, n_elems) or (208, n_elems)
    if jacobian.ndim == 3:
        jacobian = jacobian[0]  # first frequency
    print(f"   J shape: {jacobian.shape}")

    # ---- Build Hierarchy ----
    print("🏗️  Building mesh hierarchy...")
    hierarchy = build_hierarchy(centers[:, :2], elements, n_levels=3, k_neighbors=16)
    for i, h in enumerate(hierarchy):
        print(f"   Level {i}: n_nodes={h['centers'].shape[0]}")

    # ---- Load Model ----
    print("🤖 Loading DiffEIT model...")
    model = DiffEIT(
        n_elems=n_elems,
        n_meas=208,
        hidden_dim=384,
        time_dim=256,
        voltage_dim=512,
        pos_dim=35,
        T=200,
        n_levels=3,
        dropout=0.1,
        schedule='cosine',
    )
    model.configure_sigma_stats(args.sigma_min, args.sigma_max,
                                 args.sigma_mean, args.sigma_std)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state = ckpt['model']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
    else:
        state = ckpt

    # Remove '_orig_mod.' prefix if present (from torch.compile)
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    # Restore sigma stats from checkpoint metadata if available
    if isinstance(ckpt, dict) and 'sigma_min' in ckpt:
        model.configure_sigma_stats(
            ckpt['sigma_min'], ckpt['sigma_max'],
            ckpt.get('sigma_mean', 0.5), ckpt.get('sigma_std', 0.25))
        print(f"   Restored sigma stats from checkpoint: "
              f"min={ckpt['sigma_min']:.4f}, max={ckpt['sigma_max']:.4f}, "
              f"mean={ckpt.get('sigma_mean', 0.5):.4f}, std={ckpt.get('sigma_std', 0.25):.4f}")
    else:
        print(f"   ⚠️ Checkpoint lacks sigma stats — using CLI defaults "
              f"(min={args.sigma_min}, max={args.sigma_max})")

    # Restore T from checkpoint if available
    if isinstance(ckpt, dict) and 'T' in ckpt:
        model.diffusion.T = ckpt['T']
        print(f"   Restored T={ckpt['T']} from checkpoint")

    model.setup_mesh(centers[:, :2], elements, jacobian, hierarchy, sigma_ref=0.01)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Params: {n_params:,}")
    print(f"   ✅ Model loaded")

    # ---- Load Test Data ----
    print("📦 Loading test data...")
    dataset = MemoryEITDataset(args.data, split='test')
    print(f"   Test samples: {len(dataset)}")

    n_eval = min(args.n_samples, len(dataset))
    indices = np.random.RandomState(42).choice(len(dataset), n_eval, replace=False)

    # ---- Evaluate ----
    print(f"\n🔬 Evaluating {n_eval} samples (DDIM steps={args.n_steps})...")
    all_re = []
    all_cc = []
    all_ssim = []
    results = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        sigma_gt = sample['sigmas'].float()  # already torch (n_elems,)
        voltage = sample['voltages'].float()  # already torch (6, 208) or (208,)

        # Run inference
        t_start = time.time()
        with torch.no_grad():
            sigma_pred = model.sample(voltage.to(device), n_steps=args.n_steps, n_samples=1)
        elapsed = time.time() - t_start
        sigma_pred = sigma_pred.cpu()

        # Metrics
        pred_batch = sigma_pred.unsqueeze(0)
        gt_batch = sigma_gt.unsqueeze(0)
        re = relative_error(pred_batch, gt_batch)[0]
        cc = correlation_coefficient(pred_batch, gt_batch)[0]

        # SSIM on grid
        try:
            ssim = compute_ssim_grid(
                sigma_pred.numpy()[None, :],
                sigma_gt.numpy()[None, :],
                centers[:, :2])[0]
        except Exception:
            ssim = 0.0

        all_re.append(re)
        all_cc.append(cc)
        all_ssim.append(ssim)
        results.append({
            'idx': int(idx), 'RE': float(re), 'CC': float(cc), 'SSIM': float(ssim),
            'pred_min': float(sigma_pred.min()), 'pred_max': float(sigma_pred.max()),
            'gt_min': float(sigma_gt.min()), 'gt_max': float(sigma_gt.max()),
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_eval}] idx={idx:4d}  RE={re:.4f}  CC={cc:.4f}  SSIM={ssim:.4f}  ({elapsed:.1f}s)")

    # ---- Summary ----
    re_arr = np.array(all_re)
    cc_arr = np.array(all_cc)
    ssim_arr = np.array(all_ssim)

    print(f"\n{'='*60}")
    print(f"DiffEIT v4 Evaluation Summary ({n_eval} samples)")
    print(f"{'='*60}")
    print(f"  RE:   mean={re_arr.mean():.4f}  median={np.median(re_arr):.4f}  "
          f"min={re_arr.min():.4f}  max={re_arr.max():.4f}")
    print(f"  CC:   mean={cc_arr.mean():.4f}  median={np.median(cc_arr):.4f}  "
          f"min={cc_arr.min():.4f}  max={cc_arr.max():.4f}")
    print(f"  SSIM: mean={ssim_arr.mean():.4f}  median={np.median(ssim_arr):.4f}  "
          f"min={ssim_arr.min():.4f}  max={ssim_arr.max():.4f}")
    print(f"{'='*60}")

    # ---- Save Metrics JSON ----
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    summary = {
        'model': 'DiffEIT_v4',
        'checkpoint': args.checkpoint,
        'n_elems': n_elems,
        'n_params': n_params,
        'n_test': n_eval,
        'ddim_steps': args.n_steps,
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'RE_mean': float(re_arr.mean()),
            'RE_median': float(np.median(re_arr)),
            'CC_mean': float(cc_arr.mean()),
            'CC_median': float(np.median(cc_arr)),
            'SSIM_mean': float(ssim_arr.mean()),
            'model': 'DiffEIT_v4',
            'status': 'EXPERIMENTAL — undertrained',
        },
        'RE': {'mean': float(re_arr.mean()), 'median': float(np.median(re_arr)),
               'min': float(re_arr.min()), 'max': float(re_arr.max())},
        'CC': {'mean': float(cc_arr.mean()), 'median': float(np.median(cc_arr)),
               'min': float(cc_arr.min()), 'max': float(cc_arr.max())},
        'SSIM': {'mean': float(ssim_arr.mean()), 'median': float(np.median(ssim_arr)),
                 'min': float(ssim_arr.min()), 'max': float(ssim_arr.max())},
        'per_sample': results,
    }
    with open(metrics_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📊 Metrics saved to {metrics_file}")

    # ---- Generate Visualizations ----
    print("\n🎨 Generating visualizations...")

    # 1. Metric distribution plots
    plot_metric_distributions(
        all_re, all_cc, all_ssim,
        os.path.join(args.output_dir, 'metric_distributions.png'))

    # 2. RE vs CC scatter
    plot_scatter_comparison(
        all_re, all_cc,
        os.path.join(args.output_dir, 're_vs_cc.png'))

    # 3. Best/worst/median samples
    best_idx = np.argmin(re_arr)
    worst_idx = np.argmax(re_arr)
    median_idx = np.argsort(re_arr)[len(re_arr) // 2]

    for tag, si in [('best', best_idx), ('median', median_idx), ('worst', worst_idx)]:
        idx = indices[si]
        sample = dataset[idx]
        sigma_gt = sample['sigmas'].numpy()
        voltage = sample['voltages']

        with torch.no_grad():
            sigma_pred = model.sample(
                voltage.to(device),
                n_steps=args.n_steps, n_samples=1).cpu().numpy()

        plot_comparison(
            centers[:, :2], elements,
            sigma_gt, sigma_pred,
            os.path.join(args.output_dir, f'recon_{tag}_idx{idx}.png'),
            idx=idx)

    # 4. Gallery: first 12 samples
    n_gallery = min(12, n_eval)
    fig, axes = plt.subplots(3, n_gallery, figsize=(3 * n_gallery, 9))
    if n_gallery == 1:
        axes = axes[:, None]  # ensure 2D

    for j, si in enumerate(range(n_gallery)):
        idx = indices[si]
        sample = dataset[idx]
        sigma_gt = sample['sigmas'].numpy()
        voltage = sample['voltages']

        with torch.no_grad():
            sigma_pred = model.sample(
                voltage.to(device),
                n_steps=args.n_steps, n_samples=1).cpu().numpy()

        row_data = [sigma_gt, sigma_pred, np.abs(sigma_pred - sigma_gt)]
        row_labels = ['GT', 'DiffEIT', 'Error']

        try:
            from scipy.interpolate import griddata
            x = np.linspace(-0.1, 0.1, 150)
            y = np.linspace(-0.1, 0.1, 150)
            X, Y = np.meshgrid(x, y)
            grd = np.stack([X.ravel(), Y.ravel()], axis=1)
            mask = np.sqrt(X**2 + Y**2) > 0.098
            for row_i, (data, label) in enumerate(zip(row_data, row_labels)):
                ax = axes[row_i, j]
                gv = griddata(centers[:, :2], data, grd, method='linear', fill_value=0.01)
                img = np.ma.masked_where(mask, gv.reshape(150, 150))
                ax.pcolormesh(X, Y, img, cmap='inferno' if row_i < 2 else 'hot', shading='auto')
                angles = np.linspace(0, 2*np.pi, 17)[:16]
                ax.scatter(0.098*np.cos(angles), 0.098*np.sin(angles), c='cyan', s=3, alpha=0.6)
                ax.set_aspect('equal')
                ax.axis('off')
                if j == 0:
                    ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=15)
                if row_i == 0:
                    ax.set_title(f'#{idx}', fontsize=9)
        except Exception as e:
            for row_i in range(3):
                axes[row_i, j].text(0.5, 0.5, 'err', transform=axes[row_i, j].transAxes, ha='center')

    plt.suptitle(f'DiffEIT v4 Reconstruction Gallery ({n_gallery} samples)', fontsize=14, y=1.01)
    plt.tight_layout()
    gallery_path = os.path.join(args.output_dir, 'gallery.png')
    fig.savefig(gallery_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"   Gallery saved to {gallery_path}")

    # 5. Create index.html for results server
    index_html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DiffEIT v4 Evaluation — {os.path.basename(args.output_dir)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1, h2 {{ color: #00d4ff; }}
  .metrics {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
  .card {{ background: #16213e; border-radius: 12px; padding: 20px; min-width: 200px; text-align: center; }}
  .card .value {{ font-size: 2em; font-weight: bold; color: #00d4ff; }}
  .card .label {{ color: #888; font-size: 0.9em; margin-top: 5px; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 10px 0; }}
  .section {{ margin: 30px 0; }}
  .hist {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .hist img {{ max-width: 32%; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ color: #00d4ff; }}
</style>
</head>
<body>
<h1>🔬 DiffEIT v4 Evaluation Results</h1>
<p>Checkpoint: <code>{args.checkpoint}</code></p>
<p>Run: {os.path.basename(args.output_dir)} | {n_eval} test samples | {args.n_steps} DDIM steps</p>

<div class="metrics">
  <div class="card">
    <div class="value">{re_arr.mean():.4f}</div>
    <div class="label">Mean RE ↓</div>
  </div>
  <div class="card">
    <div class="value">{np.median(re_arr):.4f}</div>
    <div class="label">Median RE ↓</div>
  </div>
  <div class="card">
    <div class="value">{cc_arr.mean():.4f}</div>
    <div class="label">Mean CC ↑</div>
  </div>
  <div class="card">
    <div class="value">{ssim_arr.mean():.4f}</div>
    <div class="label">Mean SSIM ↑</div>
  </div>
  <div class="card">
    <div class="value">{n_params/1e6:.1f}M</div>
    <div class="label">Parameters</div>
  </div>
</div>

<div class="section">
  <h2>Metric Distributions</h2>
  <img src="metric_distributions.png" alt="Distributions">
</div>

<div class="section">
  <h2>RE vs CC</h2>
  <img src="re_vs_cc.png" alt="RE vs CC">
</div>

<div class="section">
  <h2>Reconstruction Gallery</h2>
  <img src="gallery.png" alt="Gallery">
</div>

<div class="section">
  <h2>Best / Median / Worst Samples</h2>
  <div class="hist">
    <img src="recon_best_idx{indices[best_idx]}.png" alt="Best">
    <img src="recon_median_idx{indices[median_idx]}.png" alt="Median">
    <img src="recon_worst_idx{indices[worst_idx]}.png" alt="Worst">
  </div>
</div>

<div class="section">
  <h2>Per-Sample Metrics</h2>
  <table>
    <tr><th>Idx</th><th>RE</th><th>CC</th><th>SSIM</th><th>Pred Range</th><th>GT Range</th></tr>
"""
    for r in sorted(results, key=lambda x: x['RE']):
        index_html += f"""    <tr>
      <td>{r['idx']}</td>
      <td>{r['RE']:.4f}</td>
      <td>{r['CC']:.4f}</td>
      <td>{r['SSIM']:.4f}</td>
      <td>[{r['pred_min']:.4f}, {r['pred_max']:.4f}]</td>
      <td>[{r['gt_min']:.4f}, {r['gt_max']:.4f}]</td>
    </tr>\n"""
    index_html += """  </table>
</div>
</body>
</html>"""

    index_path = os.path.join(args.output_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(index_html)
    print(f"   Index saved to {index_path}")

    print(f"\n✅ Evaluation complete! Results in: {args.output_dir}/")
    print(f"   View: python serve_results.py  (then open http://localhost:8080)")


if __name__ == '__main__':
    main()
