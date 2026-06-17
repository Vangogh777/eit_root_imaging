#!/usr/bin/env python3
"""
Conv-Spatial EIT 完整验证脚本
==============================
在测试集上全面评估，输出指标 + 可视化。

用法:
    python evaluate_conv_spatial_full.py --checkpoint checkpoints/conv_spatial_best.pt
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.conv_spatial_eit import ConvSpatialEIT
from data.eit_forward import EITForwardSolver
from data.datasets.eit_dataset import MemoryEITDataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from scipy.interpolate import griddata


# ────────────────────── 指标函数 ──────────────────────

def compute_re(pred, target):
    return np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-8)

def compute_cc(pred, target):
    p = pred - pred.mean()
    g = target - target.mean()
    return (p * g).sum() / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-8)

def compute_ssim_on_mesh(pred, target, centers, resolution=128):
    centers_2d = centers[:, :2] if centers.shape[1] > 2 else centers
    radius = np.max(np.abs(centers_2d)) * 1.1
    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(x, y)
    pred_img = griddata(centers_2d, pred, (X, Y), method='linear', fill_value=0)
    target_img = griddata(centers_2d, target, (X, Y), method='linear', fill_value=0)
    mask = (X**2 + Y**2) <= radius**2
    pred_img[~mask] = 0; target_img[~mask] = 0
    dr = max(target_img.max() - target_img.min(), 1e-8)
    return ssim(pred_img, target_img, data_range=dr)

def compute_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    return 20 * np.log10(np.max(target) / (np.sqrt(mse) + 1e-8))

def compute_iou_dice(mask_pred, mask_gt):
    inter = (mask_pred & mask_gt).sum()
    union = (mask_pred | mask_gt).sum()
    iou = inter / (union + 1e-8)
    dice = 2 * inter / (mask_pred.sum() + mask_gt.sum() + 1e-8)
    return iou, dice


# ────────────────────── 主函数 ──────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/conv_spatial_best.pt")
    parser.add_argument("--data", default="data/generated/mixed_dataset.h5")
    parser.add_argument("--mesh_config", default="config/mesh_config.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output", default="results/validation_best_v3")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    os.makedirs(args.output, exist_ok=True)

    # ── 1. 加载模型 ──
    print("加载模型...")
    solver = EITForwardSolver(args.mesh_config)
    centers = solver.element_centers[:, :2]
    n_elems = solver.n_elems
    mesh_nodes = solver.mesh.node[:, :2]
    mesh_elements = solver.mesh.element

    model = ConvSpatialEIT(n_elems=n_elems, gnn_hidden=512, hidden_dim=256)
    model.setup_mesh(centers, solver.mesh.element)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt if 'model_state_dict' not in ckpt else ckpt['model_state_dict'])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数: {n_params:,}")

    # ── 2. 加载数据 ──
    print(f"加载数据 ({args.split})...")
    ds = MemoryEITDataset(args.data, split=args.split, voltage_mask_ratio=0.0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"  共 {len(ds)} 个样本")

    # ── 3. 推理 ──
    print("推理中...")
    all_pred, all_gt, all_masks_pred, all_masks_gt = [], [], [], []
    times = []
    with torch.no_grad():
        for batch in tqdm(loader):
            V = batch['voltages'].to(device).view(-1, 6, 13, 16)
            t0 = time.perf_counter()
            out = model(V)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            all_pred.append(out['sigma'].cpu().numpy())
            all_gt.append(batch['sigmas'].numpy())
            # 掩码
            threshold = 0.02
            masks_gt = (batch['sigmas'].numpy() > threshold).astype(np.float32)
            masks_pred = (out['sigma'].cpu().numpy() > threshold).astype(np.float32)
            all_masks_pred.append(masks_pred)
            all_masks_gt.append(masks_gt)

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    all_masks_pred = np.concatenate(all_masks_pred, axis=0)
    all_masks_gt = np.concatenate(all_masks_gt, axis=0)
    avg_time = np.mean(times) / args.batch_size * 1000  # ms per sample

    # ── 4. 逐样本指标 ──
    print("\n计算指标...")
    re_list, cc_list, ssim_list, psnr_list, iou_list, dice_list = [], [], [], [], [], []
    for i in range(len(all_pred)):
        re_list.append(compute_re(all_pred[i], all_gt[i]))
        cc_list.append(compute_cc(all_pred[i], all_gt[i]))
        ssim_list.append(compute_ssim_on_mesh(all_pred[i], all_gt[i], centers))
        psnr_list.append(compute_psnr(all_pred[i], all_gt[i]))
        i, d = compute_iou_dice(all_masks_pred[i].astype(bool), all_masks_gt[i].astype(bool))
        iou_list.append(i)
        dice_list.append(d)

    metrics = {
        'RE': {'mean': float(np.mean(re_list)), 'std': float(np.std(re_list)),
               'min': float(np.min(re_list)), 'max': float(np.max(re_list))},
        'CC': {'mean': float(np.mean(cc_list)), 'std': float(np.std(cc_list))},
        'SSIM': {'mean': float(np.mean(ssim_list)), 'std': float(np.std(ssim_list))},
        'PSNR': {'mean': float(np.mean(psnr_list)), 'std': float(np.std(psnr_list))},
        'IoU': {'mean': float(np.mean(iou_list)), 'std': float(np.std(iou_list))},
        'Dice': {'mean': float(np.mean(dice_list)), 'std': float(np.std(dice_list))},
        'n_samples': len(all_pred),
        'n_params': n_params,
        'avg_inference_ms': float(avg_time),
    }

    # ── 5. 输出 ──
    print(f"\n{'='*55}")
    print(f"  验证结果 — {args.split.upper()} 集")
    print(f"{'='*55}")
    for k in ['RE', 'CC', 'SSIM', 'PSNR', 'IoU', 'Dice']:
        v = metrics[k]
        print(f"  {k:6s}: mean={v['mean']:.4f}  std={v.get('std', 0):.4f}", end='')
        if 'min' in v: print(f"  [{v['min']:.4f}, {v['max']:.4f}]", end='')
        print()
    print(f"  推理速度: {avg_time:.2f} ms/样本")
    print(f"{'='*55}\n")

    # 保存指标 JSON
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump({'summary': metrics}, f, indent=2)
    # 保存报告
    with open(os.path.join(args.output, 'report.txt'), 'w') as f:
        f.write(f"Conv-Spatial EIT 验证报告\n")
        f.write(f"模型: {args.checkpoint}\n")
        f.write(f"数据集: {args.data} ({args.split})\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"  RE   = {metrics['RE']['mean']:.4f} ± {metrics['RE']['std']:.4f}\n")
        f.write(f"  CC   = {metrics['CC']['mean']:.4f} ± {metrics['CC']['std']:.4f}\n")
        f.write(f"  SSIM = {metrics['SSIM']['mean']:.4f} ± {metrics['SSIM']['std']:.4f}\n")
        f.write(f"  PSNR = {metrics['PSNR']['mean']:.4f} ± {metrics['PSNR']['std']:.4f}\n")
        f.write(f"  IoU  = {metrics['IoU']['mean']:.4f} ± {metrics['IoU']['std']:.4f}\n")
        f.write(f"  Dice = {metrics['Dice']['mean']:.4f} ± {metrics['Dice']['std']:.4f}\n")
        f.write(f"  推理: {avg_time:.2f} ms/样本\n")

    # ── 6. 可视化 ──
    print("生成可视化...")
    n_vis = min(8, len(all_pred))
    errors = np.mean((all_pred - all_gt) ** 2, axis=1)
    best_idx = np.argsort(errors)[:4]
    worst_idx = np.argsort(errors)[-4:]

    # 6a. Batch 对比
    fig, axes = plt.subplots(3, n_vis, figsize=(4 * n_vis, 10))
    fig.patch.set_facecolor('white')
    for i in range(n_vis):
        idx = i
        gt, pred = all_gt[idx], all_pred[idx]
        err = np.abs(pred - gt)
        re = re_list[idx]
        
        axes[0, i].tripcolor(mesh_nodes[:,0], mesh_nodes[:,1],
                              mesh_elements, facecolors=gt,
                              cmap='viridis', vmin=0.008, vmax=0.052, shading='flat')
        axes[0, i].set_title(f'GT [{i}]', fontsize=9); axes[0, i].axis('off')
        axes[0, i].set_aspect('equal')

        axes[1, i].tripcolor(mesh_nodes[:,0], mesh_nodes[:,1],
                              mesh_elements, facecolors=pred,
                              cmap='viridis', vmin=0.008, vmax=0.052, shading='flat')
        axes[1, i].set_title(f'Pred [{i}]', fontsize=9); axes[1, i].axis('off')
        axes[1, i].set_aspect('equal')

        axes[2, i].tripcolor(mesh_nodes[:,0], mesh_nodes[:,1],
                              mesh_elements, facecolors=err,
                              cmap='hot', vmin=0, vmax=0.01, shading='flat')
        axes[2, i].set_title(f'Error RE={re:.4f}', fontsize=9); axes[2, i].axis('off')
        axes[2, i].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'batch_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 6b. Best 4
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.patch.set_facecolor('white')
    for col, idx in enumerate(best_idx):
        gt, pred = all_gt[idx], all_pred[idx]
        err = np.abs(pred - gt)
        re = re_list[idx]
        for row, (data, cmap, vmin, vmax, label) in enumerate([
            (gt, 'viridis', 0.008, 0.052, 'GT'),
            (pred, 'viridis', 0.008, 0.052, 'Pred'),
            (err, 'hot', 0, 0.01, f'Error RE={re:.4f}'),
        ]):
            axes[row, col].tripcolor(mesh_nodes[:,0], mesh_nodes[:,1],
                                      mesh_elements, facecolors=data,
                                      cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
            axes[row, col].set_aspect('equal'); axes[row, col].axis('off')
            if row == 0: axes[row, col].set_title(f'Best #{col+1}', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'best_4.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 6c. Worst 4
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.patch.set_facecolor('white')
    for col, idx in enumerate(worst_idx):
        gt, pred = all_gt[idx], all_pred[idx]
        err = np.abs(pred - gt)
        re = re_list[idx]
        for row, (data, cmap, vmin, vmax, label) in enumerate([
            (gt, 'viridis', 0.008, 0.052, 'GT'),
            (pred, 'viridis', 0.008, 0.052, 'Pred'),
            (err, 'hot', 0, 0.01, f'Error RE={re:.4f}'),
        ]):
            axes[row, col].tripcolor(mesh_nodes[:,0], mesh_nodes[:,1],
                                      mesh_elements, facecolors=data,
                                      cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
            axes[row, col].set_aspect('equal'); axes[row, col].axis('off')
            if row == 0: axes[row, col].set_title(f'Worst #{col+1}', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'worst_4.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 6d. 相关性散点图
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('white')
    all_pred_flat = all_pred.flatten()
    all_gt_flat = all_gt.flatten()
    ax.scatter(all_gt_flat, all_pred_flat, s=1, alpha=0.3, c='#3b82f6')
    lims = [min(all_gt_flat.min(), all_pred_flat.min()),
            max(all_gt_flat.max(), all_pred_flat.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Ground Truth σ'); ax.set_ylabel('Predicted σ')
    ax.set_title(f'Correlation (CC={metrics["CC"]["mean"]:.4f})')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 验证完成！结果已保存到: {args.output}/")
    print(f"   指标: metrics.json, report.txt")
    print(f"   可视化: batch_comparison.png, best_4.png, worst_4.png, correlation.png")


if __name__ == "__main__":
    main()
