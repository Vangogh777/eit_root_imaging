"""
Conv-Spatial EIT 验证脚本
=========================
加载训练好的模型，对验证集进行推理，输出指标和对比图。

用法:
    python evaluate_conv_spatial.py
    python evaluate_conv_spatial.py --checkpoint checkpoints/conv_spatial_best.pt
    python evaluate_conv_spatial.py --n_samples 4
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.conv_spatial_eit import ConvSpatialEIT
from data.eit_forward import EITForwardSolver
from data.datasets.eit_dataset import EITDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/conv_spatial_best.pt")
    parser.add_argument("--data", default="data/generated/circle_dataset.h5")
    parser.add_argument("--mesh_config", default="config/mesh_config.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--n_samples", type=int, default=8, help="画图展示的样本数")
    parser.add_argument("--output", default="results/validation_conv_spatial.png")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ========== 1. 加载模型 ==========
    print("加载模型...")
    solver = EITForwardSolver(args.mesh_config)
    centers = solver.element_centers[:, :2]
    n_elems = solver.n_elems
    # 获取网格节点和单元连接（用于 tripcolor 平滑渲染）
    mesh_nodes = solver.mesh.node[:, :2]  # (N_nodes, 2)
    mesh_elements = solver.mesh.element    # (N_elems, 3)

    model = ConvSpatialEIT(n_elems=n_elems)
    model.setup_mesh(centers, solver.mesh.element)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'ema_model' in ckpt:
        from torch.optim.swa_utils import AveragedModel
        ema_model = AveragedModel(model)
        ema_model.load_state_dict(ckpt['ema_model'])
        model.load_state_dict(ema_model.module.state_dict())
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 2. 加载数据 ==========
    print(f"加载数据 ({args.split})...")
    ds = EITDataset(args.data, split=args.split, voltage_mask_ratio=0.0)
    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    safe_batch = max(4, min(64, int(free_mem / (11466 * 256 * 4 * 2))))
    batch_size = min(64, safe_batch)
    print(f"  显存: {free_mem/1e9:.1f}GB 空闲, 使用 batch_size={batch_size}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    print(f"  共 {len(ds)} 个样本")

    # ========== 3. 推理 ==========
    print("推理中...")
    all_pred, all_gt = [], []
    with torch.no_grad():
        for batch in loader:
            V = batch['voltages'].to(device).view(-1, 6, 13, 16)
            out = model(V)
            all_pred.append(out['sigma'].cpu())
            all_gt.append(batch['sigmas'])

    all_pred = torch.cat(all_pred)
    all_gt = torch.cat(all_gt)

    # ========== 4. 指标 ==========
    re_per_sample = torch.norm(all_pred - all_gt, dim=-1) / (torch.norm(all_gt, dim=-1) + 1e-8)
    re_mean = re_per_sample.mean().item()
    re_std = re_per_sample.std().item()
    re_min = re_per_sample.min().item()
    re_max = re_per_sample.max().item()

    # CC per sample
    cc_list = []
    for i in range(len(all_pred)):
        p = all_pred[i] - all_pred[i].mean()
        g = all_gt[i] - all_gt[i].mean()
        cc = (p * g).sum() / (p.norm() * g.norm() + 1e-8)
        cc_list.append(cc.item())
    cc_mean = np.mean(cc_list)
    cc_std = np.std(cc_list)

    print(f"\n{'='*50}")
    print(f"验证结果 ({args.split})")
    print(f"{'='*50}")
    print(f"  样本数:      {len(all_pred)}")
    print(f"  RE (mean):   {re_mean:.4f}")
    print(f"  RE (std):    {re_std:.4f}")
    print(f"  RE (min):    {re_min:.4f}")
    print(f"  RE (max):    {re_max:.4f}")
    print(f"  CC (mean):   {cc_mean:.4f}")
    print(f"  CC (std):    {cc_std:.4f}")
    print(f"{'='*50}\n")

    # ========== 5. 画图 ==========
    n_show = min(args.n_samples, len(all_pred))
    # 上方留出指标区域，下方为对比图
    fig = plt.figure(figsize=(12, 3.5 + n_show * 2.8))
    fig.patch.set_facecolor('white')

    # ── 5a. 指标面板 ──
    ax_stats = fig.add_axes([0.05, 0.94, 0.9, 0.05])
    ax_stats.axis('off')
    stats_text = (
        f"Validation: {len(all_pred)} samples  |  "
        f"RE: mean={re_mean:.4f}  std={re_std:.4f}  min={re_min:.4f}  max={re_max:.4f}  |  "
        f"CC: mean={cc_mean:.4f}  std={cc_std:.4f}"
    )
    ax_stats.text(0.5, 0.5, stats_text, fontsize=11, color='#333333',
                  ha='center', va='center', fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4f8',
                            edgecolor='#c0d0e0', linewidth=1))

    # ── 5b. 对比图 ──
    indices = torch.argsort(re_per_sample)
    half = n_show // 2
    show_idx = list(indices[:half]) + list(indices[-half:])

    for row, idx in enumerate(show_idx):
        gt = all_gt[idx].numpy()
        pred = all_pred[idx].numpy()
        err = np.abs(pred - gt)
        re_i = re_per_sample[idx].item()

        y0 = 0.88 - row * 0.30
        for j, (data, cmap, vmin, vmax, label) in enumerate([
            (gt, 'viridis', 0.008, 0.052, 'Ground Truth'),
            (pred, 'viridis', 0.008, 0.052, 'Prediction'),
            (err, 'hot', 0, 0.01, f'Error  RE={re_i:.4f}'),
        ]):
            ax = fig.add_axes([0.02 + j * 0.33, y0 - 0.25, 0.30, 0.25])
            # 使用 tripcolor 填充三角形网格，消除散点噪点
            ax.tripcolor(mesh_nodes[:,0], mesh_nodes[:,1],
                        mesh_elements, facecolors=data,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='flat')
            ax.set_aspect('equal')
            ax.axis('off')
            if row == 0:
                ax.set_title(label, fontsize=10, color='#222222', fontweight='bold', pad=2)

    plt.savefig(args.output, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"对比图已保存: {args.output}")


if __name__ == "__main__":
    main()
