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

    model = ConvSpatialEIT(n_elems=n_elems)
    model.setup_mesh(centers, solver.mesh.element)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 2. 加载数据 ==========
    print(f"加载数据 ({args.split})...")
    ds = EITDataset(args.data, split=args.split, voltage_mask_ratio=0.0)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
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
    fig, axes = plt.subplots(n_show, 3, figsize=(10, n_show * 3))
    fig.patch.set_facecolor('#0a0a0f')

    # 按 RE 排序显示
    indices = torch.argsort(re_per_sample)
    # 显示最好、最差和中间的
    half = n_show // 2
    show_idx = list(indices[:half]) + list(indices[-half:])

    for row, idx in enumerate(show_idx):
        gt = all_gt[idx].numpy()
        pred = all_pred[idx].numpy()
        err = np.abs(pred - gt)
        re_i = re_per_sample[idx].item()

        titles = ['Ground Truth', 'Prediction', f'Error  RE={re_i:.4f}']
        datas = [gt, pred, err]
        cmaps = ['viridis', 'viridis', 'hot']
        vmins = [0.008, 0.008, 0]
        vmaxs = [0.052, 0.052, 0.01]

        for j in range(3):
            ax = axes[row, j] if n_show > 1 else axes[j]
            sc = ax.scatter(centers[:,0], centers[:,1], c=datas[j],
                          s=3, cmap=cmaps[j], vmin=vmins[j], vmax=vmaxs[j])
            ax.set_aspect('equal')
            ax.axis('off')
            if row == 0:
                ax.set_title(titles[j], color='#6ee7b7', fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
    print(f"对比图已保存: {args.output}")


if __name__ == "__main__":
    main()
