#!/usr/bin/env python
"""
Conv-Spatial EIT 完整评估脚本
=============================
评估训练好的模型并生成可视化结果
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.conv_spatial_eit import ConvSpatialEIT
from data.datasets.eit_dataset import MemoryEITDataset
from data.eit_forward import EITForwardSolver


def compute_metrics(pred, target):
    """计算评估指标"""
    # Relative Error
    re = np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-8)

    # Correlation Coefficient
    cc = np.corrcoef(pred.flatten(), target.flatten())[0, 1]

    # RMSE
    rmse = np.sqrt(np.mean((pred - target) ** 2))

    # SSIM (简化版)
    mu_pred = np.mean(pred)
    mu_target = np.mean(target)
    sigma_pred = np.std(pred)
    sigma_target = np.std(target)
    sigma_cross = np.mean((pred - mu_pred) * (target - mu_target))

    c1 = (0.01 * (np.max(target) - np.min(target))) ** 2
    c2 = (0.03 * (np.max(target) - np.min(target))) ** 2

    ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / \
           ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred**2 + sigma_target**2 + c2))

    return {'RE': re, 'CC': cc, 'RMSE': rmse, 'SSIM': ssim}


def visualize_reconstruction(centers, elements, gt, pred, save_path, title=""):
    """可视化重建结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmax = max(gt.max(), pred.max())
    vmin = min(gt.min(), pred.min())

    for idx, (data, name) in enumerate([(gt, 'Ground Truth'), (pred, 'Prediction'), (np.abs(pred - gt), 'Error')]):
        ax = axes[idx]
        patches = []
        for elem in elements:
            poly = Polygon(centers[elem], closed=True)
            patches.append(poly)
        p = PatchCollection(patches, alpha=1.0)
        p.set_array(data)
        if idx < 2:
            p.set_clim(vmin, vmax)
        ax.add_collection(p)
        ax.set_xlim(centers[:, 0].min() - 0.5, centers[:, 0].max() + 0.5)
        ax.set_ylim(centers[:, 1].min() - 0.5, centers[:, 1].max() + 0.5)
        ax.set_aspect('equal')
        ax.set_title(name)
        plt.colorbar(p, ax=ax)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/20260622_231207_v2_unsupervised_hd256/final.pt')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--output', type=str, default='results/evaluation')
    args = parser.parse_args()

    print("="*60)
    print("Conv-Spatial EIT 模型评估")
    print("="*60)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # 加载checkpoint
    print(f"\n加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 获取配置
    n_elems = checkpoint.get('n_elems', 1500)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    gnn_hidden = checkpoint.get('gnn_hidden', 256)
    gnn_layers = checkpoint.get('gnn_layers', 4)
    use_gat = not checkpoint.get('use_gat', False)
    n_heads = checkpoint.get('n_heads', 4)

    print(f"模型配置:")
    print(f"  - hidden_dim: {hidden_dim}")
    print(f"  - gnn_hidden: {gnn_hidden}")
    print(f"  - gnn_layers: {gnn_layers}")
    print(f"  - use_gat: {use_gat}")
    print(f"  - n_heads: {n_heads}")
    print(f"  - n_elems: {n_elems}")

    # 创建模型
    model = ConvSpatialEIT(
        n_frequencies=6,
        n_meas=208,
        n_elems=n_elems,
        hidden_dim=hidden_dim,
        gnn_hidden=gnn_hidden,
        gnn_layers=gnn_layers,
        use_gat=use_gat,
        n_heads=n_heads
    )

    # 加载权重（过滤掉J和J_T）
    if 'ema_model' in checkpoint:
        print("\n使用EMA模型权重")
        state_dict = checkpoint['ema_model']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()
                          if k not in ['n_averaged']}
        # 过滤J和J_T（会在setup_mesh时重新设置）
        state_dict = {k: v for k, v in state_dict.items() if k not in ['J', 'J_T']}
        model.load_state_dict(state_dict, strict=False)
    elif 'model' in checkpoint:
        print("\n使用标准模型权重")
        state_dict = checkpoint['model']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if k not in ['J', 'J_T']}
        model.load_state_dict(state_dict, strict=False)

    # 加载网格
    print("\n加载网格数据...")
    solver = EITForwardSolver("config/mesh_config.yaml")
    centers = solver.element_centers[:, :2]
    elements = solver.mesh.element

    # 加载Jacobian
    jacobian_path = "data/generated/jacobian.npy"
    jacobian = np.load(jacobian_path) if os.path.exists(jacobian_path) else None
    if jacobian is not None:
        print(f"Jacobian形状: {jacobian.shape}")

    # 设置mesh
    model.setup_mesh(centers, elements, jacobian, sigma_ref=0.01)
    print(f"网格设置完成: {n_elems} 单元")

    # 移到设备
    model = model.to(device)
    model.eval()

    # 加载数据集
    print(f"\n加载测试数据集...")
    h5_path = "data/generated/mixed_dataset.h5"
    dataset = MemoryEITDataset(h5_path, split='test')
    print(f"数据集大小: {len(dataset)} 样本")

    n_samples = min(args.n_samples, len(dataset))
    print(f"评估 {n_samples} 个样本...")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 评估
    all_metrics = []
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="评估中"):
            sample = dataset[i]
            voltages = sample['voltages'].unsqueeze(0).to(device)
            sigma_gt = sample['sigma'].cpu().numpy()

            # 前向传播
            sigma_pred = model(voltages).cpu().numpy()[0]

            # 计算指标
            metrics = compute_metrics(sigma_pred, sigma_gt)
            all_metrics.append(metrics)

            all_preds.append(sigma_pred)
            all_gts.append(sigma_gt)

    # 汇总结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)

    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        std = np.std(values)
        print(f"{key:10s}: {avg_metrics[key]:.6f} ± {std:.6f}")

    # 保存指标
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\n指标已保存到: {args.output}/metrics.json")

    # 可视化
    print("\n生成可视化...")

    errors = [m['RE'] for m in all_metrics]
    best_idx = np.argsort(errors)[:4]
    worst_idx = np.argsort(errors)[-4:][::-1]

    # 最佳样本
    for i, idx in enumerate(best_idx):
        visualize_reconstruction(
            centers, elements, all_gts[idx], all_preds[idx],
            os.path.join(args.output, f'best_sample_{i}.png'),
            title=f'Best Sample {i} (RE={errors[idx]:.4f})'
        )

    # 最差样本
    for i, idx in enumerate(worst_idx):
        visualize_reconstruction(
            centers, elements, all_gts[idx], all_preds[idx],
            os.path.join(args.output, f'worst_sample_{i}.png'),
            title=f'Worst Sample {i} (RE={errors[idx]:.4f})'
        )

    # 随机样本
    np.random.seed(42)
    random_idx = np.random.choice(n_samples, 4, replace=False)
    for i, idx in enumerate(random_idx):
        visualize_reconstruction(
            centers, elements, all_gts[idx], all_preds[idx],
            os.path.join(args.output, f'random_sample_{i}.png'),
            title=f'Random Sample {i} (RE={errors[idx]:.4f})'
        )

    # 误差分布
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Relative Error (RE)')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.axvline(avg_metrics['RE'], color='red', linestyle='--',
                label=f'Mean RE: {avg_metrics["RE"]:.4f}')
    plt.legend()
    plt.savefig(os.path.join(args.output, 'error_distribution.png'), dpi=150)
    plt.close()

    print(f"\n可视化结果已保存到: {args.output}/")
    print("="*60)
    print("评估完成！")


if __name__ == '__main__':
    main()
