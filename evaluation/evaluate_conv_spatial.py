"""
Conv-Spatial EIT 评估脚本
=========================
适配 ConvSpatialEIT 模型的评估
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.conv_spatial_eit import ConvSpatialEIT
from data.datasets.eit_dataset import MemoryEITDataset
from data.eit_forward import EITForwardSolver


def compute_metrics(pred, target, mask=None):
    """计算评估指标"""
    # Relative Error
    re = np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-8)

    # Correlation Coefficient
    cc = np.corrcoef(pred.flatten(), target.flatten())[0, 1]

    # RMSE
    rmse = np.sqrt(np.mean((pred - target) ** 2))

    # SSIM (简化版，在mesh上计算)
    if mask is not None:
        pred_mask = pred[mask > 0.5] if mask is not None else pred
        target_mask = target[mask > 0.5] if mask is not None else target
    else:
        pred_mask = pred
        target_mask = target

    # 结构相似性 (简化)
    mu_pred = np.mean(pred_mask)
    mu_target = np.mean(target_mask)
    sigma_pred = np.std(pred_mask)
    sigma_target = np.std(target_mask)
    sigma_cross = np.mean((pred_mask - mu_pred) * (target_mask - mu_target))

    c1 = (0.01 * (np.max(target) - np.min(target))) ** 2
    c2 = (0.03 * (np.max(target) - np.min(target))) ** 2

    ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / \
           ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred**2 + sigma_target**2 + c2))

    return {
        'RE': re,
        'CC': cc,
        'RMSE': rmse,
        'SSIM': ssim
    }


def visualize_reconstruction(centers, elements, gt, pred, save_path, title="Reconstruction"):
    """可视化重建结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground Truth
    ax = axes[0]
    patches = []
    for elem in elements:
        poly = Polygon(centers[elem], closed=True)
        patches.append(poly)
    p = PatchCollection(patches, alpha=1.0)
    p.set_array(gt)
    ax.add_collection(p)
    ax.set_xlim(centers[:, 0].min() - 0.5, centers[:, 0].max() + 0.5)
    ax.set_ylim(centers[:, 1].min() - 0.5, centers[:, 1].max() + 0.5)
    ax.set_aspect('equal')
    ax.set_title('Ground Truth')
    plt.colorbar(p, ax=ax)

    # Prediction
    ax = axes[1]
    patches = []
    for elem in elements:
        poly = Polygon(centers[elem], closed=True)
        patches.append(poly)
    p = PatchCollection(patches, alpha=1.0)
    p.set_array(pred)
    ax.add_collection(p)
    ax.set_xlim(centers[:, 0].min() - 0.5, centers[:, 0].max() + 0.5)
    ax.set_ylim(centers[:, 1].min() - 0.5, centers[:, 1].max() + 0.5)
    ax.set_aspect('equal')
    ax.set_title('Prediction')
    plt.colorbar(p, ax=ax)

    # Error
    ax = axes[2]
    patches = []
    for elem in elements:
        poly = Polygon(centers[elem], closed=True)
        patches.append(poly)
    p = PatchCollection(patches, alpha=1.0)
    error = np.abs(pred - gt)
    p.set_array(error)
    ax.add_collection(p)
    ax.set_xlim(centers[:, 0].min() - 0.5, centers[:, 0].max() + 0.5)
    ax.set_ylim(centers[:, 1].min() - 0.5, centers[:, 1].max() + 0.5)
    ax.set_aspect('equal')
    ax.set_title('Absolute Error')
    plt.colorbar(p, ax=ax)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--config', type=str, default='config/mesh_config.yaml')
    parser.add_argument('--split', type=str, default='test', help='数据集划分')
    parser.add_argument('--output', type=str, default='results/evaluation', help='输出目录')
    parser.add_argument('--n_samples', type=int, default=100, help='评估样本数')
    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载 checkpoint
    print(f"\n加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 提取模型配置
    n_elems = checkpoint.get('n_elems', 1500)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    gnn_hidden = checkpoint.get('gnn_hidden', 256)
    gnn_layers = checkpoint.get('gnn_layers', 4)
    use_gat = not checkpoint.get('use_gat', False)
    n_heads = checkpoint.get('n_heads', 4)

    print(f"模型配置: hidden_dim={hidden_dim}, gnn_hidden={gnn_hidden}, "
          f"gnn_layers={gnn_layers}, use_gat={use_gat}, n_heads={n_heads}")

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
    ).to(device)

    # 加载权重 (优先使用 EMA 模型)
    if 'ema_model' in checkpoint:
        print("使用 EMA 模型权重")
        state_dict = checkpoint['ema_model']
        # 移除 DataParallel 的 "module." 前缀
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("检测到 DataParallel 包装，移除 'module.' 前缀")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k != 'n_averaged'}
        model.load_state_dict(state_dict, strict=False)
    elif 'model' in checkpoint:
        print("使用标准模型权重")
        state_dict = checkpoint['model']
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        print("尝试直接加载 state_dict")
        state_dict = checkpoint
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # 设置mesh信息
    print("设置网格结构...")
    jacobian_path = "data/generated/jacobian.npy"
    jacobian = None
    if os.path.exists(jacobian_path):
        jacobian = np.load(jacobian_path)
        print(f"加载Jacobian: {jacobian.shape}")
    model.setup_mesh(centers, elements, jacobian, sigma_ref=0.01)

    # 加载网格数据
    print("\n加载网格数据...")
    solver = EITForwardSolver(args.config)
    centers = solver.element_centers
    if centers.shape[1] > 2:
        centers = centers[:, :2]
    elements = solver.mesh.element

    # 加载数据集
    print(f"加载 {args.split} 数据集...")
    h5_path = "data/generated/mixed_dataset.h5"
    dataset = MemoryEITDataset(h5_path, split=args.split)
    print(f"数据集大小: {len(dataset)} 样本")

    # 限制样本数
    n_samples = min(args.n_samples, len(dataset))
    print(f"评估 {n_samples} 个样本...")

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

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 保存指标
    import json
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\n指标已保存到: {args.output}/metrics.json")

    # 可视化
    print("\n生成可视化...")

    # 计算每个样本的误差
    errors = [m['RE'] for m in all_metrics]

    # Best samples
    best_idx = np.argsort(errors)[:4]
    print(f"最佳样本索引: {best_idx}")

    # Worst samples
    worst_idx = np.argsort(errors)[-4:][::-1]
    print(f"最差样本索引: {worst_idx}")

    # 可视化最佳样本
    for i, idx in enumerate(best_idx):
        visualize_reconstruction(
            centers, elements, all_gts[idx], all_preds[idx],
            os.path.join(args.output, f'best_sample_{i}.png'),
            title=f'Best Sample {i} (RE={errors[idx]:.4f})'
        )

    # 可视化最差样本
    for i, idx in enumerate(worst_idx):
        visualize_reconstruction(
            centers, elements, all_gts[idx], all_preds[idx],
            os.path.join(args.output, f'worst_sample_{i}.png'),
            title=f'Worst Sample {i} (RE={errors[idx]:.4f})'
        )

    # 可视化随机样本
    np.random.seed(42)
    random_idx = np.random.choice(n_samples, 4, replace=False)
    for i, idx in enumerate(random_idx):
        visualize_reconstruction(
            centers, elements, all_gts[idx], all_preds[idx],
            os.path.join(args.output, f'random_sample_{i}.png'),
            title=f'Random Sample {i} (RE={errors[idx]:.4f})'
        )

    # 绘制误差分布直方图
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


if __name__ == '__main__':
    main()
