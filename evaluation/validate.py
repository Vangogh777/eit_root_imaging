"""
EIT 重建质量评估脚本
====================
评估训练好的模型在测试集上的重建质量

用法:
    python evaluation/validate.py --checkpoint checkpoints/server_model.pt
    python evaluation/validate.py --checkpoint checkpoints/server_model.pt --n_samples 100
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.eit_forward import EITForwardSolver
from models.simple_model import SimpleSFSBLC
from models.universal_eit import PhysicsInformedEIT, UniversalPhantomGenerator


def compute_re(pred: np.ndarray, target: np.ndarray) -> float:
    """
    相对误差 RE = ||pred - target|| / ||target||
    """
    return np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-8)


def compute_cc(pred: np.ndarray, target: np.ndarray) -> float:
    """
    相关系数 CC (Correlation Coefficient)
    """
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    cov = np.sum(pred_centered * target_centered)
    std_pred = np.sqrt(np.sum(pred_centered ** 2) + 1e-8)
    std_target = np.sqrt(np.sum(target_centered ** 2) + 1e-8)
    return cov / (std_pred * std_target + 1e-8)


def compute_ssim_on_mesh(pred: np.ndarray, target: np.ndarray,
                         centers: np.ndarray, resolution: int = 128) -> float:
    """
    将网格数据插值到规则网格上计算 SSIM
    """
    from scipy.interpolate import griddata

    # 只取前两维坐标（x, y），处理可能的3D/4D坐标
    centers_2d = centers[:, :2] if centers.shape[1] > 2 else centers

    # 创建规则网格
    radius = np.max(np.abs(centers_2d)) * 1.1
    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(x, y)

    # 插值
    pred_img = griddata(centers_2d, pred, (X, Y), method='linear', fill_value=0)
    target_img = griddata(centers_2d, target, (X, Y), method='linear', fill_value=0)

    # 创建圆形mask
    mask = (X**2 + Y**2) <= radius**2
    pred_img[~mask] = 0
    target_img[~mask] = 0

    # 计算 SSIM
    data_range = max(target_img.max() - target_img.min(), 1e-8)
    return ssim(pred_img, target_img, data_range=data_range)


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """
    峰值信噪比 PSNR (Peak Signal-to-Noise Ratio)
    """
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0  # 几乎相同
    max_val = max(target.max() - target.min(), 1e-8)
    return 10 * np.log10(max_val ** 2 / mse)


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    均方根误差 RMSE
    """
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """
    平均绝对误差 MAE
    """
    return np.mean(np.abs(pred - target))


def compute_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.02) -> float:
    """
    交并比 IoU (Intersection over Union)
    用于检测异常区域的定位精度
    """
    pred_mask = pred > threshold
    target_mask = target > threshold

    intersection = np.sum(pred_mask & target_mask)
    union = np.sum(pred_mask | target_mask)

    if union == 0:
        return 1.0  # 两者都没有异常区域
    return intersection / union


def compute_dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.02) -> float:
    """
    Dice 系数 (F1 Score)
    """
    pred_mask = pred > threshold
    target_mask = target > threshold

    intersection = np.sum(pred_mask & target_mask)
    total = np.sum(pred_mask) + np.sum(target_mask)

    if total == 0:
        return 1.0
    return 2 * intersection / total


def plot_comparison(centers: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                    save_path: str, sample_id: int = 0, metrics: dict = None):
    """
    绘制单个样本的对比图
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 确定颜色范围
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())

    # Ground Truth
    ax = axes[0]
    sc = ax.scatter(centers[:, 0], centers[:, 1], c=gt, s=1, cmap='viridis',
                    vmin=vmin, vmax=vmax)
    ax.set_title(f'Ground Truth [{sample_id}]')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # Prediction
    ax = axes[1]
    sc = ax.scatter(centers[:, 0], centers[:, 1], c=pred, s=1, cmap='viridis',
                    vmin=vmin, vmax=vmax)
    ax.set_title('Prediction')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # Error
    ax = axes[2]
    error = np.abs(pred - gt)
    sc = ax.scatter(centers[:, 0], centers[:, 1], c=error, s=1, cmap='hot',
                    vmin=0, vmax=np.percentile(error, 95))
    ax.set_title('Absolute Error')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # Difference
    ax = axes[3]
    diff = pred - gt
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    sc = ax.scatter(centers[:, 0], centers[:, 1], c=diff, s=1, cmap='RdBu_r',
                    vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_title('Difference (Pred - GT)')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # 添加指标文本
    if metrics:
        metrics_text = f"RE: {metrics['RE']:.4f} | CC: {metrics['CC']:.4f} | SSIM: {metrics['SSIM']:.4f}"
        fig.suptitle(metrics_text, fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_batch_comparison(centers: np.ndarray, gt_batch: np.ndarray,
                          pred_batch: np.ndarray, save_path: str, n_samples: int = 8):
    """
    绘制批量样本对比图
    """
    n_samples = min(n_samples, len(gt_batch))

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    vmin = min(gt_batch[:n_samples].min(), pred_batch[:n_samples].min())
    vmax = max(gt_batch[:n_samples].max(), pred_batch[:n_samples].max())

    for i in range(n_samples):
        gt = gt_batch[i]
        pred = pred_batch[i]
        error = np.abs(pred - gt)
        diff = pred - gt

        # GT
        ax = axes[i, 0]
        ax.scatter(centers[:, 0], centers[:, 1], c=gt, s=1, cmap='viridis',
                   vmin=vmin, vmax=vmax)
        ax.set_title(f'GT [{i}]')
        ax.set_aspect('equal')
        ax.axis('off')

        # Prediction
        ax = axes[i, 1]
        ax.scatter(centers[:, 0], centers[:, 1], c=pred, s=1, cmap='viridis',
                   vmin=vmin, vmax=vmax)
        ax.set_title(f'Pred [{i}]')
        ax.set_aspect('equal')
        ax.axis('off')

        # Error
        ax = axes[i, 2]
        ax.scatter(centers[:, 0], centers[:, 1], c=error, s=1, cmap='hot',
                   vmin=0, vmax=np.percentile(error, 95))
        ax.set_title(f'Error [{i}]')
        ax.set_aspect('equal')
        ax.axis('off')

        # Difference
        ax = axes[i, 3]
        vmax_diff = max(abs(diff.min()), abs(diff.max()))
        ax.scatter(centers[:, 0], centers[:, 1], c=diff, s=1, cmap='RdBu_r',
                   vmin=-vmax_diff, vmax=vmax_diff)
        ax.set_title(f'Diff [{i}]')
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_histogram(metrics_list: list, save_path: str):
    """
    绘制指标分布直方图
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    metric_names = ['RE', 'CC', 'SSIM', 'PSNR', 'RMSE', 'MAE', 'IoU', 'Dice']

    for idx, name in enumerate(metric_names):
        ax = axes[idx // 4, idx % 4]
        values = [m[name] for m in metrics_list]
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
        ax.set_xlabel(name)
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Distribution')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_correlation(gt_batch: np.ndarray, pred_batch: np.ndarray,
                              save_path: str):
    """
    绘制 GT vs Pred 散点图
    """
    gt_flat = gt_batch.flatten()
    pred_flat = pred_batch.flatten()

    # 随机采样避免过多点
    if len(gt_flat) > 10000:
        idx = np.random.choice(len(gt_flat), 10000, replace=False)
        gt_flat = gt_flat[idx]
        pred_flat = pred_flat[idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gt_flat, pred_flat, alpha=0.3, s=1)

    # 理想线
    min_val = min(gt_flat.min(), pred_flat.min())
    max_val = max(gt_flat.max(), pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Prediction')
    ax.set_title('GT vs Prediction Correlation')
    ax.legend()
    ax.set_aspect('equal')

    # 计算整体相关系数
    cc = np.corrcoef(gt_flat, pred_flat)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {cc:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="EIT 重建质量评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--n_samples", type=int, default=100, help="评估样本数")
    parser.add_argument("--output", type=str, default="results/validation", help="输出目录")
    parser.add_argument("--noise_db", type=float, default=-30, help="噪声水平 (dB)")
    args = parser.parse_args()

    print("=" * 60)
    print("📊 EIT 重建质量评估")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "samples"), exist_ok=True)

    # ============ 1. 加载模型 ============
    print(f"\n[1/4] 加载模型: {args.checkpoint}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    print(f"  配置: n_elems={config['n_elems']}, n_freq={config['n_freq']}, "
          f"n_meas={config['n_meas']}, hidden_dim={config.get('hidden_dim', 512)}")

    if checkpoint.get('model_type', 'simple') == 'simple':
        model = SimpleSFSBLC(
            input_dim=config['n_meas'],
            hidden_dim=config.get('hidden_dim', 512),
            n_frequencies=config['n_freq'],
            n_elems=config['n_elems'],
        ).to(device)
    else:
        model = PhysicsInformedEIT(
            input_dim=config['n_meas'],
            hidden_dim=config.get('hidden_dim', 512),
            n_frequencies=config['n_freq'],
            n_elems=config['n_elems'],
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  模型已加载 (Epoch {checkpoint.get('best_epoch', '?')}, "
          f"Val RE: {checkpoint.get('best_val_re', '?'):.4f})")

    # ============ 2. 初始化求解器 ============
    print("\n[2/4] 初始化 EIT 求解器...")
    solver = EITForwardSolver("config/mesh_config.yaml")
    n_elems = solver.n_elems
    n_freq = len(solver.frequencies)
    n_meas = solver.n_measurements

    # 检查网格单元数是否匹配
    if n_elems != config['n_elems']:
        print(f"  ⚠️  警告: 网格单元数不匹配 (模型: {config['n_elems']}, 当前: {n_elems})")
        print(f"  使用模型配置的网格...")

    print(f"  网格: {n_elems} 单元, {n_freq} 频率, {n_meas} 测量通道")

    # 获取单元中心坐标
    centers = np.mean(solver.mesh.node[solver.mesh.element], axis=1)

    # ============ 3. 生成测试数据 ============
    print(f"\n[3/4] 生成 {args.n_samples} 个测试样本...")

    phantom_gen = UniversalPhantomGenerator(
        solver.mesh.node,
        solver.mesh.element,
        domain_radius=solver.cfg['mesh']['radius'],
        sigma_background=0.01,
        sigma_inclusion=0.05
    )

    gt_list = []
    pred_list = []
    metrics_list = []

    with torch.no_grad():
        for i in tqdm(range(args.n_samples), desc="  评估中"):
            # 生成样本
            sigma = phantom_gen.generate_random(seed=10000 + i)
            V = solver.solve_multi_frequency(sigma)

            if np.isnan(V).any():
                V = np.random.randn(n_freq, n_meas).astype(np.float32) * 1e-6

            V_noisy = solver.add_noise(V, noise_db=args.noise_db)

            # 模型推理
            V_tensor = torch.from_numpy(V_noisy).float().unsqueeze(0).to(device)
            out = model(V_tensor)
            pred = out['sigma'][0].cpu().numpy()

            # 确保预测和GT维度一致
            if len(pred) != len(sigma):
                # 如果维度不匹配，跳过或插值
                print(f"  ⚠️ 维度不匹配: pred={len(pred)}, gt={len(sigma)}")
                continue

            gt_list.append(sigma)
            pred_list.append(pred)

            # 计算指标
            metrics = {
                'RE': compute_re(pred, sigma),
                'CC': compute_cc(pred, sigma),
                'SSIM': compute_ssim_on_mesh(pred, sigma, centers),
                'PSNR': compute_psnr(pred, sigma),
                'RMSE': compute_rmse(pred, sigma),
                'MAE': compute_mae(pred, sigma),
                'IoU': compute_iou(pred, sigma),
                'Dice': compute_dice(pred, sigma),
            }
            metrics_list.append(metrics)

    gt_batch = np.array(gt_list)
    pred_batch = np.array(pred_list)

    # ============ 4. 计算汇总统计 ============
    print("\n[4/4] 生成评估报告...")

    # 汇总统计
    summary = {}
    for name in metrics_list[0].keys():
        values = [m[name] for m in metrics_list]
        summary[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
        }

    # 打印结果
    print("\n" + "=" * 60)
    print("📈 评估结果汇总")
    print("=" * 60)
    print(f"  样本数: {len(metrics_list)}")
    print("-" * 60)
    print(f"  {'指标':<10} {'均值':>10} {'标准差':>10} {'中位数':>10} {'最小':>10} {'最大':>10}")
    print("-" * 60)
    for name, stats in summary.items():
        print(f"  {name:<10} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['median']:>10.4f} {stats['min']:>10.4f} {stats['max']:>10.4f}")
    print("=" * 60)

    # ============ 5. 保存结果 ============

    # 保存汇总报告
    report_path = os.path.join(args.output, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("EIT 重建质量评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型: {args.checkpoint}\n")
        f.write(f"样本数: {len(metrics_list)}\n")
        f.write(f"噪声水平: {args.noise_db} dB\n")
        f.write(f"网格单元数: {n_elems}\n\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'指标':<10} {'均值':>10} {'标准差':>10} {'中位数':>10} {'最小':>10} {'最大':>10}\n")
        f.write("-" * 60 + "\n")
        for name, stats in summary.items():
            f.write(f"{name:<10} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                   f"{stats['median']:>10.4f} {stats['min']:>10.4f} {stats['max']:>10.4f}\n")
        f.write("=" * 60 + "\n")
    print(f"  报告已保存: {report_path}")

    # 保存指标JSON
    import json
    json_path = os.path.join(args.output, "metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'per_sample': metrics_list,
        }, f, indent=2)
    print(f"  指标已保存: {json_path}")

    # 保存指标分布图
    plot_metrics_histogram(metrics_list, os.path.join(args.output, "metrics_histogram.png"))
    print(f"  直方图已保存: {args.output}/metrics_histogram.png")

    # 保存相关性散点图
    plot_scatter_correlation(gt_batch, pred_batch, os.path.join(args.output, "correlation.png"))
    print(f"  相关性图已保存: {args.output}/correlation.png")

    # 保存批量对比图
    plot_batch_comparison(centers, gt_batch, pred_batch,
                          os.path.join(args.output, "batch_comparison.png"), n_samples=8)
    print(f"  批量对比已保存: {args.output}/batch_comparison.png")

    # 保存单个样本对比图 (最好的和最差的)
    sorted_indices = np.argsort([m['RE'] for m in metrics_list])

    # 最好的4个
    best_indices = sorted_indices[:4]
    for rank, idx in enumerate(best_indices):
        metrics = metrics_list[idx]
        plot_comparison(centers, gt_batch[idx], pred_batch[idx],
                       os.path.join(args.output, "samples", f"best_{rank+1}_RE{metrics['RE']:.4f}.png"),
                       sample_id=idx, metrics=metrics)

    # 最差的4个
    worst_indices = sorted_indices[-4:][::-1]
    for rank, idx in enumerate(worst_indices):
        metrics = metrics_list[idx]
        plot_comparison(centers, gt_batch[idx], pred_batch[idx],
                       os.path.join(args.output, "samples", f"worst_{rank+1}_RE{metrics['RE']:.4f}.png"),
                       sample_id=idx, metrics=metrics)

    print(f"  样本对比已保存: {args.output}/samples/")

    print("\n" + "=" * 60)
    print("✅ 评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
