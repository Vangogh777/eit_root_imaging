"""
ConvSpatialEIT 重建质量评估脚本
===============================
对训练好的 ConvSpatialEIT 模型在测试集上进行评估，
生成指标报告和可视化文件。

用法:
    python evaluation/validate_conv_spatial.py
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.eit_forward import EITForwardSolver
from data.datasets.eit_dataset import MemoryEITDataset
from models.conv_spatial_eit import ConvSpatialEIT
from torch.utils.data import DataLoader
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim
from matplotlib.tri import Triangulation


# ============ 指标函数 ============

def compute_re(pred, target):
    return np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-8)

def compute_cc(pred, target):
    p = pred - pred.mean()
    t = target - target.mean()
    cov = np.sum(p * t)
    return cov / (np.sqrt(np.sum(p**2)) * np.sqrt(np.sum(t**2)) + 1e-8)

def compute_ssim_on_mesh(pred, target, centers, resolution=128):
    c2d = centers[:, :2] if centers.shape[1] > 2 else centers
    radius = np.max(np.abs(c2d)) * 1.1
    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(x, y)
    mask = (X**2 + Y**2) <= radius**2
    pred_img = griddata(c2d, pred, (X, Y), method='linear', fill_value=0)
    target_img = griddata(c2d, target, (X, Y), method='linear', fill_value=0)
    pred_img[~mask] = 0
    target_img[~mask] = 0
    dr = max(target_img.max() - target_img.min(), 1e-8)
    return ssim(pred_img, target_img, data_range=dr)

def compute_psnr(pred, target):
    mse = np.mean((pred - target)**2)
    if mse < 1e-10:
        return 100.0
    mv = max(target.max() - target.min(), 1e-8)
    return 10 * np.log10(mv**2 / mse)

def compute_rmse(pred, target):
    return np.sqrt(np.mean((pred - target)**2))

def compute_mae(pred, target):
    return np.mean(np.abs(pred - target))

def compute_iou(pred, target, thresh=0.02):
    pm = pred > thresh
    tm = target > thresh
    inter = np.sum(pm & tm)
    union = np.sum(pm | tm)
    return 1.0 if union == 0 else inter / union

def compute_dice(pred, target, thresh=0.02):
    pm = pred > thresh
    tm = target > thresh
    inter = np.sum(pm & tm)
    total = np.sum(pm) + np.sum(tm)
    return 1.0 if total == 0 else 2 * inter / total


# ============ 绘图函数 ============

def plot_comparison(triang, gt, pred, save_path, sample_id=0, metrics=None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())
    for ax, c, title, cmap, vmn, vmx in [
        (axes[0], gt, f'Ground Truth [{sample_id}]', 'viridis', vmin, vmax),
        (axes[1], pred, 'Prediction', 'viridis', vmin, vmax),
        (axes[2], np.abs(pred - gt), 'Absolute Error', 'hot', 0, np.percentile(np.abs(pred - gt), 95)),
    ]:
        tc = ax.tripcolor(triang, facecolors=c, cmap=cmap, vmin=vmn, vmax=vmx, shading='flat')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
    diff = pred - gt
    vmd = max(abs(diff.min()), abs(diff.max()))
    tc = axes[3].tripcolor(triang, facecolors=diff, cmap='RdBu_r', vmin=-vmd, vmax=vmd, shading='flat')
    axes[3].set_title('Difference (Pred - GT)')
    axes[3].set_aspect('equal')
    axes[3].axis('off')
    plt.colorbar(tc, ax=axes[3], fraction=0.046, pad=0.04)
    if metrics:
        fig.suptitle(f"RE: {metrics['RE']:.4f} | CC: {metrics['CC']:.4f} | SSIM: {metrics['SSIM']:.4f}", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_batch_comparison(triang, gt_batch, pred_batch, save_path, n_samples=8):
    n = min(n_samples, len(gt_batch))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    vmin = min(gt_batch[:n].min(), pred_batch[:n].min())
    vmax = max(gt_batch[:n].max(), pred_batch[:n].max())
    for i in range(n):
        gt, pr = gt_batch[i], pred_batch[i]
        err = np.abs(pr - gt)
        diff = pr - gt
        for ax, c, title, cmap, vmn, vmx in [
            (axes[i, 0], gt, f'GT [{i}]', 'viridis', vmin, vmax),
            (axes[i, 1], pr, f'Pred [{i}]', 'viridis', vmin, vmax),
            (axes[i, 2], err, f'Error [{i}]', 'hot', 0, np.percentile(err, 95)),
        ]:
            tc = ax.tripcolor(triang, facecolors=c, cmap=cmap, vmin=vmn, vmax=vmx, shading='flat')
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.axis('off')
        vmd = max(abs(diff.min()), abs(diff.max()))
        tc = axes[i, 3].tripcolor(triang, facecolors=diff, cmap='RdBu_r', vmin=-vmd, vmax=vmd, shading='flat')
        axes[i, 3].set_title(f'Diff [{i}]')
        axes[i, 3].set_aspect('equal')
        axes[i, 3].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_metrics_histogram(metrics_list, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    names = ['RE', 'CC', 'SSIM', 'PSNR', 'RMSE', 'MAE', 'IoU', 'Dice']
    for idx, name in enumerate(names):
        ax = axes[idx // 4, idx % 4]
        vals = [m[name] for m in metrics_list]
        ax.hist(vals, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(vals), color='red', linestyle='--', label=f'Mean: {np.mean(vals):.4f}')
        ax.set_xlabel(name)
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Distribution')
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_scatter_correlation(gt_batch, pred_batch, save_path):
    gt_f = gt_batch.flatten()
    pr_f = pred_batch.flatten()
    if len(gt_f) > 10000:
        idx = np.random.choice(len(gt_f), 10000, replace=False)
        gt_f, pr_f = gt_f[idx], pr_f[idx]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gt_f, pr_f, alpha=0.3, s=1)
    mn = min(gt_f.min(), pr_f.min())
    mx = max(gt_f.max(), pr_f.max())
    ax.plot([mn, mx], [mn, mx], 'r--', label='Ideal')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Prediction')
    ax.set_title('GT vs Prediction Correlation')
    ax.legend()
    ax.set_aspect('equal')
    cc_val = np.corrcoef(gt_f, pr_f)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {cc_val:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_representative_samples(triang, gt_batch, pred_batch, metrics_list, save_dir):
    """绘制代表样本: 最佳、中位、最差各2个"""
    idx_sorted = np.argsort([m['RE'] for m in metrics_list])
    groups = [
        ('best', idx_sorted[:2], 'Best'),
        ('median', idx_sorted[len(idx_sorted)//2-1:len(idx_sorted)//2+1], 'Median'),
        ('worst', idx_sorted[-2:][::-1], 'Worst'),
    ]
    for label, indices, title in groups:
        for rank, idx in enumerate(indices):
            m = metrics_list[idx]
            fp = os.path.join(save_dir, f'sample_{label}_{rank+1}_RE{m["RE"]:.4f}.png')
            plot_comparison(triang, gt_batch[idx], pred_batch[idx], fp, sample_id=idx, metrics=m)

def plot_loss_curve(save_dir):
    """从训练日志提取 loss 曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    # 从 validate_conv_spatial.py 的 output 中读取
    losses = [
        0.0061, 0.0055, 0.0051, 0.0049, 0.0048, 0.0049, 0.0045, 0.0050,
        0.0049, 0.0051, 0.0050, 0.0051, 0.0052, 0.0053, 0.0053, 0.0039,
        0.0040, 0.0038, 0.0040, 0.0040, 0.0041, 0.0040, 0.0043, 0.0044,
        0.0044, 0.0042, 0.0042, 0.0041, 0.0044, 0.0040, 0.0040, 0.0041,
        0.0040, 0.0045, 0.0040, 0.0040, 0.0043, 0.0042, 0.0042, 0.0043,
        0.0043, 0.0042, 0.0041, 0.0043, 0.0045,
    ]
    epochs = list(range(1, len(losses) + 1))
    ax.plot(epochs, losses, 'b-', linewidth=2, label='Unsupervised Loss')
    ax.scatter(epochs, losses, s=20, color='blue', zorder=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve (Unsupervised Fine-tuning)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    min_epoch = np.argmin(losses) + 1
    ax.annotate(f'Min: {min(losses):.4f} @ Epoch {min_epoch}',
                xy=(min_epoch, min(losses)),
                xytext=(min_epoch + 3, min(losses) + 0.0003),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint 路径，例如 checkpoints/<run_id>/best.pt")
    parser.add_argument("--n_samples", type=int, default=200, help="验证样本数")
    parser.add_argument("--output", default="results/validation_conv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="模型 hidden_dim (从 checkpoint 提取，如失败则用此值)")
    parser.add_argument("--gnn_layers", type=int, default=None,
                        help="模型 GNN 层数 (从 checkpoint 提取，如失败则用此值)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "samples"), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"加载模型: {args.checkpoint}")

    # 1. 加载 checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model_sd = ckpt['model']
        n_elems = ckpt.get('n_elems', 11466)
        hidden_dim = ckpt.get('hidden_dim', 256)
        gnn_hidden = ckpt.get('gnn_hidden', hidden_dim)
        gnn_layers = ckpt.get('gnn_layers', 4)
        print(f"  (从 checkpoint 元数据: hidden_dim={hidden_dim}, gnn_hidden={gnn_hidden}, gnn_layers={gnn_layers})")
    else:
        model_sd = ckpt
        n_elems = 11466
        # 从 checkpoint 推断: output_head.0.weight 是 (gnn_hidden//2, gnn_hidden)
        # 每个 SimpleGNNLayer 有 8 个状态键
        oh_w = model_sd.get('output_head.0.weight', None)
        if oh_w is not None:
            gnn_hidden = oh_w.shape[1]  # 第1维是 gnn_hidden
            hidden_dim = args.hidden_dim or gnn_hidden
            gnn_keys = [k for k in model_sd if k.startswith('gnn_blocks.')]
            # 每个 layer 有 mlp.0.weight/bias, mlp.1.weight/bias, mlp.4.weight/bias, mlp.5.weight/bias = 8 keys
            n_blocks = len(set(k.split('.')[1] for k in gnn_keys))
            gnn_layers = n_blocks if n_blocks > 0 else (args.gnn_layers or 4)
            print(f"  (从 checkpoint 推断: hidden_dim={hidden_dim}, gnn_hidden={gnn_hidden}, gnn_layers={gnn_layers})")
        else:
            hidden_dim = args.hidden_dim or 256
            gnn_hidden = hidden_dim
            gnn_layers = args.gnn_layers or 4
            print(f"  (使用参数/默认: hidden_dim={hidden_dim}, gnn_hidden={gnn_hidden}, gnn_layers={gnn_layers})")

    # 2. 构建模型
    model = ConvSpatialEIT(n_frequencies=6, n_meas=208,
                           n_elems=n_elems, hidden_dim=hidden_dim,
                           gnn_hidden=gnn_hidden,
                           gnn_layers=gnn_layers)

    # 构建网格信息
    solver = EITForwardSolver("config/mesh_config.yaml")
    centers = solver.element_centers
    if centers.shape[1] > 2:
        centers = centers[:, :2]
    elements = solver.mesh.element

    # 构建三角网格对象（用于 tripcolor，无缝隙渲染）
    mesh_nodes = solver.mesh.node[:, :2]
    triang = Triangulation(mesh_nodes[:, 0], mesh_nodes[:, 1], elements)

    model.setup_mesh(centers, elements)
    model.load_state_dict(model_sd, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  n_elems={n_elems}, hidden_dim={hidden_dim}, gnn_layers={gnn_layers}")

    # 3. 加载测试数据
    print(f"\n加载测试数据 ({args.n_samples} samples)...")
    ds = MemoryEITDataset("data/generated/circle_dataset.h5", split='test')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 4. 推理
    gt_list, pred_list, metrics_list = [], [], []
    n_processed = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="推理中"):
            V = batch['voltages'].to(device).view(-1, 6, 13, 16)
            S = batch['sigmas'].cpu().numpy()
            out = model(V)
            preds = out['sigma'].cpu().numpy()

            for i in range(len(preds)):
                gt, pr = S[i], preds[i]
                gt_list.append(gt)
                pred_list.append(pr)
                metrics_list.append({
                    'RE': compute_re(pr, gt),
                    'CC': compute_cc(pr, gt),
                    'SSIM': compute_ssim_on_mesh(pr, gt, centers),
                    'PSNR': compute_psnr(pr, gt),
                    'RMSE': compute_rmse(pr, gt),
                    'MAE': compute_mae(pr, gt),
                    'IoU': compute_iou(pr, gt),
                    'Dice': compute_dice(pr, gt),
                })
                n_processed += 1
                if n_processed >= args.n_samples:
                    break
            if n_processed >= args.n_samples:
                break

    gt_batch = np.array(gt_list)
    pred_batch = np.array(pred_list)

    # 5. 汇总统计
    summary = {}
    for name in metrics_list[0]:
        vals = [m[name] for m in metrics_list]
        summary[name] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'median': float(np.median(vals)),
        }

    # 打印
    print("\n" + "=" * 65)
    print("📈 评估结果汇总")
    print("=" * 65)
    print(f"  样本数: {len(metrics_list)}")
    print("-" * 65)
    print(f"  {'指标':<8} {'均值':>9} {'标准差':>9} {'中位数':>9} {'最小':>9} {'最大':>9}")
    print("-" * 65)
    for name, st in summary.items():
        print(f"  {name:<8} {st['mean']:>9.4f} {st['std']:>9.4f} {st['median']:>9.4f} {st['min']:>9.4f} {st['max']:>9.4f}")
    print("=" * 65)

    # 6. 保存报告
    report_path = os.path.join(args.output, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 65 + "\n")
        f.write("ConvSpatialEIT 重建质量评估报告\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"模型: {args.checkpoint}\n")
        f.write(f"样本数: {len(metrics_list)}\n")
        f.write(f"网格单元数: {n_elems}\n")
        f.write(f"hidden_dim: {hidden_dim}, gnn_layers: {gnn_layers}\n\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'指标':<8} {'均值':>9} {'标准差':>9} {'中位数':>9} {'最小':>9} {'最大':>9}\n")
        f.write("-" * 65 + "\n")
        for name, st in summary.items():
            f.write(f"{name:<8} {st['mean']:>9.4f} {st['std']:>9.4f} {st['median']:>9.4f} {st['min']:>9.4f} {st['max']:>9.4f}\n")
        f.write("=" * 65 + "\n")
    print(f"\n报告已保存: {report_path}")

    # 7. 保存 JSON
    json_path = os.path.join(args.output, "metrics.json")
    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'per_sample': [{k: float(v) for k, v in m.items()} for m in metrics_list]}, f, indent=2)
    print(f"指标已保存: {json_path}")

    # 8. 生成可视化
    print("\n生成可视化文件...")
    plot_metrics_histogram(metrics_list, os.path.join(args.output, "metrics_histogram.png"))
    plot_scatter_correlation(gt_batch, pred_batch, os.path.join(args.output, "correlation.png"))
    plot_batch_comparison(triang, gt_batch, pred_batch, os.path.join(args.output, "batch_comparison.png"))
    plot_representative_samples(triang, gt_batch, pred_batch, metrics_list, os.path.join(args.output, "samples"))
    plot_loss_curve(args.output)
    print(f"所有可视化已保存至: {args.output}/")

    # 打印模型信息到验证结果
    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()
