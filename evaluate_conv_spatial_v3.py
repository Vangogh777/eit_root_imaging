#!/usr/bin/env python3
"""
ConvSpatialEIT v3 评估流水线
=============================
在 4 个测试集上评估训练好的模型:
  - test:         标准测试集（随机噪声 -40~-20dB）
  - test_low_noise:     固定 -30dB 低噪声
  - test_high_noise:    固定 -15dB 高噪声
  - test_near_boundary: 固定 -25dB + 近边界含物

用法:
  python evaluate_conv_spatial_v3.py --checkpoint checkpoints/{run_id}/best.pt
  python evaluate_conv_spatial_v3.py --checkpoint checkpoints/{run_id}/best.pt --output results/conv_spatial_v3_eval
"""

import os, sys, json, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.conv_spatial_eit import ConvSpatialEIT
from data.datasets.eit_dataset import MemoryEITDataset
from data.eit_forward import EITForwardSolver


# ============ Metrics ============

def relative_error(pred, target):
    """每样本 RE: ||pred - target|| / ||target||"""
    B = pred.shape[0]
    err = torch.norm(pred.view(B, -1) - target.view(B, -1), dim=1)
    norm = torch.norm(target.view(B, -1), dim=1)
    return (err / (norm + 1e-8)).cpu().numpy()


def correlation_coefficient(pred, target):
    """每样本 CC"""
    B = pred.shape[0]
    pf = pred.view(B, -1)
    tf = target.view(B, -1)
    pc = pf - pf.mean(dim=1, keepdim=True)
    tc = tf - tf.mean(dim=1, keepdim=True)
    num = (pc * tc).sum(dim=1)
    den = torch.sqrt((pc ** 2).sum(dim=1) * (tc ** 2).sum(dim=1))
    return (num / (den + 1e-8)).cpu().numpy()


def compute_ssim(pred_np, target_np, mesh_nodes, mesh_elements, grid_size=128):
    """在插值到规则网格后计算 SSIM（2D 图像）"""
    try:
        from scipy.interpolate import griddata
        from skimage.metrics import structural_similarity
        centers = np.mean(mesh_nodes[mesh_elements], axis=1)[:, :2]
        x = np.linspace(-0.1, 0.1, grid_size)
        y = np.linspace(-0.1, 0.1, grid_size)
        X, Y = np.meshgrid(x, y)
        grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        # 圆形域掩码
        inside = np.sqrt(grid_pts[:, 0]**2 + grid_pts[:, 1]**2) < 0.098
        ssim_vals = []
        for i in range(len(pred_np)):
            pred_grid = griddata(centers, pred_np[i], grid_pts, method='linear', fill_value=0)
            target_grid = griddata(centers, target_np[i], grid_pts, method='linear', fill_value=0)
            # 重塑为 2D 图像
            p_img = pred_grid.reshape(grid_size, grid_size)
            t_img = target_grid.reshape(grid_size, grid_size)
            # 域外置 0 会影响 SSIM，但 skimage 要求 2D 输入
            lo, hi = p_img.min(), p_img.max()
            if hi - lo > 1e-8:
                p_img = (p_img - lo) / (hi - lo)
            lo, hi = t_img.min(), t_img.max()
            if hi - lo > 1e-8:
                t_img = (t_img - lo) / (hi - lo)
            s = structural_similarity(p_img, t_img, data_range=1.0, win_size=7)
            ssim_vals.append(s)
        return np.array(ssim_vals)
    except ImportError as e:
        print(f"  SSIM 失败 (缺少依赖): {e}")
        return np.full(len(pred_np), np.nan)


# ============ Visualization ============

def render_mesh_panel(ax, mesh_nodes, mesh_elements, values, title,
                       cmap="viridis", vmin=None, vmax=None):
    """在 FEM 三角网格上渲染电导率分布"""
    from matplotlib.tri import Triangulation
    triang = Triangulation(mesh_nodes[:, 0], mesh_nodes[:, 1], mesh_elements)
    im = ax.tripcolor(triang, facecolors=values, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="flat", edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=3)
    return im


def render_sample_row(ax_row, mesh_nodes, mesh_elements, target, pred, coarse, title,
                       vmin=0.0, vmax=0.12):
    """一行 4 面板: GT | Pred | Coarse | Error"""
    err = np.abs(pred - target)
    panels = [
        (target, "Ground Truth", "viridis", vmin, vmax),
        (pred, "Prediction", "viridis", vmin, vmax),
        (coarse, "Coarse (sigma_0)", "viridis", vmin, vmax),
        (err, "Abs Error", "hot", 0, max(err.max(), 0.005)),
    ]
    for i, (data, lbl, cmap, lo, hi) in enumerate(panels):
        im = render_mesh_panel(ax_row[i], mesh_nodes, mesh_elements, data,
                               lbl, cmap=cmap, vmin=lo, vmax=hi)
    ax_row[0].set_ylabel(f"RE={title:.4f}" if isinstance(title, float) else title, fontsize=9)


def generate_report(metrics_dict, num_samples, run_id="?"):
    """生成文本报告"""
    lines = [
        "ConvSpatialEIT v3 评估报告",
        "=" * 50,
        f"时间: {datetime.now().isoformat()}",
        f"Checkpoint: {run_id}",
        f"", 
    ]
    for split_name, m in metrics_dict.items():
        lines += [
            f"── {split_name} ──",
            f"  样本数: {num_samples.get(split_name, '?')}",
            f"  RE:  {m['RE']['mean']:.4f} ± {m['RE']['std']:.4f}",
            f"  CC:  {m['CC']['mean']:.4f} ± {m['CC']['std']:.4f}",
            f"  SSIM:{m['SSIM']['mean']:.4f} ± {m['SSIM']['std']:.4f}",
            f"",
        ]
    return "\n".join(lines)


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="ConvSpatialEIT v3 评估")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    parser.add_argument("--data", default="data/generated/mixed_dataset.h5", help="数据集路径")
    parser.add_argument("--mesh_config", default="config/mesh_config.yaml", help="网格配置")
    parser.add_argument("--output", default="results/conv_spatial_v3_eval", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=64, help="评估 batch size")
    parser.add_argument("--n_viz", type=int, default=6, help="每类可视化样本数")
    parser.add_argument("--device", default="cuda", help="设备")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    print(f"输出目录: {out_dir}")
    print(f"设备: {device}")

    # ── 1. 加载网格 ──
    print("加载网格...")
    solver = EITForwardSolver(args.mesh_config)
    centers = solver.element_centers[:, :2]
    elements = solver.mesh.element
    n_elems = solver.n_elems
    n_meas = solver.n_measurements
    n_freq = len(solver.frequencies)
    print(f"  单元: {n_elems}, 测量: {n_meas}, 频率: {n_freq}")

    # ── 2. 加载 checkpoint ──
    print(f"加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # 提取 state_dict
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("ema_model", ckpt.get("model", ckpt.get("model_state_dict", ckpt)))
    else:
        state_dict = ckpt
    # 提取超参
    hidden_dim = ckpt.get("hidden_dim", ckpt.get("gnn_hidden", 256))
    gnn_layers = ckpt.get("gnn_layers", 4)
    use_gat = ckpt.get("use_gat", True)
    n_heads = ckpt.get("n_heads", 4)
    run_id = ckpt.get("run_id", args.checkpoint.split("/")[-2] if "/" in args.checkpoint else "?")
    best_re = ckpt.get("best_re", "?")
    print(f"  hidden_dim={hidden_dim}, gnn_layers={gnn_layers}, GAT={use_gat}, heads={n_heads}")
    print(f"  best_re={best_re}, run_id={run_id}")

    # ── 3. 构建模型 ──
    model = ConvSpatialEIT(
        n_frequencies=n_freq, n_meas=n_meas, n_elems=n_elems,
        hidden_dim=hidden_dim, gnn_hidden=hidden_dim,
        gnn_layers=gnn_layers, use_gat=use_gat, n_heads=n_heads,
    )
    # 先 setup_mesh（构建 GNN 层）再加载权重
    model.setup_mesh(centers, elements)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # ── 4. 定义测试集 ──
    # 获取 mesh_nodes 用于 SSIM / 可视化
    mesh_nodes = solver.mesh.node
    if mesh_nodes.shape[1] > 2:
        mesh_nodes_2d = mesh_nodes[:, :2]
    else:
        mesh_nodes_2d = mesh_nodes

    test_splits = {
        "test": "test",
        "test_low_noise": "test_low_noise",
        "test_high_noise": "test_high_noise",
        "test_near_boundary": "test_near_boundary",
    }

    all_metrics = {}
    all_num_samples = {}

    for split_name, h5_group in test_splits.items():
        print(f"\n{'='*50}")
        print(f"评估: {split_name}")

        # 加载数据
        ds = MemoryEITDataset(args.data, split=h5_group, voltage_mask_ratio=0.0,
                               load_sigmas=True, load_masks=True)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        all_num_samples[split_name] = len(ds)
        print(f"  样本数: {len(ds)}")

        # 推理
        pred_list, target_list, coarse_list = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                V = batch["voltages"].to(device)  # (B, n_freq, n_meas)
                target = batch["sigmas"].to(device)
                out = model(V)
                pred_list.append(out["sigma"].cpu())
                target_list.append(target.cpu())
                coarse_list.append(out.get("sigma_0", torch.zeros_like(out["sigma"])).cpu())
        elapsed = time.time() - t0
        print(f"  推理耗时: {elapsed:.1f}s ({len(ds)/elapsed:.0f} samples/s)")

        pred = torch.cat(pred_list, dim=0)
        target = torch.cat(target_list, dim=0)
        coarse = torch.cat(coarse_list, dim=0)

        # 指标
        re_per = relative_error(pred, target)
        cc_per = correlation_coefficient(pred, target)
        ssim_per = compute_ssim(pred.numpy(), target.numpy(),
                                 mesh_nodes_2d, elements)
        # 粗估计指标（用于对比改进幅度）
        coarse_re_per = relative_error(coarse, target)

        metrics = {
            "RE": {"mean": float(re_per.mean()), "std": float(re_per.std())},
            "CC": {"mean": float(cc_per.mean()), "std": float(cc_per.std())},
            "SSIM": {"mean": float(np.nanmean(ssim_per)), "std": float(np.nanstd(ssim_per))},
            "Coarse_RE": {"mean": float(coarse_re_per.mean()), "std": float(coarse_re_per.std())},
        }
        all_metrics[split_name] = metrics
        print(f"  RE:  {re_per.mean():.4f} ± {re_per.std():.4f}")
        print(f"  CC:  {cc_per.mean():.4f} ± {cc_per.std():.4f}")
        print(f"  SSIM:{np.nanmean(ssim_per):.4f} ± {np.nanstd(ssim_per):.4f}")
        print(f"  Coarse RE: {coarse_re_per.mean():.4f} ± {coarse_re_per.std():.4f}")
        print(f"  改进: {(coarse_re_per.mean() - re_per.mean()):.4f} "
              f"({((coarse_re_per.mean() - re_per.mean()) / coarse_re_per.mean() * 100):.1f}%)")

        # ── 可视化（仅对 test 和 test_near_boundary 做详细图）──
        if split_name in ("test", "test_near_boundary"):
            split_out = os.path.join(out_dir, split_name)
            os.makedirs(split_out, exist_ok=True)
            sorted_idx = np.argsort(re_per)
            n_viz = min(args.n_viz, len(sorted_idx))
            best_idx = sorted_idx[:n_viz]
            worst_idx = sorted_idx[-n_viz:][::-1]
            vmin = float(target.min().item())
            vmax = float(target.max().item())

            # Best samples
            fig, axes = plt.subplots(n_viz, 4, figsize=(16, 3 * n_viz))
            for i, idx in enumerate(best_idx):
                render_sample_row(axes[i], mesh_nodes_2d, elements,
                                  target[idx].numpy(), pred[idx].numpy(), coarse[idx].numpy(),
                                  re_per[idx], vmin=vmin, vmax=vmax)
            plt.tight_layout()
            fig.savefig(os.path.join(split_out, "best_samples.png"), dpi=150)
            plt.close(fig)

            # Worst samples
            fig, axes = plt.subplots(n_viz, 4, figsize=(16, 3 * n_viz))
            for i, idx in enumerate(worst_idx):
                render_sample_row(axes[i], mesh_nodes_2d, elements,
                                  target[idx].numpy(), pred[idx].numpy(), coarse[idx].numpy(),
                                  re_per[idx], vmin=vmin, vmax=vmax)
            plt.tight_layout()
            fig.savefig(os.path.join(split_out, "worst_samples.png"), dpi=150)
            plt.close(fig)

            # RE histogram
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(re_per, bins=25, alpha=0.7, color="steelblue", edgecolor="white")
            ax.axvline(re_per.mean(), color="red", linestyle="--",
                       label=f"Mean RE={re_per.mean():.4f}")
            ax.set_xlabel("Relative Error")
            ax.set_ylabel("Count")
            ax.set_title(f"{split_name} — RE Distribution (n={len(re_per)})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(split_out, "re_histogram.png"), dpi=150)
            plt.close(fig)

            print(f"  可视化已保存到 {split_out}/")

    # ── 5. 保存汇总指标 ──
    # 主结果用 test 分片的指标（serve_results.py 需要 summary 字段）
    test_metrics = all_metrics.get("test", {})
    summary_metrics = {
        "summary": {
            "RE": test_metrics.get("RE", {"mean": 0, "std": 0}),
            "CC": test_metrics.get("CC", {"mean": 0, "std": 0}),
            "SSIM": test_metrics.get("SSIM", {"mean": 0, "std": 0}),
            "Coarse_RE": test_metrics.get("Coarse_RE", {"mean": 0, "std": 0}),
        },
        "run_id": run_id,
        "best_re": str(best_re),
        "timestamp": datetime.now().isoformat(),
        "model_params": n_params,
        "results": all_metrics,  # 4 个测试集的完整指标
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"\n指标已保存: {out_dir}/metrics.json")

    # ── 6. 汇总对比图：4个测试集的 RE 柱状图 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(test_splits.keys())
        means = [all_metrics[n]["RE"]["mean"] for n in names]
        stds = [all_metrics[n]["RE"]["std"] for n in names]
        cc_means = [all_metrics[n]["CC"]["mean"] for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        bars = ax1.bar(names, means, yerr=stds, capsize=5,
                       color=["#3b82f6", "#22c55e", "#ef4444", "#f59e0b"])
        ax1.set_ylabel("RE")
        ax1.set_title("Relative Error by Test Set")
        ax1.grid(True, alpha=0.3, axis="y")
        for bar, m in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{m:.4f}", ha="center", fontsize=9)

        bars2 = ax2.bar(names, cc_means, capsize=5,
                        color=["#3b82f6", "#22c55e", "#ef4444", "#f59e0b"])
        ax2.set_ylabel("CC")
        ax2.set_title("Correlation Coefficient by Test Set")
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, m in zip(bars2, cc_means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{m:.4f}", ha="center", fontsize=9)

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "summary_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"汇总对比图: {out_dir}/summary_comparison.png")
    except Exception as e:
        print(f"汇总图生成失败: {e}")

    # ── 7. 报告 ──
    report = generate_report(all_metrics, all_num_samples, run_id)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report)
    print(f"\n{report}")
    print(f"\n全部完成! 结果保存在 {out_dir}/")


if __name__ == "__main__":
    main()
