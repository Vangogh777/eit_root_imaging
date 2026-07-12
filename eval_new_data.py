"""
全面评估脚本：对最新训练的 ConvSpatialEIT 模型
评估所有测试子集并生成可视化
"""
import os, sys, json, numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.conv_spatial_eit import ConvSpatialEIT
from data.eit_forward import EITForwardSolver
from data.datasets.eit_dataset import MemoryEITDataset


def compute_metrics(pred, gt):
    """计算每个样本的 RE 和 CC"""
    B = pred.shape[0]
    re = torch.norm(pred - gt, dim=-1) / (torch.norm(gt, dim=-1) + 1e-8)
    cc_list = []
    for i in range(B):
        p = pred[i] - pred[i].mean()
        g = gt[i] - gt[i].mean()
        c = (p * g).sum() / (p.norm() * g.norm() + 1e-8)
        cc_list.append(c)
    cc = torch.tensor(cc_list)
    return re, cc


def evaluate_model(model, data_path, split, device, label="", n_show=6):
    """评估模型在指定 split 上的性能"""
    ds = MemoryEITDataset(data_path, split=split, voltage_mask_ratio=0.0)
    loader = DataLoader(ds, batch_size=16, shuffle=False)  # GATv2 显存占用大，用小 batch
    print(f"\n{'='*60}")
    print(f"评估: {label} | split={split} | 样本数={len(ds)}")
    print(f"{'='*60}")

    all_pred, all_gt = [], []
    with torch.no_grad():
        for batch in loader:
            V = batch['voltages'].to(device).view(-1, 6, 13, 16)
            out = model(V)
            all_pred.append(out['sigma'].cpu())
            all_gt.append(batch['sigmas'])

    all_pred = torch.cat(all_pred)
    all_gt = torch.cat(all_gt)

    re, cc = compute_metrics(all_pred, all_gt)

    # 统计指标
    result = {
        'split': split,
        'n_samples': len(ds),
        're_mean': re.mean().item(),
        're_std': re.std().item(),
        're_min': re.min().item(),
        're_max': re.max().item(),
        're_median': re.median().item(),
        'cc_mean': cc.mean().item(),
        'cc_std': cc.std().item(),
        'cc_min': cc.min().item(),
        'cc_max': cc.max().item(),
    }

    print(f"  RE:  mean={result['re_mean']:.4f} ± {result['re_std']:.4f}")
    print(f"       min={result['re_min']:.4f}, max={result['re_max']:.4f}, median={result['re_median']:.4f}")
    print(f"  CC:  mean={result['cc_mean']:.4f} ± {result['cc_std']:.4f}")
    print(f"       min={result['cc_min']:.4f}, max={result['cc_max']:.4f}")

    return result, all_pred, all_gt, re, cc


def plot_samples(mesh_nodes, mesh_elements, all_pred, all_gt, re_vals,
                 output_path, title="", n_show=8):
    """绘制预测 vs GT 对比图"""
    # 排序：选最好和最差的
    indices = torch.argsort(re_vals)
    half = n_show // 2
    show_idx = list(indices[:half]) + list(indices[-half:])

    fig = plt.figure(figsize=(14, 2.5 + len(show_idx) * 2.8))
    fig.patch.set_facecolor('white')

    # 标题
    ax_title = fig.add_axes([0.05, 0.96, 0.9, 0.03])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, title, fontsize=13, ha='center', va='center',
                  fontweight='bold', color='#222')

    for row, idx in enumerate(show_idx):
        gt = all_gt[idx].numpy()
        pred = all_pred[idx].numpy()
        err = np.abs(pred - gt)
        re_i = re_vals[idx].item()

        y0 = 0.93 - row * 0.28
        for j, (data, cmap, vmin, vmax, label) in enumerate([
            (gt, 'viridis', 0.008, 0.055, 'Ground Truth'),
            (pred, 'viridis', 0.008, 0.055, 'Prediction'),
            (err, 'hot', 0, 0.01, f'Error | RE={re_i:.4f}'),
        ]):
            ax = fig.add_axes([0.02 + j * 0.33, y0 - 0.23, 0.30, 0.23])
            ax.tripcolor(mesh_nodes[:, 0], mesh_nodes[:, 1],
                         mesh_elements, facecolors=data,
                         cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
            ax.set_aspect('equal')
            ax.axis('off')
            if row == 0:
                ax.set_title(label, fontsize=10, fontweight='bold', pad=1)

    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {output_path}")


def plot_re_histogram(re_dict, output_path, title=""):
    """绘制各子集 RE 分布直方图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    for i, (split, re_vals) in enumerate(re_dict.items()):
        ax.hist(re_vals.numpy(), bins=30, alpha=0.6, color=colors[i % len(colors)],
                label=f"{split} (mean={re_vals.mean():.4f})")

    ax.set_xlabel('Relative Error (RE)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {output_path}")


def plot_loss_curves(epochs_file, output_path):
    """从训练记录绘制损失曲线"""
    import json
    sup_epochs, unsup_epochs = [], []
    sup_re, unsup_re = [], []
    sup_loss, unsup_loss = [], []

    with open(epochs_file) as f:
        for line in f:
            d = json.loads(line)
            if d['phase'] == 'supervised':
                sup_epochs.append(d['epoch'])
                sup_re.append(d.get('re', 0))
                sup_loss.append(d.get('loss', 0))
            elif d['phase'] == 'unsupervised':
                unsup_epochs.append(d['epoch'])
                unsup_re.append(d.get('val_re', 0))
                unsup_loss.append(d.get('loss', 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.patch.set_facecolor('white')

    # RE 曲线
    ax = axes[0]
    if sup_epochs:
        ax.plot(sup_epochs, sup_re, 'b-o', markersize=3, label='Supervised RE', linewidth=1.5)
    if unsup_epochs:
        ax.plot([e + max(sup_epochs) for e in unsup_epochs], unsup_re,
                'r-s', markersize=3, label='Unsupervised RE', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('RE', fontsize=11)
    ax.set_title('Validation RE', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Loss 曲线
    ax = axes[1]
    if sup_epochs:
        ax.plot(sup_epochs, sup_loss, 'b-o', markersize=3, label='Supervised Loss', linewidth=1.5)
    if unsup_epochs:
        ax.plot([e + max(sup_epochs) for e in unsup_epochs], unsup_loss,
                'r-s', markersize=3, label='Unsupervised Loss', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if max(sup_loss + unsup_loss) > min(sup_loss + unsup_loss) * 10:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {output_path}")


def main():
    # ========== 配置 ==========
    ckpt_path = "checkpoints/20260624_225603_v2_both_hd256/best.pt"
    # 同时也评估旧模型（如果文件存在且 n_elems 匹配）
    old_ckpt_path = "checkpoints/conv_spatial_best.pt"
    mesh_config = "config/mesh_config.yaml"
    data_path = "data/generated/mixed_dataset.h5"
    output_dir = "results/eval_new_data"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")

    # ========== 获取网格 ==========
    solver = EITForwardSolver(mesh_config)
    centers = solver.element_centers[:, :2].copy()
    n_elems = solver.n_elems
    mesh_nodes = solver.mesh.node[:, :2].copy()
    mesh_elements = solver.mesh.element.copy()
    print(f"网格: {n_elems} 单元, {mesh_nodes.shape[0]} 节点")

    # ========== 加载新模型 ==========
    print(f"\n加载新模型: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 加载 Jacobian（如果模型使用了）
    jac_path = "data/generated/jacobian.npy"
    model_jacobian = None
    if os.path.exists(jac_path):
        jac = np.load(jac_path)
        if jac.ndim == 2:
            model_jacobian = jac  # (n_meas, n_elems)
        elif jac.ndim == 3:
            model_jacobian = jac[0]  # 取第1频率
        print(f"  Jacobian 已加载: {model_jacobian.shape}")

    model = ConvSpatialEIT(
        n_frequencies=6, n_meas=208, n_elems=n_elems,
        hidden_dim=ckpt.get('hidden_dim', 256),
        gnn_hidden=ckpt.get('gnn_hidden', 256),
        gnn_layers=ckpt.get('gnn_layers', 4),
        use_gat=ckpt.get('use_gat', True),
        n_heads=ckpt.get('n_heads', 4),
    )
    model.setup_mesh(centers, solver.mesh.element, jacobian=model_jacobian)

    # 加载 EMA 模型权重（最佳来源通常是 EMA）
    if 'ema_model' in ckpt:
        from torch.optim.swa_utils import AveragedModel
        ema_model = AveragedModel(model)
        ema_model.load_state_dict(ckpt['ema_model'])
        model.load_state_dict(ema_model.module.state_dict())
        print(f"  使用 EMA 模型权重 (保存时 epoch={ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(ckpt.get('model', ckpt))
        print(f"  使用 raw 模型权重")
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # ========== 识别 checkpoint 来源 ==========
    run_id = ckpt.get('run_id', 'unknown')
    ckpt_epoch = ckpt.get('epoch', '?')
    ckpt_val_re = ckpt.get('val_re', '?')
    ckpt_best_re = ckpt.get('best_re', '?')
    ckpt_mode = ckpt.get('mode', '?')
    print(f"  run_id: {run_id}, epoch: {ckpt_epoch}, val_re: {ckpt_val_re}, best_re: {ckpt_best_re}, mode: {ckpt_mode}")

    # ========== 评估所有子集 ==========
    splits = ['val', 'test', 'test_high_noise', 'test_low_noise', 'test_near_boundary']
    all_results = {}
    all_re_dict = {}
    summary_rows = []

    for split in splits:
        if split not in ['val', 'test']:
            # test subsets are in the mixed_dataset.h5
            pass
        result, pred, gt, re, cc = evaluate_model(
            model, data_path, split, device,
            label=f"新模型 ({run_id})", n_show=6)

        all_results[split] = result
        all_re_dict[split] = re
        summary_rows.append(result)

        # 绘制样本对比图
        plot_samples(
            mesh_nodes, mesh_elements, pred, gt, re,
            f"{output_dir}/{split}_samples.png",
            title=f"Split: {split} | RE={result['re_mean']:.4f}±{result['re_std']:.4f} | CC={result['cc_mean']:.4f}",
            n_show=8,
        )

    # ========== RE 分布直方图 ==========
    plot_re_histogram(all_re_dict, f"{output_dir}/re_histogram.png",
                      title=f"RE Distribution by Split (model: {run_id[:16]}...)")

    # ========== 摘要 ==========
    print(f"\n{'='*60}")
    print(f"评估摘要")
    print(f"{'='*60}")
    print(f"{'Split':<25} {'RE mean':<12} {'RE std':<12} {'CC mean':<12} {'n':<8}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for r in summary_rows:
        print(f"{r['split']:<25} {r['re_mean']:<12.4f} {r['re_std']:<12.4f} {r['cc_mean']:<12.4f} {r['n_samples']:<8}")

    # 保存摘要到 JSON
    summary_path = f"{output_dir}/summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "model": {
                "run_id": run_id,
                "ckpt_path": ckpt_path,
                "n_params": n_params,
                "hidden_dim": ckpt.get('hidden_dim', 256),
                "gnn_hidden": ckpt.get('gnn_hidden', 256),
                "gnn_layers": ckpt.get('gnn_layers', 4),
                "use_gat": ckpt.get('use_gat', True),
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\n摘要已保存: {summary_path}")

    # ========== 训练曲线 ==========
    epochs_file = f"training_records/{run_id}/epochs.jsonl"
    if os.path.exists(epochs_file):
        plot_loss_curves(epochs_file, f"{output_dir}/loss_curves.png")

    print(f"\n✅ 所有结果已保存到 {output_dir}/")


if __name__ == "__main__":
    main()
