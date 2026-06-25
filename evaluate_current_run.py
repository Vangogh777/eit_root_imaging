"""
评估当前训练 (20260622_015538_v2_both_hd256)
加载 best.pt 的 meta 信息自动构建正确模型
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

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/20260622_015538_v2_both_hd256/best.pt")
parser.add_argument("--data", default="data/generated/mixed_dataset.h5")
parser.add_argument("--mesh_config", default="config/mesh_config.yaml")
parser.add_argument("--split", default="test")
parser.add_argument("--output", default="results/eval_20260622_both_hd256")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")
os.makedirs(args.output, exist_ok=True)

# ── 1. 从 checkpoint 读取配置 ──
ckpt = torch.load(args.checkpoint, map_location='cpu')
if isinstance(ckpt, dict):
    hidden_dim = ckpt.get('hidden_dim', 256)
    gnn_hidden = ckpt.get('gnn_hidden', 256)
    gnn_layers = ckpt.get('gnn_layers', 4)
    use_gat = ckpt.get('use_gat', False)
    n_heads = ckpt.get('n_heads', 4)
else:
    hidden_dim = 256; gnn_hidden = 256; gnn_layers = 4
    use_gat = False; n_heads = 4
print(f"模型配置: hidden_dim={hidden_dim} gnn_hidden={gnn_hidden} "
      f"layers={gnn_layers} use_gat={use_gat} heads={n_heads}")

# ── 2. 加载网格 ──
solver = EITForwardSolver(args.mesh_config)
centers = solver.element_centers[:, :2]
n_elems = solver.n_elems
mesh_nodes = solver.mesh.node[:, :2]
mesh_elements = solver.mesh.element
print(f"网格: {n_elems} 单元, {mesh_nodes.shape[0]} 节点")

# ── 3. 构建模型 ──
model = ConvSpatialEIT(
    n_elems=n_elems,
    hidden_dim=hidden_dim,
    gnn_hidden=gnn_hidden,
    gnn_layers=gnn_layers,
    use_gat=use_gat,
    n_heads=n_heads,
)
model.setup_mesh(centers, solver.mesh.element)

# 加载权重
if isinstance(ckpt, dict) and 'ema_model' in ckpt:
    from torch.optim.swa_utils import AveragedModel
    ema_model = AveragedModel(model)
    ema_model.load_state_dict(ckpt['ema_model'])
    model.load_state_dict(ema_model.module.state_dict())
    print("加载 EMA 模型权重")
elif isinstance(ckpt, dict) and 'model' in ckpt:
    model.load_state_dict(ckpt['model'])
    print("加载 raw 模型权重")
else:
    model.load_state_dict(ckpt)
    print("加载直接 state_dict")
model.to(device).eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {n_params:,}")

# ── 4. 加载数据 ──
print(f"\n加载数据 ({args.split})...")
ds = MemoryEITDataset(args.data, split=args.split, voltage_mask_ratio=0.0)
batch_size = 32
loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
print(f"共 {len(ds)} 个样本, batch_size={batch_size}")

# ── 5. 推理 ──
print("推理中...")
all_pred, all_gt = [], []
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

all_pred = np.concatenate(all_pred, axis=0)
all_gt = np.concatenate(all_gt, axis=0)
avg_time = np.mean(times) / batch_size * 1000
print(f"推理速度: {avg_time:.2f} ms/样本")

# ── 6. 逐样本指标 ──
print("\n计算指标...")
re_list, cc_list = [], []
for i in range(len(all_pred)):
    p, g = all_pred[i], all_gt[i]
    re = np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-8)
    re_list.append(re)
    p_m = p - p.mean(); g_m = g - g.mean()
    cc = (p_m * g_m).sum() / (np.linalg.norm(p_m) * np.linalg.norm(g_m) + 1e-8)
    cc_list.append(cc)

# 按形状分组统计（通过检查 sigma 分布判断形状类型）
# 简单分组：按 sigma 的方差大致区分单内含物/多内含物
simple_groups = {'all': re_list}

metrics = {
    'RE': {'mean': float(np.mean(re_list)), 'std': float(np.std(re_list)),
           'min': float(np.min(re_list)), 'max': float(np.max(re_list))},
    'CC': {'mean': float(np.mean(cc_list)), 'std': float(np.std(cc_list))},
    'n_samples': len(all_pred),
    'n_params': n_params,
    'avg_inference_ms': float(avg_time),
}

# ── 输出 ──
print(f"\n{'='*55}")
print(f"  验证结果 — {args.split.upper()} 集")
print(f"{'='*55}")
print(f"  RE    : mean={metrics['RE']['mean']:.4f}  std={metrics['RE']['std']:.4f}")
print(f"          [{metrics['RE']['min']:.4f}, {metrics['RE']['max']:.4f}]")
print(f"  CC    : mean={metrics['CC']['mean']:.4f}  std={metrics['CC']['std']:.4f}")
print(f"  推理   : {avg_time:.2f} ms/样本")
print(f"{'='*55}\n")

# 保存指标
with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
    json.dump({'summary': metrics}, f, indent=2)
with open(os.path.join(args.output, 'report.txt'), 'w') as f:
    f.write(f"模型: {args.checkpoint}\n")
    f.write(f"数据集: {args.data} ({args.split})\n\n")
    f.write(f"  RE   = {metrics['RE']['mean']:.4f} ± {metrics['RE']['std']:.4f}\n")
    f.write(f"  CC   = {metrics['CC']['mean']:.4f} ± {metrics['CC']['std']:.4f}\n")
    f.write(f"  推理 = {avg_time:.2f} ms/样本\n")

# ── 7. 可视化 ──
print("生成可视化...")
n_vis = min(8, len(all_pred))

# 找到 best 4 和 worst 4
errors = np.mean((all_pred - all_gt) ** 2, axis=1)
best_idx = np.argsort(errors)[:4]
worst_idx = np.argsort(errors)[-4:]
mid_idx = np.argsort(errors)[len(errors)//2 - 2: len(errors)//2 + 2]

# 7a. 连续 8 个样本对比
fig, axes = plt.subplots(3, n_vis, figsize=(4 * n_vis, 10))
fig.patch.set_facecolor('white')
for i in range(n_vis):
    gt, pred = all_gt[i], all_pred[i]
    err = np.abs(pred - gt)
    re = re_list[i]
    
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
    axes[2, i].set_title(f'Error RE={re:.4f}', fontsize=8); axes[2, i].axis('off')
    axes[2, i].set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'batch_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7b. Best 4
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

# 7c. Worst 4
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

# 7d. Mid 4
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.patch.set_facecolor('white')
for col, idx in enumerate(mid_idx):
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
        if row == 0: axes[row, col].set_title(f'Mid #{col+1}', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(args.output, 'mid_4.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7e. 相关性散点图
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

# 7f. RE 分布直方图
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor('white')
ax.hist(re_list, bins=30, color='#3b82f6', alpha=0.7, edgecolor='white')
ax.axvline(metrics['RE']['mean'], color='red', linestyle='--', label=f"mean={metrics['RE']['mean']:.4f}")
ax.axvline(metrics['RE']['mean'] - metrics['RE']['std'], color='orange', linestyle=':', label=f"±1σ")
ax.axvline(metrics['RE']['mean'] + metrics['RE']['std'], color='orange', linestyle=':')
ax.set_xlabel('RE'); ax.set_ylabel('Count')
ax.set_title(f'RE Distribution — {metrics["RE"]["mean"]:.4f} ± {metrics["RE"]["std"]:.4f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.output, 're_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7g. 形状分组 RE 箱线图（简单分组：按 sigma 均值聚类）
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor('white')
# 按 sigma 的 kurtosis 分 4 组
from scipy.stats import kurtosis
kurt_values = [kurtosis(g) for g in all_gt]
groups = {'group_' + str(i): [] for i in range(4)}
for i, k in enumerate(kurt_values):
    gid = min(3, int(k / 2 + 2))
    groups['group_' + str(gid)].append(re_list[i])
labels = [f'{k}\nn={len(v)}' for k, v in groups.items() if v]
data = [v for v in groups.values() if v]
bp = ax.boxplot(data, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('RE'); ax.set_title('RE by Shape Group')
plt.tight_layout()
plt.savefig(os.path.join(args.output, 're_by_group.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ 完成！结果保存至: {args.output}/")
print(f"   指标: metrics.json, report.txt")
print(f"   可视化: batch_comparison.png")
print(f"          best_4.png / mid_4.png / worst_4.png")
print(f"          correlation.png / re_distribution.png / re_by_group.png")
