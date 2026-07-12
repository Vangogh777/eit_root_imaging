"""
完整可视化：有监督 vs 无监督阶段对比
生成:
  1. 训练曲线（RE + Loss）带阶段标注
  2. 当前模型（无监督 epoch 2）重建图像
  3. 诊断图：预测对比度 vs 真实对比度
  4. 失败模式分析
"""
import os, sys, json, numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.conv_spatial_eit import ConvSpatialEIT
from data.eit_forward import EITForwardSolver
from data.datasets.eit_dataset import MemoryEITDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

OUTPUT = "results/eval_new_data_viz"
os.makedirs(OUTPUT, exist_ok=True)

# ========== 1. 训练曲线 ==========
print("\n=== 1. 训练曲线 ===")
run_id = "20260624_225603_v2_both_hd256"
with open(f"training_records/{run_id}/epochs.jsonl") as f:
    lines = [json.loads(l) for l in f]

sup_epochs = [d['epoch'] for d in lines if d['phase'] == 'supervised']
sup_re = [d.get('re', 0) for d in lines if d['phase'] == 'supervised']
sup_loss = [d.get('loss', 0) for d in lines if d['phase'] == 'supervised']
unsup_epochs = [d['epoch'] + max(sup_epochs) for d in lines if d['phase'] == 'unsupervised']
unsup_re = [d.get('val_re', 0) for d in lines if d['phase'] == 'unsupervised']
unsup_loss = [d.get('loss', 0) for d in lines if d['phase'] == 'unsupervised']

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
fig.patch.set_facecolor('white')

# RE 曲线
ax = axes[0]
ax.plot(sup_epochs, sup_re, 'b-o', markersize=4, linewidth=1.8, label='Supervised Val RE', color='#2196F3')
# 标注有监督最佳点
best_sup_idx = np.argmin(sup_re)
ax.plot(sup_epochs[best_sup_idx], sup_re[best_sup_idx], 'r*', markersize=15,
        label=f'Best Sup RE={sup_re[best_sup_idx]:.4f}')
ax.axvline(x=max(sup_epochs), color='gray', linestyle='--', alpha=0.5, label='Phase Boundary')
ax.plot(unsup_epochs, unsup_re, 'r-s', markersize=5, linewidth=1.8, label='Unsupervised Val RE', color='#E91E63')
# 标注无监督最佳点（实际是退化）
ax.text(unsup_epochs[0]+0.5, unsup_re[0], f'Unsup best\nRE={unsup_re[0]:.4f}\n(退化!)',
        fontsize=9, color='#E91E63', fontweight='bold')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('RE', fontsize=12)
ax.set_title('Validation RE: 有监督 → 无监督', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
ax.set_yscale('log')
ax.set_ylim(0.1, 1.0)
# 标记 trivial baseline
ax.axhline(y=0.75, color='gray', linestyle=':', alpha=0.5)
ax.text(max(sup_epochs)*0.7, 0.76, 'Trivial Baseline\n(全背景预测)', fontsize=8, color='gray', alpha=0.7)

# 在RE曲线旁边加 tiny 说明框
props = dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#FF9800', alpha=0.8)
ax2 = axes[0].inset_axes([0.55, 0.15, 0.40, 0.20])
ax2.axis('off')
ax2.text(0.5, 0.5,
    f'Sup 最佳: RE={sup_re[best_sup_idx]:.4f} (epoch {sup_epochs[best_sup_idx]})\n'
    f'Unsup 最佳: RE={unsup_re[0]:.4f} (epoch 1)\n'
    f'⚠️ Unsup 使 RE 恶化 67%',
    fontsize=9, ha='center', va='center', fontweight='bold',
    transform=ax2.transAxes)

# Loss 曲线
ax = axes[1]
ax.plot(sup_epochs, sup_loss, 'b-o', markersize=4, linewidth=1.8, label='Supervised Loss', color='#2196F3')
ax.axvline(x=max(sup_epochs), color='gray', linestyle='--', alpha=0.5, label='Phase Boundary')
ax.plot(unsup_epochs, unsup_loss, 'r-s', markersize=5, linewidth=1.8, label='Unsupervised Loss', color='#E91E63')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(f"{OUTPUT}/training_curves.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  → {OUTPUT}/training_curves.png")

# ========== 2. 加载模型并推理 ==========
print("\n=== 2. 加载模型 ===")
solver = EITForwardSolver('config/mesh_config.yaml')
centers = solver.element_centers[:, :2].copy()
n_elems = solver.n_elems
mesh_nodes = solver.mesh.node[:, :2].copy()
mesh_elements = solver.mesh.element.copy()

jac_path = 'data/generated/jacobian.npy'
model_jacobian = np.load(jac_path)
if model_jacobian.ndim == 3:
    model_jacobian = model_jacobian[0]

model = ConvSpatialEIT(n_frequencies=6, n_meas=208, n_elems=n_elems,
                       hidden_dim=256, gnn_hidden=256, gnn_layers=4, use_gat=True, n_heads=4)
model.setup_mesh(centers, solver.mesh.element, jacobian=model_jacobian)

ckpt = torch.load(f'checkpoints/{run_id}/best.pt', map_location='cpu', weights_only=False)
from torch.optim.swa_utils import AveragedModel
ema_model = AveragedModel(model)
ema_model.load_state_dict(ckpt['ema_model'])
model.load_state_dict(ema_model.module.state_dict())
model = model.to(device).eval()
print(f"  模型: {run_id}/best.pt (unsupervised epoch 2, val_re={ckpt.get('val_re', '?'):.4f})")

# ========== 3. 推理 val 集 ==========
print("\n=== 3. 推理 ===")
ds = MemoryEITDataset('data/generated/mixed_dataset.h5', split='val', voltage_mask_ratio=0.0)
loader = DataLoader(ds, batch_size=16, shuffle=False)

all_pred, all_gt, all_v = [], [], []
with torch.no_grad():
    for batch in loader:
        V = batch['voltages'].to(device).view(-1, 6, 13, 16)
        out = model(V)
        all_pred.append(out['sigma'].cpu())
        all_gt.append(batch['sigmas'])
        all_v.append(batch['voltages'])
all_pred = torch.cat(all_pred)
all_gt = torch.cat(all_gt)
re_vals = torch.norm(all_pred - all_gt, dim=-1) / (torch.norm(all_gt, dim=-1) + 1e-8)

# ========== 4. 诊断图：预测 vs 真实对比度 ==========
print("\n=== 4. 对比度诊断 ===")
# 每个样本分析
n_samples = min(500, len(all_gt))
gt_max = all_gt[:n_samples].max(dim=1)[0]
pred_max = all_pred[:n_samples].max(dim=1)[0]
gt_contrast = gt_max / 0.01  # 对比度倍数 (相对背景)
pred_contrast = pred_max / 0.01

# 根区平均
root_thresh = 0.015
gt_root_mean = torch.zeros(n_samples)
pred_root_mean = torch.zeros(n_samples)
for i in range(n_samples):
    mask = all_gt[i] > root_thresh
    if mask.any():
        gt_root_mean[i] = all_gt[i][mask].mean()
        pred_root_mean[i] = all_pred[i][mask].mean()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor('white')

# ── 左上: 单个样本 RE 分布 ──
ax = axes[0, 0]
ax.hist(re_vals[:n_samples].numpy(), bins=30, color='#E91E63', alpha=0.7, edgecolor='white')
ax.axvline(re_vals[:n_samples].mean(), color='#C62828', linestyle='--', linewidth=2,
           label=f'Mean RE={re_vals[:n_samples].mean():.4f}')
ax.set_xlabel('RE', fontsize=11)
ax.set_ylabel('样本数', fontsize=11)
ax.set_title('单样本 RE 分布', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# ── 右上: 预测最大值 vs 真实最大值 ──
ax = axes[0, 1]
ax.scatter(gt_max.numpy(), pred_max.numpy(), alpha=0.3, s=10, c='#2196F3')
ax.plot([0, 0.1], [0, 0.1], 'k--', alpha=0.5, label='Ideal (y=x)')
ax.plot([0, 0.1], [0, 0.025], 'r--', alpha=0.5, label='Current (~25% of true)')
ax.set_xlabel('True σ_max', fontsize=11)
ax.set_ylabel('Pred σ_max', fontsize=11)
ax.set_title('峰值对比度: Pred vs True', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0.01, 0.11)
ax.set_ylim(0, 0.08)

# ── 左下: 根区平均预测 vs 真实 ──
ax = axes[1, 0]
valid = gt_root_mean > 0
ax.scatter(gt_root_mean[valid].numpy(), pred_root_mean[valid].numpy(), alpha=0.3, s=10, c='#FF9800')
ax.plot([0.01, 0.1], [0.01, 0.1], 'k--', alpha=0.5, label='Ideal')
# 拟合线
from numpy.polynomial import polynomial as P
mask_valid = valid.numpy()
coefs = P.polyfit(gt_root_mean[valid].numpy(), pred_root_mean[valid].numpy(), 1)
x_fit = np.linspace(0.01, 0.1, 100)
ax.plot(x_fit, P.polyval(x_fit, coefs), 'r-', linewidth=2,
        label=f'Fit: y={coefs[1]:.3f}x+{coefs[0]:.4f}')
ax.set_xlabel('True Root Mean σ', fontsize=11)
ax.set_ylabel('Pred Root Mean σ', fontsize=11)
ax.set_title(f'根区平均 σ: Pred vs True', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── 右下: 对比度压缩比 vs 真实对比度 ──
ax = axes[1, 1]
contrast_ratio = pred_max / (gt_max + 1e-8)
ax.scatter(gt_contrast.numpy(), contrast_ratio.numpy(), alpha=0.3, s=10, c='#4CAF50')
ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal (100%)')
ax.axhline(y=0.39, color='r', linestyle='--', alpha=0.5, label='Avg=39%')
ax.set_xlabel('True Contrast (× background)', fontsize=11)
ax.set_ylabel('Pred/True Ratio', fontsize=11)
ax.set_title('对比度压缩比 vs 真实对比度', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/contrast_diagnosis.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  → {OUTPUT}/contrast_diagnosis.png")

# ========== 5. 重建图像对比（选最好/最差/中等样本）==========
print("\n=== 5. 重建图像 ===")

# 选展示样本
n_show = 9
sorted_idx = torch.argsort(re_vals)
# 选最好3个、中等3个、最差3个
show_idx = list(sorted_idx[:3]) + list(sorted_idx[len(sorted_idx)//2 - 1: len(sorted_idx)//2 + 2]) + list(sorted_idx[-3:])
titles = ['Best', '', '', 'Medium', '', '', 'Worst', '', '']

fig, axes = plt.subplots(n_show, 3, figsize=(15, 3.6 * n_show))
fig.patch.set_facecolor('white')

# 顶行标题
fig.suptitle(f'ConvSpatialEIT 重建对比\n'
             f'模型: Unsupervised Epoch 2 (RE={re_vals.mean():.4f}) | 数据: mixed_dataset val',
             fontsize=14, fontweight='bold', y=0.995)

for row, idx in enumerate(show_idx):
    gt = all_gt[idx].numpy()
    pred = all_pred[idx].numpy()
    err = np.abs(pred - gt)
    re_i = re_vals[idx].item()
    # 根区误差
    root_mask = gt > 0.015
    root_err = np.abs(pred - gt)[root_mask].mean() if root_mask.any() else 0

    for col, (data, cmap, vmin, vmax, label) in enumerate([
        (gt, 'viridis', 0.008, 0.055, 'Ground Truth'),
        (pred, 'viridis', 0.008, 0.055, 'Prediction'),
        (err, 'hot', 0, 0.015, '|Error|'),
    ]):
        ax = axes[row, col]
        ax.tripcolor(mesh_nodes[:, 0], mesh_nodes[:, 1],
                     mesh_elements, facecolors=data,
                     cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax.set_aspect('equal')
        ax.axis('off')

        # 行首标注
        if col == 0:
            ax.set_ylabel(f'Sample {idx}\nGT root={gt[gt>0.015].mean():.3f}\nPred root={pred[gt>0.015].mean():.3f}',
                         fontsize=9, fontweight='bold', color='#333')
            ax.set_title(f'Ground Truth\nRE={re_i:.4f}', fontsize=10, fontweight='bold', pad=2)
        elif col == 1:
            ax.set_title(f'Prediction\nContrast={pred.max()/0.01:.1f}x', fontsize=10, fontweight='bold', pad=2)
        else:
            ax.set_title(f'Error Map\nRoot MAE={root_err:.4f}', fontsize=10, fontweight='bold', pad=2)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/reconstruction_comparison.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  → {OUTPUT}/reconstruction_comparison.png")

# ========== 6. 相同样本的 有监督阶段 vs 无监督阶段 模拟对比 ==========
print("\n=== 6. 阶段对比（训练曲线证据）===")
# 由于有监督阶段 ckpt 已被覆盖，我们用训练曲线数据生成对比说明图

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('white')

# 绘制 RE vs Epoch
ax.fill_between(sup_epochs, 0, 1, alpha=0.08, color='#2196F3', label='Supervised Phase')
ax.fill_between(unsup_epochs, 0, 1, alpha=0.08, color='#E91E63', label='Unsupervised Phase')

ax.plot(sup_epochs, sup_re, 'b-o', markersize=5, linewidth=2, color='#1565C0', label='Supervised RE')
ax.plot(unsup_epochs, unsup_re, 'r-s', markersize=6, linewidth=2, color='#C62828', label='Unsupervised RE')

# 标注关键点
ax.plot(sup_epochs[best_sup_idx], sup_re[best_sup_idx], 'g*', markersize=20,
        label=f'Best Supervised: RE={sup_re[best_sup_idx]:.4f}')
ax.annotate(f'有监督最佳\nRE={sup_re[best_sup_idx]:.4f}\n(epoch {sup_epochs[best_sup_idx]})',
           xy=(sup_epochs[best_sup_idx], sup_re[best_sup_idx]),
           xytext=(sup_epochs[best_sup_idx]-8, sup_re[best_sup_idx]-0.25),
           fontsize=10, fontweight='bold', color='#1565C0',
           arrowprops=dict(arrowstyle='->', color='#1565C0', linewidth=1.5))

ax.annotate(f'⚠️ 无监督覆盖\nbest.pt 后\nRE 跳回 0.613',
           xy=(unsup_epochs[0], unsup_re[0]),
           xytext=(unsup_epochs[0]+0.5, 0.5),
           fontsize=10, fontweight='bold', color='#C62828',
           arrowprops=dict(arrowstyle='->', color='#C62828', linewidth=1.5))

ax.annotate(f'⚠️ Checkpoint\n被无监督 epoch 2\n覆盖 (RE=0.575)',
           xy=(unsup_epochs[1], unsup_re[1]),
           xytext=(unsup_epochs[1]+0.5, 0.35),
           fontsize=10, fontweight='bold', color='#E65100',
           arrowprops=dict(arrowstyle='->', color='#E65100', linewidth=1.5))

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation RE', fontsize=12)
ax.set_title('有监督 vs 无监督阶段对比（训练曲线 + 关键事件标注）', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)
ax.set_yscale('log')
ax.set_ylim(0.15, 1.0)

# 右侧加图示框
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax.text(0.98, 0.15,
    f'📊 结论:\n'
    f'• 有监督 50 epoch: RE 0.40→0.34\n'
    f'• 无监督 2 epoch:  RE 0.34→0.61\n'
    f'• best.pt 被无监督覆盖无法恢复\n'
    f'• 预测对比度仅还原 39%\n'
    f'• 建议: 修复ckpt保存 + 增大模型',
    fontsize=9, ha='right', va='bottom', transform=ax.transAxes,
    bbox=props)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/phase_comparison.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  → {OUTPUT}/phase_comparison.png")

# ========== 7. 失败模式总结图 ==========
print("\n=== 7. 失败模式总结 ===")
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
ax.text(0.5, 0.5,
    f'❌ 模型效果差的原因总结\n\n'
    f'1. 预测对比度严重不足（仅恢复 39% 的真实对比度）\n'
    f'   - 背景 (gt=0.01):  Pred={all_pred[all_gt<0.015].mean():.4f} ✅ 接近\n'
    f'   - 根区 (gt=0.065): Pred={all_pred[all_gt>0.015].mean():.4f} ❌ 仅 39%\n'
    f'   - 高对比 (gt=0.09): Pred={all_pred[all_gt>0.08].mean():.4f} ❌ 仅 34%\n\n'
    f'2. MSE 损失被 95% 背景元素主导\n'
    f'   - 背景/根比例 = {all_gt[all_gt<0.015].numel()/all_gt.numel()*100:.0f}:{all_gt[all_gt>=0.015].numel()/all_gt.numel()*100:.0f}\n'
    f'   - 模型偏向预测背景值，根区细节丢失\n\n'
    f'3. GNN 平滑效应 + Sigmoid 输出压缩\n'
    f'   - GATv2 邻居聚合模糊边缘\n'
    f'   - Sigmoid 将输出限制在 [0.005, 0.065]\n\n'
    f'4. Jacobian 条件数 2e13 → 无监督反向优化\n\n'
    f'5. best.pt 被覆盖（有监督最佳 RE=0.344 → best.pt 实际保存 RE=0.575）',
    fontsize=11, ha='center', va='center', linespacing=1.5,
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2))
plt.savefig(f"{OUTPUT}/failure_summary.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  → {OUTPUT}/failure_summary.png")

print(f"\n✅ 所有图像已生成到 {OUTPUT}/")
print(f"  文件列表:")
for f in sorted(os.listdir(OUTPUT)):
    sz = os.path.getsize(f"{OUTPUT}/{f}") / 1024
    print(f"    {f}: {sz:.0f}KB")
