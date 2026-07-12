"""
诊断：为什么RE不变？
"""

import torch
import numpy as np
import h5py

print("=" * 60)
print("诊断：为什么RE不变？")
print("=" * 60)

# 1. 检查验证集大小
with h5py.File('data/generated/mixed_dataset.h5', 'r') as f:
    n_train = f['train/voltages'].shape[0]
    n_val = f['val/voltages'].shape[0]
    print(f"\n数据集大小:")
    print(f"  训练集: {n_train} 样本")
    print(f"  验证集: {n_val} 样本")

    # 检查验证集batch数
    val_batch_size = 32  # 从配置
    n_val_batches = n_val // val_batch_size + (1 if n_val % val_batch_size else 0)
    print(f"  验证集batch数 (batch_size={val_batch_size}): {n_val_batches}")

# 2. 检查sigma_0和g的多样性
with h5py.File('data/generated/mixed_dataset.h5', 'r') as f:
    sigma_0 = f['val/sigma_0'][:]
    g = f['val/physics_g'][:]
    target = f['val/sigmas'][:]

    print(f"\n输入特征多样性:")

    # sigma_0在不同样本之间的差异
    sigma_0_mean = sigma_0.mean(axis=1)  # 每个样本的均值
    print(f"  sigma_0 均值范围: [{sigma_0_mean.min():.4f}, {sigma_0_mean.max():.4f}]")
    print(f"  sigma_0 均值std: {sigma_0_mean.std():.4f}")

    # g在不同样本之间的差异
    g_mean = g.mean(axis=1)
    print(f"  g 均值范围: [{g_mean.min():.4f}, {g_mean.max():.4f}]")
    print(f"  g 均值std: {g_mean.std():.4f}")

    # 检查是否所有样本的sigma_0几乎相同
    sigma_0_sample_var = sigma_0.std(axis=1)  # 每个样本内部的标准差
    print(f"  sigma_0 样本内std范围: [{sigma_0_sample_var.min():.4f}, {sigma_0_sample_var.max():.4f}]")

# 3. 检查target的多样性
print(f"\n目标(target)多样性:")
target_mean = target.mean(axis=1)
print(f"  target 均值范围: [{target_mean.min():.4f}, {target_mean.max():.4f}]")
print(f"  target 均值std: {target_mean.std():.4f}")

# 4. 关键问题：sigma_0和target的差异（需要校正的量）
correction_needed = target - sigma_0
print(f"\n需要校正量:")
print(f"  平均校正量: {correction_needed.abs().mean():.4f}")
print(f"  不同样本校正量差异: {correction_needed.mean(axis=1).std():.4f}")

# 5. 检查验证集是否太小导致统计不稳定
print(f"\n验证集统计稳定性:")
print(f"  如果batch数<5，验证指标可能不稳定")
if n_val_batches < 5:
    print(f"  ⚠ 验证集batch数={n_val_batches} 太少!")
else:
    print(f"  ✓ 验证集batch数={n_val_batches} 足够")

# 6. 检查模型输出是否对不同输入产生不同结果
print(f"\n模型输出多样性测试:")
from models.residual_eit import ResidualEIT
from data.datasets.eit_dataset import EITDataModule
import yaml

dm = EITDataModule(
    h5_path="data/generated/mixed_dataset.h5",
    batch_size=4,
    load_residual_features=True,
    voltage_mask_ratio=0.0,
)

with open("config/residual_eit_config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

ds = dm.val_dataset
centers = np.mean(ds.mesh_nodes[ds.mesh_elements], axis=1)
elements = ds.mesh_elements

J = np.load("data/generated/jacobian.npy").astype(np.float32)
if J.ndim == 3:
    J = J[0]

model = ResidualEIT(
    n_frequencies=ds.n_freq,
    n_meas=ds.n_meas,
    n_elems=ds.n_elems,
    hidden_dim=cfg['model']['hidden_dim'],
    delta_scale=cfg['model']['delta_scale'],
    jacobian=J,
)
model.setup_mesh(centers, elements)
model.eval()

# 测试两个不同的batch
loader = dm.val_dataloader()
batch1 = next(iter(loader))
batch2 = next(iter(loader))

with torch.no_grad():
    out1 = model(
        voltages=batch1["voltages"],
        sigma_0=batch1["sigma_0"],
        g=batch1["physics_g"],
        residual=batch1["voltage_residual"],
    )
    out2 = model(
        voltages=batch2["voltages"],
        sigma_0=batch2["sigma_0"],
        g=batch2["physics_g"],
        residual=batch2["voltage_residual"],
    )

sigma1 = out1["sigma"]
sigma2 = out2["sigma"]

print(f"  Batch1 sigma均值: {sigma1.mean():.4f}")
print(f"  Batch2 sigma均值: {sigma2.mean():.4f}")
print(f"  两个batch输出差异: {(sigma1.mean() - sigma2.mean()).abs():.4f}")

if (sigma1.mean() - sigma2.mean()).abs() < 0.001:
    print(f"  ⚠ 模型对不同输入产生几乎相同的输出!")
else:
    print(f"  ✓ 模型对不同输入产生不同输出")

# 检查delta_sigma是否变化
delta1 = out1["delta_sigma"]
delta2 = out2["delta_sigma"]
print(f"\n  delta_sigma1 std: {delta1.std():.4f}")
print(f"  delta_sigma2 std: {delta2.std():.4f}")

if delta1.std() < 0.001:
    print(f"  ⚠ delta_sigma几乎为零 - 模型没有做校正!")
else:
    print(f"  ✓ delta_sigma有变化")

print("\n" + "=" * 60)
print("诊断结论")
print("=" * 60)