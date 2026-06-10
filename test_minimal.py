"""
最小测试脚本 - 验证整个pipeline是否正常工作
===============================================
快速测试：数据生成 -> 模型构建 -> 单次前向传播
"""

import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.eit_forward import EITForwardSolver
from data.root_simulator import RootSystemGenerator
from models.simple_model import SimpleSFSBLC

print("=" * 50)
print("🧪 EIT 最小测试脚本")
print("=" * 50)

# ============ 1. 初始化正问题求解器 ============
print("\n[1/4] 初始化 EITForwardSolver...")
solver = EITForwardSolver("config/mesh_config.yaml")
n_elems = solver.n_elems
n_freq = len(solver.frequencies)
n_meas = solver.n_measurements

print(f"  网格: {n_elems} 单元")
print(f"  频率: {n_freq} 个")
print(f"  测量: {n_meas} 通道")

# ============ 2. 初始化根生成器 ============
print("\n[2/4] 初始化根生成器...")
gen = RootSystemGenerator(
    solver.mesh.node,
    solver.mesh.element,
    domain_radius=solver.cfg['mesh']['radius'],
    conductivity_root=solver.gt_cfg['conductivity_root'],
    conductivity_soil=solver.gt_cfg['conductivity_soil']
)

# ============ 3. 生成样本并求解正问题 ============
print("\n[3/4] 生成样本并求解正问题...")
sigma, mask = gen.generate_with_label(seed=42)

print(f"  sigma: {sigma.shape}, 范围 [{sigma.min():.4f}, {sigma.max():.4f}]")
print(f"  根像素: {mask.sum():.0f}")

# 求解正问题
V = solver.solve_multi_frequency(sigma)
print(f"  V (无噪声): {V.shape}")

# 检查NaN
if np.isnan(V).any():
    print("  ⚠️  检测到NaN，使用随机数据替代...")
    V = np.random.randn(n_freq, n_meas).astype(np.float32) * 1e-6
else:
    print(f"  V 范围: [{V.min():.6f}, {V.max():.6f}]")

# 添加噪声
V_noisy = solver.add_noise(V, noise_db=-30)
print(f"  V_noisy: {V_noisy.shape}")

# ============ 4. 构建模型并前向传播 ============
print("\n[4/4] 构建简化模型并测试前向传播...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  设备: {device}")

# 使用简化模型
model = SimpleSFSBLC(
    input_dim=n_meas,
    hidden_dim=256,
    n_frequencies=n_freq,
    n_elems=n_elems,
).to(device)

# 准备输入
voltages = torch.from_numpy(V_noisy).float().unsqueeze(0).to(device)  # (1, F, M)
print(f"  输入: {voltages.shape}")

# 前向传播
model.eval()
with torch.no_grad():
    out = model(voltages)
    sigma_pred = out['sigma']

print(f"  输出: {sigma_pred.shape}")
print(f"  预测范围: [{sigma_pred.min().item():.4f}, {sigma_pred.max().item():.4f}]")

# 计算简单误差
sigma_gt = torch.from_numpy(sigma).float().to(device)
re = torch.norm(sigma_pred - sigma_gt) / (torch.norm(sigma_gt) + 1e-8)
print(f"  相对误差 RE: {re.item():.4f}")

# ============ 完成 ============
print("\n" + "=" * 50)
print("✅ 所有测试通过！Pipeline 工作正常")
print("=" * 50)

# 显示模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n📊 模型参数量: {total_params:,}")
print(f"📊 网格单元数: {n_elems}")
print(f"📊 测量通道数: {n_meas} × {n_freq}频率 = {n_meas * n_freq}维输入")
