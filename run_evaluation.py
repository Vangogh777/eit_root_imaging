#!/usr/bin/env python
"""简化的评估脚本"""
import os
import sys
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.conv_spatial_eit import ConvSpatialEIT
from data.eit_forward import EITForwardSolver

print("开始评估...")

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 加载模型
checkpoint_path = "checkpoints/20260622_231207_v2_unsupervised_hd256/final.pt"
print(f"加载checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

# 获取配置
n_elems = checkpoint.get('n_elems', 1500)
hidden_dim = checkpoint.get('hidden_dim', 256)
print(f"配置: n_elems={n_elems}, hidden_dim={hidden_dim}")

# 先在CPU上创建模型
model = ConvSpatialEIT(n_frequencies=6, n_meas=208, n_elems=n_elems, hidden_dim=hidden_dim)

# 加载网格（需要在加载权重之前）
solver = EITForwardSolver("config/mesh_config.yaml")
centers = solver.element_centers[:, :2]
elements = solver.mesh.element

jacobian_path = "data/generated/jacobian.npy"
jacobian = np.load(jacobian_path) if os.path.exists(jacobian_path) else None
model.setup_mesh(centers, elements, jacobian)
print("已设置网格")

# 加载权重
if 'ema_model' in checkpoint:
    state_dict = checkpoint['ema_model']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k != 'n_averaged'}
    
    # 检查缺失和意外的键
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"已加载EMA模型权重")
    if missing:
        print(f"缺失的键 ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"意外的键 ({len(unexpected)}): {unexpected[:5]}...")

# 然后移到GPU
model = model.to(device)
model.eval()

# 测试推理
print("\n测试推理...")
dummy_input = torch.randn(1, 6, 208).to(device)
with torch.no_grad():
    output = model(dummy_input)
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.6f}, {output.max().item():.6f}]")

print("\n✓ 模型加载和推理成功！")
