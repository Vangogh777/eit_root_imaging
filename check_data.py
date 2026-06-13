"""
数据检查脚本
============
检查数据质量和正向求解器
"""

import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.eit_forward import EITForwardSolver
from models.universal_eit import UniversalPhantomGenerator

def main():
    print("=" * 50)
    print("正向求解器测试")
    print("=" * 50)

    # 测试正向求解器
    print("\n[1] 初始化求解器...")
    try:
        solver = EITForwardSolver("config/mesh_config.yaml")
        phantom_gen = UniversalPhantomGenerator(
            solver.mesh.node,
            solver.mesh.element,
            domain_radius=0.1,
            sigma_background=0.01,
            sigma_inclusion=0.05
        )
        print(f"    网格: {solver.n_elems} 单元")
    except Exception as e:
        print(f"    [ERROR] 初始化失败: {e}")
        return

    # 测试样本
    print("\n[2] 测试正向求解...")
    nan_count = 0
    valid_count = 0
    correlations = []

    for i in range(20):
        sigma = phantom_gen.generate_single_circle(seed=i)
        V = solver.solve_multi_frequency(sigma)

        if np.isnan(V).any():
            nan_count += 1
            print(f"    样本 {i}: NaN")
        else:
            valid_count += 1
            # 计算电压和电导率的简单相关性
            v_flat = V.flatten()
            s_flat = sigma
            corr = np.corrcoef(v_flat[:100], s_flat[:100])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
            print(f"    样本 {i}: OK, V range=[{V.min():.2e}, {V.max():.2e}]")

    print(f"\n[3] 结果:")
    print(f"    有效样本: {valid_count}/20")
    print(f"    NaN样本: {nan_count}/20")

    if valid_count > 0:
        print(f"    平均相关性: {np.mean(correlations):.4f}")

    if nan_count == 0:
        print("\n    ✅ 正向求解器工作正常!")
        print("    可以开始生成数据: python train_server.py --generate --wandb")
    elif valid_count > 0:
        print("\n    ⚠️ 部分样本失败，但仍可使用")
    else:
        print("\n    ❌ 正向求解器全部失败!")

    # 检查是否有缓存数据
    print("\n[4] 检查缓存数据...")
    cache_path = 'data/generated/server_data_20000_1000_2824.npz'
    if os.path.exists(cache_path):
        print(f"    发现缓存: {cache_path}")
        data = np.load(cache_path)
        v = data['train_voltages'].squeeze(1)
        s = data['train_sigmas']
        print(f"    电压形状: {v.shape}")
        print(f"    电导率形状: {s.shape}")
    else:
        print("    无缓存数据")
        print("    请运行: python train_server.py --generate --wandb")

    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
