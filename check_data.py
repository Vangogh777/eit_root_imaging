"""
数据检查脚本
============
检查数据质量和线性回归基准
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.eit_forward import EITForwardSolver
from models.universal_eit import UniversalPhantomGenerator

def main():
    print("=" * 50)
    print("数据质量检查")
    print("=" * 50)

    # 加载数据
    data = np.load('data/generated/server_data_20000_1000_2824.npz')
    v = data['train_voltages'].squeeze(1)
    s = data['train_sigmas']

    # 基本统计
    print("\n[1] 数据范围:")
    print(f"    电压: [{v.min():.2e}, {v.max():.2e}], mean={v.mean():.2e}")
    print(f"    电导率: [{s.min():.4f}, {s.max():.4f}], mean={s.mean():.4f}")

    # 样本多样性
    print("\n[2] 样本多样性:")
    sample_means = s.mean(axis=1)
    print(f"    样本均值范围: [{sample_means.min():.4f}, {sample_means.max():.4f}]")
    print(f"    样本均值标准差: {sample_means.std():.4f}")

    if sample_means.std() < 0.001:
        print("    [WARN] 样本差异太小，可能数据生成有问题!")

    # 线性回归基准
    print("\n[3] 线性回归基准测试:")
    print("    训练样本: 1000, 测试样本: 1000")

    model = LinearRegression()
    model.fit(v[:1000], s[:1000])
    pred = model.predict(v[1000:2000])

    re = np.linalg.norm(pred - s[1000:2000], axis=1) / np.linalg.norm(s[1000:2000], axis=1)
    print(f"    线性回归 RE: {re.mean():.4f} ± {re.std():.4f}")

    if re.mean() > 0.5:
        print("    [WARN] 线性回归 RE > 0.5，说明问题本身较难")
    elif re.mean() > 0.3:
        print("    [INFO] 线性回归 RE ≈ 0.3，深度网络应该能降到 < 0.1")
    else:
        print("    [INFO] 线性回归效果不错，深度网络应该能进一步改进")

    # 检查电压和电导率的相关性
    print("\n[4] 电压-电导率相关性:")
    corr_matrix = np.corrcoef(v[:100, 0], s[:100, 0])[0, 1]
    print(f"    第一通道相关性: {corr_matrix:.4f}")

    if abs(corr_matrix) < 0.1:
        print("    [WARN] 相关性太低，可能正向求解有问题!")

    # 测试正向求解器
    print("\n[5] 测试正向求解器:")
    try:
        solver = EITForwardSolver("config/mesh_config.yaml")
        phantom_gen = UniversalPhantomGenerator(
            solver.mesh.node,
            solver.mesh.element,
            domain_radius=0.1,
            sigma_background=0.01,
            sigma_inclusion=0.05
        )

        # 测试几个样本
        nan_count = 0
        valid_count = 0
        for i in range(10):
            sigma = phantom_gen.generate_single_circle(seed=i)
            V = solver.solve_multi_frequency(sigma)
            if np.isnan(V).any():
                nan_count += 1
            else:
                valid_count += 1

        print(f"    测试10个样本: {valid_count} 有效, {nan_count} NaN")
        if nan_count > 0:
            print("    [WARN] 正向求解器返回NaN，数据可能无效!")

    except Exception as e:
        print(f"    [ERROR] 正向求解器测试失败: {e}")

    print("\n" + "=" * 50)
    print("检查完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
