"""
生成新的EIT数据集并创建代表图。

修改：
1. 网格单元数: ~6500 (之前11466)
2. 样本数: 1200 (之前21500)
3. 单频输入 (之前6频)
4. g特征保留原始值 (之前per-sample标准化)
"""

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.eit_forward import EITForwardSolver
from data.root_simulator import RootSystemGenerator


def visualize_sample(mesh_nodes, mesh_elements, sigma, title="EIT Sample", save_path=None):
    """可视化单个EIT样本的电导率分布。"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 计算单元中心
    centers = np.mean(mesh_nodes[mesh_elements], axis=1)

    # 使用tripcolor绘制
    x = mesh_nodes[:, 0]
    y = mesh_nodes[:, 1]
    triangles = mesh_elements

    # 绘制电导率分布
    im = ax.tripcolor(x, y, triangles, sigma, shading='flat', cmap='viridis')

    # 绘制电极位置
    n_electrodes = 16
    angles = np.linspace(0, 2*np.pi, n_electrodes, endpoint=False)
    radius = 0.098
    for i, angle in enumerate(angles):
        ex = radius * np.cos(angle)
        ey = radius * np.sin(angle)
        circle = Circle((ex, ey), 0.005, color='red', zorder=5)
        ax.add_patch(circle)
        ax.text(ex*1.08, ey*1.08, str(i+1), ha='center', va='center', fontsize=8)

    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.12, 0.12)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Conductivity (S/m)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {save_path}")

    plt.close()


def visualize_voltage(voltage, title="Boundary Voltage", save_path=None):
    """可视化边界电压。"""
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(voltage, 'b-', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel('Measurement Index', fontsize=12)
    ax.set_ylabel('Voltage (V)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {save_path}")

    plt.close()


def visualize_g_feature(g, title="g Feature (J^T r)", save_path=None):
    """可视化g特征的分布。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：g值分布直方图
    axes[0].hist(g, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('g value', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'{title}\nDistribution', fontsize=14)
    axes[0].axvline(g.mean(), color='red', linestyle='--', label=f'Mean={g.mean():.4f}')
    axes[0].legend()

    # 右图：g值随网格单元的变化
    axes[1].plot(g, 'b-', linewidth=0.5, alpha=0.7)
    axes[1].set_xlabel('Element Index', fontsize=12)
    axes[1].set_ylabel('g value', fontsize=12)
    axes[1].set_title(f'{title}\nSpatial Variation', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {save_path}")

    plt.close()


def main():
    print("=" * 70)
    print("生成新的EIT数据集并创建代表图")
    print("=" * 70)

    # 1. 检查是否需要删除旧数据
    h5_path = "data/generated/mixed_dataset.h5"
    if os.path.exists(h5_path):
        print(f"\n[1] 删除旧数据集: {h5_path}")
        os.remove(h5_path)

    # 2. 加载配置
    print("\n[2] 加载配置...")
    from data.generate_dataset import generate_dataset

    # 3. 生成新数据集
    print("\n[3] 生成新数据集...")
    print("  配置:")
    print("    - 网格分辨率: 0.004m (约6500单元)")
    print("    - 训练样本: 1000")
    print("    - 验证样本: 100")
    print("    - 测试样本: 100")
    print("    - 频率: 单频 1kHz")

    generate_dataset(
        config_path="config/mesh_config.yaml",
        n_train=1000,
        n_val=100,
        n_test=100,
        output_dir="data/generated",
        seed_start=42,
        num_workers=0  # 单进程避免多进程问题
    )

    # 4. 验证新数据集
    print("\n[4] 验证新数据集...")
    with h5py.File(h5_path, 'r') as f:
        train_v = f['train/voltages']
        train_s = f['train/sigmas']
        val_v = f['val/voltages']
        val_s = f['val/sigmas']

        print(f"  训练集: {train_v.shape}")
        print(f"  验证集: {val_v.shape}")
        print(f"  网格单元数: {train_s.shape[1]}")
        print(f"  频率数: {train_v.shape[1]}")

        # 获取网格信息
        meta = f['metadata']
        mesh_nodes = meta['mesh_nodes'][:]
        mesh_elements = meta['mesh_elements'][:]
        print(f"  网格节点数: {mesh_nodes.shape[0]}")
        print(f"  网格单元数: {mesh_elements.shape[0]}")

    # 5. 计算Jacobian
    print("\n[5] 计算Jacobian矩阵...")
    from data.precompute_jacobian import main as precompute_jacobian_main
    sys.argv = ['precompute_jacobian.py']
    try:
        precompute_jacobian_main()
    except SystemExit:
        pass

    # 6. 预计算残差特征
    print("\n[6] 预计算残差特征...")
    from data.precompute_residual_features import main as precompute_features_main
    sys.argv = [
        'precompute_residual_features.py',
        '--h5', h5_path,
        '--jacobian', 'data/generated/jacobian.npy',
        '--method', 'bp',
        '--force'
    ]
    try:
        precompute_features_main()
    except SystemExit:
        pass

    # 7. 创建代表图
    print("\n[7] 创建代表图...")
    output_dir = "docs/figures/dataset_samples"
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        meta = f['metadata']
        mesh_nodes = meta['mesh_nodes'][:]
        mesh_elements = meta['mesh_elements'][:]

        # 从验证集选几个样本可视化
        val_sigmas = f['val/sigmas'][:]
        val_voltages = f['val/voltages'][:]
        val_sigma_0 = f['val/sigma_0'][:]
        val_g = f['val/physics_g'][:]

        for i in range(min(5, len(val_sigmas))):
            print(f"\n  样本 {i+1}:")

            # 真实电导率分布
            visualize_sample(
                mesh_nodes, mesh_elements, val_sigmas[i],
                title=f"Ground Truth Conductivity (Sample {i+1})",
                save_path=f"{output_dir}/sample_{i+1}_ground_truth.png"
            )

            # 传统重建结果
            visualize_sample(
                mesh_nodes, mesh_elements, val_sigma_0[i],
                title=f"Traditional Reconstruction BP (Sample {i+1})",
                save_path=f"{output_dir}/sample_{i+1}_sigma0.png"
            )

            # 边界电压
            visualize_voltage(
                val_voltages[i, 0],  # 只用第一个频率
                title=f"Boundary Voltage (Sample {i+1})",
                save_path=f"{output_dir}/sample_{i+1}_voltage.png"
            )

            # g特征
            visualize_g_feature(
                val_g[i],
                title=f"g Feature J^T r (Sample {i+1})",
                save_path=f"{output_dir}/sample_{i+1}_g_feature.png"
            )

    # 8. 创建汇总对比图
    print("\n[8] 创建汇总对比图...")
    with h5py.File(h5_path, 'r') as f:
        mesh_nodes = f['metadata/mesh_nodes'][:]
        mesh_elements = f['metadata/mesh_elements'][:]
        val_sigmas = f['val/sigmas'][:]
        val_sigma_0 = f['val/sigma_0'][:]

        # 创建一个3x3的对比图
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for row in range(3):
            sample_idx = row

            # Ground Truth
            x = mesh_nodes[:, 0]
            y = mesh_nodes[:, 1]
            triangles = mesh_elements

            im0 = axes[row, 0].tripcolor(x, y, triangles, val_sigmas[sample_idx],
                                         shading='flat', cmap='viridis', vmin=0.005, vmax=0.1)
            axes[row, 0].set_title(f'Sample {sample_idx+1}: Ground Truth', fontsize=12)
            axes[row, 0].set_aspect('equal')
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

            # Traditional BP
            im1 = axes[row, 1].tripcolor(x, y, triangles, val_sigma_0[sample_idx],
                                         shading='flat', cmap='viridis', vmin=0.005, vmax=0.1)
            axes[row, 1].set_title(f'Sample {sample_idx+1}: BP Reconstruction', fontsize=12)
            axes[row, 1].set_aspect('equal')
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

            # 差异图
            diff = val_sigmas[sample_idx] - val_sigma_0[sample_idx]
            vmax_diff = max(abs(diff.min()), abs(diff.max()))
            im2 = axes[row, 2].tripcolor(x, y, triangles, diff,
                                         shading='flat', cmap='RdBu_r',
                                         vmin=-vmax_diff, vmax=vmax_diff)
            axes[row, 2].set_title(f'Sample {sample_idx+1}: Difference (GT-BP)', fontsize=12)
            axes[row, 2].set_aspect('equal')
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_grid.png", dpi=150, bbox_inches='tight')
        print(f"  保存汇总图到: {output_dir}/comparison_grid.png")
        plt.close()

    # 9. 检查g特征的样本间差异
    print("\n[9] 检查g特征样本间差异...")
    with h5py.File(h5_path, 'r') as f:
        g = f['val/physics_g'][:]

        g_means = g.mean(axis=1)
        g_stds = g.std(axis=1)

        print(f"  g 样本间均值差异: {g_means.std():.6f} (应该>0)")
        print(f"  g 样本内标准差范围: [{g_stds.min():.4f}, {g_stds.max():.4f}]")

        if g_means.std() > 0.1:
            print("  ✓ g特征有样本间差异，模型可以区分不同样本!")
        else:
            print("  ⚠ g特征样本间差异较小")

    print("\n" + "=" * 70)
    print("数据集生成和可视化完成!")
    print("=" * 70)
    print(f"\n代表图保存位置: {output_dir}/")
    print("\n下一步:")
    print("  1. 查看代表图: docs/figures/dataset_samples/")
    print("  2. 开始训练: python train_residual_eit.py")


if __name__ == "__main__":
    main()
