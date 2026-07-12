#!/usr/bin/env python3
"""
生成根系数据集的预览图
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os

def generate_root_preview():
    # 加载数据集
    data_path = '/home/ubuntu/EIT/eit_root_imaging/data/generated/eit_dataset.h5'
    output_dir = '/home/ubuntu/EIT/eit_root_imaging/results/dataset_preview'
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(data_path, 'r') as f:
        # 获取网格信息
        mesh_nodes = f['metadata/mesh_nodes'][:]
        mesh_elements = f['metadata/mesh_elements'][:]

        # 获取训练数据
        sigmas = f['train/sigmas'][:]
        voltages = f['train/voltages'][:]
        masks = f['train/masks'][:]

        print(f"Dataset: {sigmas.shape[0]} samples")
        print(f"Mesh: {mesh_nodes.shape[0]} nodes, {mesh_elements.shape[0]} elements")

        # 选择有代表性的样本（有根系的样本）
        sample_indices = []
        for i in range(min(100, len(sigmas))):
            # 计算电导率差异，选择有明显根系的样本
            sigma = sigmas[i]
            sigma_range = sigma.max() - sigma.min()
            if sigma_range > 0.02:  # 有明显变化
                sample_indices.append(i)
            if len(sample_indices) >= 12:
                break

        if len(sample_indices) < 12:
            sample_indices = list(range(min(12, len(sigmas))))

        # 创建大图显示多个样本
        n_samples = min(12, len(sample_indices))
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        # 准备三角剖分
        x = mesh_nodes[:, 0]
        y = mesh_nodes[:, 1]
        triangles = mesh_elements

        for idx, ax in enumerate(axes):
            if idx >= n_samples:
                ax.axis('off')
                continue

            sample_idx = sample_indices[idx]
            sigma = sigmas[sample_idx]

            # 创建三角剖分
            triang = Triangulation(x, y, triangles)

            # 绘制电导率分布 (使用facecolors，因为sigma定义在元素上)
            im = ax.tripcolor(triang, facecolors=sigma, edgecolors='none', cmap='viridis')
            ax.set_aspect('equal')
            ax.set_title(f'Sample {sample_idx}', fontsize=10)
            ax.axis('off')

            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'Root Dataset Preview (Train set, {sigmas.shape[0]} samples)\n'
                    f'Mesh: {mesh_nodes.shape[0]} nodes, {mesh_elements.shape[0]} elements',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存预览图
        output_path = os.path.join(output_dir, 'root_samples.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close()

        # 生成详细视图（单个样本）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        sample_idx = sample_indices[0]
        sigma = sigmas[sample_idx]
        voltage = voltages[sample_idx]
        mask = masks[sample_idx]

        # 电导率分布
        triang = Triangulation(x, y, triangles)
        im1 = axes[0].tripcolor(triang, facecolors=sigma, edgecolors='none', cmap='viridis')
        axes[0].set_aspect('equal')
        axes[0].set_title(f'Conductivity (σ)\nRange: [{sigma.min():.4f}, {sigma.max():.4f}]')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # 边界电压
        voltage_2d = voltage[0].reshape(-1, 1) if voltage.ndim == 1 else voltage.T
        im2 = axes[1].imshow(voltage_2d, aspect='auto', cmap='RdBu')
        axes[1].set_title(f'Boundary Voltage\nShape: {voltage.shape}')
        axes[1].set_xlabel('Frequency')
        axes[1].set_ylabel('Measurement')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # 根系掩码
        im3 = axes[2].tripcolor(triang, facecolors=mask, edgecolors='none', cmap='gray')
        axes[2].set_aspect('equal')
        axes[2].set_title(f'Root Mask\nCoverage: {mask.mean()*100:.1f}%')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

        plt.suptitle(f'Root Sample Detail (Index {sample_idx})', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path_detail = os.path.join(output_dir, 'root_detail.png')
        plt.savefig(output_path_detail, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path_detail}")
        plt.close()

        # 生成数据集统计信息
        print("\n=== Dataset Statistics ===")
        print(f"Train: {sigmas.shape[0]} samples")
        print(f"Val: {f['val/sigmas'].shape[0]} samples")
        print(f"Test: {f['test/sigmas'].shape[0]} samples")
        print(f"Voltage shape: {voltages.shape}")
        print(f"Conductivity range: [{sigmas.min():.4f}, {sigmas.max():.4f}]")
        print(f"Root coverage: {masks.mean()*100:.1f}% (average)")

if __name__ == "__main__":
    generate_root_preview()
