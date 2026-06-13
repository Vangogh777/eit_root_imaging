"""
EIDORS数据转换器
将MATLAB/EIDORS生成的MAT文件转换为PyTorch可用的HDF5格式

使用方法:
    python convert_eidors_data.py --input ./data/eit_eidors_dataset.mat --output ./data/eidors_converted.h5
"""

import argparse
import h5py
import numpy as np
from scipy.io import loadmat
from pathlib import Path


def convert_mat_to_h5(mat_path: str, h5_path: str, add_noise: bool = False):
    """
    将EIDORS MAT文件转换为HDF5格式

    参数:
        mat_path: 输入MAT文件路径
        h5_path: 输出HDF5文件路径
        add_noise: 是否添加额外的噪声
    """
    print(f"加载MAT文件: {mat_path}")
    mat_data = loadmat(mat_path)

    # 提取数据
    voltages = mat_data['voltages']  # (n_samples, n_meas)
    conductivities = mat_data['conductivities']  # (n_samples, n_elements)
    nodes = mat_data['nodes']  # (n_nodes, 2)
    elements = mat_data['elements']  # (n_elements, 3)

    # 可选数据
    inclusion_masks = mat_data.get('inclusion_masks', None)
    voltages_noisy = mat_data.get('voltages_noisy', None)

    n_samples, n_meas = voltages.shape
    n_elements = conductivities.shape[1]
    n_nodes = nodes.shape[0]

    print(f"数据集信息:")
    print(f"  样本数: {n_samples}")
    print(f"  电压测量数: {n_meas}")
    print(f"  FEM单元数: {n_elements}")
    print(f"  FEM节点数: {n_nodes}")

    # 计算单元中心坐标
    elem_centers = np.zeros((n_elements, 2))
    for e in range(n_elements):
        elem_nodes = elements[e, :]
        elem_centers[e] = nodes[elem_nodes].mean(axis=0)

    # 数据归一化
    voltage_mean = voltages.mean()
    voltage_std = voltages.std()
    sigma_mean = conductivities.mean()
    sigma_std = conductivities.std()

    print(f"\n数据统计:")
    print(f"  电压: mean={voltage_mean:.6f}, std={voltage_std:.6f}")
    print(f"  电导率: mean={sigma_mean:.6f}, std={sigma_std:.6f}")

    # 划分数据集
    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)
    n_test = n_samples - n_train - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"\n数据集划分:")
    print(f"  训练集: {n_train}")
    print(f"  验证集: {n_val}")
    print(f"  测试集: {n_test}")

    # 保存为HDF5
    print(f"\n保存HDF5文件: {h5_path}")
    Path(h5_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'w') as f:
        # 存储元数据
        f.attrs['n_samples'] = n_samples
        f.attrs['n_meas'] = n_meas
        f.attrs['n_elements'] = n_elements
        f.attrs['n_nodes'] = n_nodes
        f.attrs['voltage_mean'] = voltage_mean
        f.attrs['voltage_std'] = voltage_std
        f.attrs['sigma_mean'] = sigma_mean
        f.attrs['sigma_std'] = sigma_std

        # 存储网格信息
        f.create_dataset('mesh/nodes', data=nodes.astype(np.float32))
        f.create_dataset('mesh/elements', data=elements.astype(np.int32))
        f.create_dataset('mesh/elem_centers', data=elem_centers.astype(np.float32))

        # 存储训练数据
        train_group = f.create_group('train')
        train_group.create_dataset('voltages', data=voltages[train_idx].astype(np.float32))
        train_group.create_dataset('conductivities', data=conductivities[train_idx].astype(np.float32))
        if inclusion_masks is not None:
            train_group.create_dataset('masks', data=inclusion_masks[train_idx].astype(np.float32))

        # 存储验证数据
        val_group = f.create_group('val')
        val_group.create_dataset('voltages', data=voltages[val_idx].astype(np.float32))
        val_group.create_dataset('conductivities', data=conductivities[val_idx].astype(np.float32))
        if inclusion_masks is not None:
            val_group.create_dataset('masks', data=inclusion_masks[val_idx].astype(np.float32))

        # 存储测试数据
        test_group = f.create_group('test')
        test_group.create_dataset('voltages', data=voltages[test_idx].astype(np.float32))
        test_group.create_dataset('conductivities', data=conductivities[test_idx].astype(np.float32))
        if inclusion_masks is not None:
            test_group.create_dataset('masks', data=inclusion_masks[test_idx].astype(np.float32))

        # 存储带噪声的电压数据（如果存在）
        if voltages_noisy is not None:
            f.create_dataset('voltages_noisy', data=voltages_noisy.astype(np.float32))

    print("转换完成！")
    return h5_path


def visualize_sample(h5_path: str, sample_idx: int = 0):
    """可视化一个样本"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    with h5py.File(h5_path, 'r') as f:
        nodes = f['mesh/nodes'][:]
        elements = f['mesh/elements'][:]
        voltage = f['train/voltages'][sample_idx]
        sigma = f['train/conductivities'][sample_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制电导率分布
    ax1 = axes[0]
    patches = []
    for elem in elements:
        triangle = Polygon(nodes[elem], closed=True)
        patches.append(triangle)

    p = PatchCollection(patches, alpha=1.0)
    p.set_array(sigma)
    p.set_cmap('jet')
    ax1.add_collection(p)
    ax1.set_xlim(nodes[:, 0].min() - 0.02, nodes[:, 0].max() + 0.02)
    ax1.set_ylim(nodes[:, 1].min() - 0.02, nodes[:, 1].max() + 0.02)
    ax1.set_aspect('equal')
    ax1.set_title(f'电导率分布 (样本 #{sample_idx})')
    plt.colorbar(p, ax=ax1, label='电导率 (S/m)')

    # 绘制电压测量
    ax2 = axes[1]
    ax2.plot(voltage, 'b-o', linewidth=1.5, markersize=3)
    ax2.set_xlabel('测量索引')
    ax2.set_ylabel('电压 (V)')
    ax2.set_title('边界电压测量')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sample_visualization.png', dpi=150)
    plt.show()
    print("可视化已保存到 sample_visualization.png")


def main():
    parser = argparse.ArgumentParser(description='转换EIDORS数据为HDF5格式')
    parser.add_argument('--input', type=str, required=True, help='输入MAT文件路径')
    parser.add_argument('--output', type=str, default='./data/eidors_converted.h5', help='输出HDF5文件路径')
    parser.add_argument('--visualize', action='store_true', help='可视化样本')
    args = parser.parse_args()

    convert_mat_to_h5(args.input, args.output)

    if args.visualize:
        visualize_sample(args.output)


if __name__ == '__main__':
    main()
