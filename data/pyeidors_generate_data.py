#!/usr/bin/env python3
"""
使用 PyEIDORS 生成 EIT 训练数据
===============================
运行前确保已安装 FEniCS:
  conda install -c conda-forge fenics
或使用 Docker:
  docker run -ti fenicsproject/stable:latest
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
import sys

# 检查环境
try:
    from pyeidors import EITSystem
    from pyeidors.data.structures import PatternConfig, EITImage
    from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
except ImportError as e:
    print(f"❌ PyEIDORS 导入失败: {e}")
    print("\n请先安装 FEniCS:")
    print("  conda install -c conda-forge fenics")
    print("  pip install -e /path/to/PyEIDORS")
    sys.exit(1)


class PyEIDORSDataGenerator:
    """PyEIDORS 数据生成器"""

    def __init__(
        self,
        n_elec: int = 16,
        radius: float = 0.1,  # 10cm 半径 = 20cm 直径
        refinement: int = 10,
        contact_impedance: float = 0.01,
        mesh_dir: str = "eit_meshes"
    ):
        self.n_elec = n_elec
        self.radius = radius
        self.refinement = refinement

        print("=" * 50)
        print("PyEIDORS 数据生成器初始化")
        print("=" * 50)

        # 创建网格
        print(f"\n[1] 创建网格 (refinement={refinement})...")
        self.mesh = load_or_create_mesh(
            mesh_dir=mesh_dir,
            n_elec=n_elec,
            radius=1.0,  # 归一化半径
            refinement=refinement,
            electrode_coverage=0.5
        )

        # 配置激励模式
        self.pattern_config = PatternConfig(
            n_elec=n_elec,
            stim_pattern='{ad}',  # 相邻激励
            meas_pattern='{ad}',  # 相邻测量
            amplitude=1.0
        )

        # 创建 EIT 系统
        print(f"\n[2] 初始化 EIT 系统...")
        self.eit_system = EITSystem(
            n_elec=n_elec,
            pattern_config=self.pattern_config,
            contact_impedance=np.ones(n_elec) * contact_impedance,
            base_conductivity=1.0
        )
        self.eit_system.setup(mesh=self.mesh)

        # 获取网格信息
        self.n_elems = len(self._get_homogeneous_sigma().vector()[:])
        self.n_meas = self.eit_system.fwd_model.pattern_manager.n_meas_total

        print(f"\n[3] 系统信息:")
        print(f"    电极数: {n_elec}")
        print(f"    网格单元数: {self.n_elems}")
        print(f"    测量数: {self.n_meas}")
        print(f"    激励模式: 相邻")

    def _get_homogeneous_sigma(self):
        """获取均匀电导率分布"""
        from fenics import Function, FunctionSpace
        V_sigma = self.eit_system.fwd_model.V_sigma
        sigma = Function(V_sigma)
        sigma.vector()[:] = 1.0
        return sigma

    def create_random_phantom(
        self,
        background: float = 1.0,
        n_anomalies: int = 1,
        conductivity_range: tuple = (0.5, 3.0),
        radius_range: tuple = (0.1, 0.3),
        seed: int = None
    ):
        """创建随机 phantom"""
        if seed is not None:
            np.random.seed(seed)

        anomalies = []
        for _ in range(n_anomalies):
            # 随机位置（确保在域内）
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.6)  # 距中心距离
            center = (r * np.cos(angle), r * np.sin(angle))

            anomalies.append({
                'center': center,
                'radius': np.random.uniform(*radius_range),
                'conductivity': np.random.uniform(*conductivity_range)
            })

        return self._create_phantom(background, anomalies)

    def _create_phantom(self, background: float, anomalies: list):
        """创建电导率分布"""
        from fenics import Function

        sigma = self._get_homogeneous_sigma()
        sigma.vector()[:] = background

        # 获取自由度坐标
        V_sigma = self.eit_system.fwd_model.V_sigma
        dof_coords = V_sigma.tabulate_dof_coordinates()

        # 设置异常区域电导率
        values = sigma.vector()[:]
        for anomaly in anomalies:
            cx, cy = anomaly['center']
            radius = anomaly['radius']
            cond = anomaly['conductivity']

            for i, coord in enumerate(dof_coords):
                dist = np.sqrt((coord[0] - cx)**2 + (coord[1] - cy)**2)
                if dist <= radius:
                    values[i] = cond

        sigma.vector()[:] = values
        return sigma

    def forward_solve(self, sigma):
        """前向求解"""
        img = EITImage(
            elem_data=sigma.vector()[:],
            fwd_model=self.eit_system.fwd_model
        )
        eit_data, _ = self.eit_system.fwd_model.fwd_solve(img)
        return eit_data.meas

    def generate_sample(
        self,
        background_range: tuple = (0.8, 1.2),
        n_anomalies_range: tuple = (1, 3),
        conductivity_range: tuple = (0.3, 3.0),
        radius_range: tuple = (0.1, 0.25),
        noise_level: float = 0.01,
        seed: int = None
    ):
        """生成单个样本"""
        if seed is not None:
            np.random.seed(seed)

        # 随机背景电导率
        background = np.random.uniform(*background_range)

        # 随机异常数量
        n_anomalies = np.random.randint(*n_anomalies_range)

        # 创建 phantom
        sigma = self.create_random_phantom(
            background=background,
            n_anomalies=n_anomalies,
            conductivity_range=conductivity_range,
            radius_range=radius_range
        )

        # 前向求解
        voltages = self.forward_solve(sigma)

        # 添加噪声
        if noise_level > 0:
            noise = noise_level * np.std(voltages) * np.random.randn(len(voltages))
            voltages = voltages + noise

        return {
            'sigma': sigma.vector()[:].copy(),
            'voltages': voltages.copy(),
            'background': background,
            'n_anomalies': n_anomalies
        }

    def generate_dataset(
        self,
        n_samples: int,
        output_path: str,
        background_range: tuple = (0.8, 1.2),
        n_anomalies_range: tuple = (1, 3),
        conductivity_range: tuple = (0.3, 3.0),
        radius_range: tuple = (0.1, 0.25),
        noise_level: float = 0.01,
        seed: int = 42
    ):
        """生成完整数据集"""
        print(f"\n[4] 生成数据集 ({n_samples} 样本)...")

        np.random.seed(seed)

        # 预分配数组
        all_sigmas = np.zeros((n_samples, self.n_elems), dtype=np.float32)
        all_voltages = np.zeros((n_samples, self.n_meas), dtype=np.float32)

        # 生成样本
        for i in range(n_samples):
            sample = self.generate_sample(
                background_range=background_range,
                n_anomalies_range=n_anomalies_range,
                conductivity_range=conductivity_range,
                radius_range=radius_range,
                noise_level=noise_level
            )
            all_sigmas[i] = sample['sigma']
            all_voltages[i] = sample['voltages']

            if (i + 1) % 100 == 0:
                print(f"    进度: {i+1}/{n_samples}")

        # 保存到 HDF5
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('conductivity', data=all_sigmas)
            f.create_dataset('voltages', data=all_voltages)
            f.attrs['n_elec'] = self.n_elec
            f.attrs['n_meas'] = self.n_meas
            f.attrs['n_elems'] = self.n_elems

        print(f"\n✅ 数据已保存: {output_path}")
        print(f"   电导率形状: {all_sigmas.shape}")
        print(f"   电压形状: {all_voltages.shape}")

        return all_sigmas, all_voltages


def main():
    parser = argparse.ArgumentParser(description='使用 PyEIDORS 生成 EIT 训练数据')
    parser.add_argument('--n_train', type=int, default=1000, help='训练样本数')
    parser.add_argument('--n_val', type=int, default=200, help='验证样本数')
    parser.add_argument('--n_test', type=int, default=100, help='测试样本数')
    parser.add_argument('--refinement', type=int, default=10, help='网格细化程度')
    parser.add_argument('--noise', type=float, default=0.01, help='噪声水平')
    parser.add_argument('--output', type=str, default='data/pyeidors_data.h5', help='输出路径')
    parser.add_argument('--mesh_dir', type=str, default='eit_meshes', help='网格存储目录')

    args = parser.parse_args()

    # 初始化生成器
    generator = PyEIDORSDataGenerator(
        n_elec=16,
        radius=0.1,
        refinement=args.refinement,
        mesh_dir=args.mesh_dir
    )

    # 生成训练集
    print("\n" + "=" * 50)
    print("生成训练集")
    print("=" * 50)
    train_sigmas, train_voltages = generator.generate_dataset(
        n_samples=args.n_train,
        output_path=args.output.replace('.h5', '_train.h5'),
        noise_level=args.noise,
        seed=42
    )

    # 生成验证集
    print("\n" + "=" * 50)
    print("生成验证集")
    print("=" * 50)
    val_sigmas, val_voltages = generator.generate_dataset(
        n_samples=args.n_val,
        output_path=args.output.replace('.h5', '_val.h5'),
        noise_level=args.noise,
        seed=123
    )

    # 生成测试集
    print("\n" + "=" * 50)
    print("生成测试集")
    print("=" * 50)
    test_sigmas, test_voltages = generator.generate_dataset(
        n_samples=args.n_test,
        output_path=args.output.replace('.h5', '_test.h5'),
        noise_level=args.noise,
        seed=456
    )

    print("\n" + "=" * 50)
    print("🎉 数据生成完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
