#!/usr/bin/env python3
"""
PyEIDORS 数据生成器
==================
替代 pyEIT 的前向求解器，使用 FEniCS 后端生成更稳定的训练数据。

使用方法:
    # 在安装了 FEniCS 的环境中运行
    python data/pyeidors_data_generator.py --n_train 20000 --n_val 1000

环境要求:
    conda install -c conda-forge fenics
    pip install -e /path/to/PyEIDORS
"""

import numpy as np
import h5py
import yaml
import argparse
from pathlib import Path
import sys
import time
from typing import Dict, Tuple, Optional

# 检查 PyEIDORS 环境
try:
    from pyeidors import EITSystem
    from pyeidors.data.structures import PatternConfig, EITImage
    from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
    PYEIDORS_AVAILABLE = True
except ImportError:
    PYEIDORS_AVAILABLE = False
    print("⚠️  PyEIDORS 未安装，请先安装:")
    print("    conda install -c conda-forge fenics")
    print("    pip install -e /path/to/PyEIDORS")


class PyEIDORSDataGenerator:
    """
    PyEIDORS 数据生成器

    生成与现有训练流程兼容的 HDF5 数据集格式。
    """

    def __init__(self, config_path: str = "config/mesh_config.yaml"):
        """初始化生成器"""
        if not PYEIDORS_AVAILABLE:
            raise RuntimeError("PyEIDORS 未安装")

        # 加载配置
        self.config = self._load_config(config_path)
        self._parse_config()

        print("=" * 60)
        print("PyEIDORS 数据生成器")
        print("=" * 60)

        # 初始化 EIT 系统
        self._setup_eit_system()

        # 打印系统信息
        self._print_info()

    def _load_config(self, config_path: str) -> dict:
        """加载 YAML 配置"""
        config_path = Path(config_path)
        if not config_path.exists():
            # 使用默认配置
            return self._default_config()

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _default_config(self) -> dict:
        """默认配置"""
        return {
            'mesh': {
                'type': 'circle',
                'radius': 0.1,
                'mesh_resolution': 0.005
            },
            'electrodes': {
                'count': 16
            },
            'stimulation': {
                'pattern': 'adjacent',
                'amplitude': 1e-3,
                'frequencies': [1000, 5000, 10000, 50000, 100000, 500000]
            },
            'ground_truth': {
                'conductivity_soil': 0.01,
                'conductivity_root': 0.05,
                'conductivity_air': 1e-8
            }
        }

    def _parse_config(self):
        """解析配置参数"""
        mesh_cfg = self.config.get('mesh', {})
        elec_cfg = self.config.get('electrodes', {})
        stim_cfg = self.config.get('stimulation', {})
        gt_cfg = self.config.get('ground_truth', {})

        # 网格参数
        self.domain_radius = mesh_cfg.get('radius', 0.1)
        self.mesh_resolution = mesh_cfg.get('mesh_resolution', 0.005)

        # 电极参数
        self.n_elec = elec_cfg.get('count', 16)

        # 激励参数
        self.frequencies = stim_cfg.get('frequencies', [10000])
        self.n_freq = len(self.frequencies)
        self.amplitude = stim_cfg.get('amplitude', 1e-3)

        # 电导率参数
        self.sigma_soil = gt_cfg.get('conductivity_soil', 0.01)
        self.sigma_root = gt_cfg.get('conductivity_root', 0.05)
        self.sigma_air = gt_cfg.get('conductivity_air', 1e-8)

    def _setup_eit_system(self):
        """设置 EIT 系统"""
        print(f"\n[1] 创建网格...")
        print(f"    域半径: {self.domain_radius} m")
        print(f"    网格分辨率: {self.mesh_resolution} m")

        # 计算细化程度 (从 mesh_resolution 估算)
        # refinement 越大，网格越细
        refinement = int(0.1 / self.mesh_resolution)
        refinement = max(6, min(refinement, 20))  # 限制在 6-20

        # 创建网格
        mesh_dir = Path("data/pyeidors_meshes")
        mesh_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.mesh = load_or_create_mesh(
                mesh_dir=str(mesh_dir),
                n_elec=self.n_elec,
                radius=1.0,  # 归一化半径
                refinement=refinement,
                electrode_coverage=0.5
            )
        except Exception as e:
            print(f"    网格加载失败: {e}")
            print("    尝试创建新网格...")
            # 尝试备用方法
            self.mesh = self._create_simple_mesh()

        # 配置激励模式
        self.pattern_config = PatternConfig(
            n_elec=self.n_elec,
            stim_pattern='{ad}',  # 相邻激励
            meas_pattern='{ad}',  # 相邻测量
            amplitude=1.0
        )

        # 创建 EIT 系统
        print(f"\n[2] 初始化 EIT 系统...")
        self.eit_system = EITSystem(
            n_elec=self.n_elec,
            pattern_config=self.pattern_config,
            contact_impedance=np.ones(self.n_elec) * 0.01,
            base_conductivity=self.sigma_soil
        )
        self.eit_system.setup(mesh=self.mesh)

        # 获取维度信息
        self._get_dimensions()

    def _create_simple_mesh(self):
        """创建简单网格（备用方法）"""
        from pyeidors.geometry.simple_mesh_generator import create_simple_eit_mesh
        return create_simple_eit_mesh(
            n_elec=self.n_elec,
            radius=1.0,
            mesh_size=self.mesh_resolution
        )

    def _get_dimensions(self):
        """获取网格维度"""
        # 获取电导率自由度数量
        from fenics import Function
        V_sigma = self.eit_system.fwd_model.V_sigma
        self.n_elems = len(Function(V_sigma).vector()[:])

        # 获取测量数
        self.n_meas = self.eit_system.fwd_model.pattern_manager.n_meas_total

        # 获取节点坐标（用于可视化）
        V = self.eit_system.fwd_model.V
        self.n_nodes = V.dim()

        # 获取 DOF 坐标
        self.dof_coords = V_sigma.tabulate_dof_coordinates()

    def _print_info(self):
        """打印系统信息"""
        print(f"\n[3] 系统信息:")
        print(f"    电极数: {self.n_elec}")
        print(f"    网格单元数: {self.n_elems}")
        print(f"    节点数: {self.n_nodes}")
        print(f"    测量数/频率: {self.n_meas}")
        print(f"    频率数: {self.n_freq}")
        print(f"    频率: {self.frequencies}")
        print(f"    电导率范围: [{self.sigma_soil}, {self.sigma_root}] S/m")

    def create_phantom(
        self,
        background: float = None,
        n_inclusions: int = 1,
        inclusion_type: str = 'random',
        seed: int = None
    ) -> np.ndarray:
        """
        创建 phantom 电导率分布

        Args:
            background: 背景电导率
            n_inclusions: 包含体数量
            inclusion_type: 包含体类型 ('random', 'circle', 'root')
            seed: 随机种子

        Returns:
            sigma: (n_elems,) 电导率分布
        """
        if seed is not None:
            np.random.seed(seed)

        if background is None:
            background = self.sigma_soil

        # 初始化为均匀背景
        sigma = np.ones(self.n_elems) * background

        if inclusion_type == 'random':
            # 随机圆形包含体
            for _ in range(n_inclusions):
                # 随机位置
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, 0.6)  # 距中心距离
                cx, cy = r * np.cos(angle), r * np.sin(angle)

                # 随机半径和电导率
                radius = np.random.uniform(0.1, 0.25)
                cond = np.random.uniform(self.sigma_soil * 0.5, self.sigma_root * 2)

                # 设置包含体
                for i, coord in enumerate(self.dof_coords):
                    dist = np.sqrt((coord[0] - cx)**2 + (coord[1] - cy)**2)
                    if dist <= radius:
                        sigma[i] = cond

        elif inclusion_type == 'root':
            # 模拟根结构（简化版）
            # 主根
            root_angle = np.random.uniform(0, 2 * np.pi)
            root_length = np.random.uniform(0.3, 0.6)
            root_width = np.random.uniform(0.05, 0.1)

            for i, coord in enumerate(self.dof_coords):
                # 沿径向的根
                x, y = coord[0], coord[1]
                # 旋转坐标
                x_rot = x * np.cos(root_angle) + y * np.sin(root_angle)
                y_rot = -x * np.sin(root_angle) + y * np.cos(root_angle)

                # 检查是否在根区域内
                if 0 < x_rot < root_length and abs(y_rot) < root_width:
                    sigma[i] = self.sigma_root

            # 侧根
            n_branches = np.random.randint(1, 4)
            for _ in range(n_branches):
                branch_pos = np.random.uniform(0.1, root_length - 0.1)
                branch_angle = np.random.choice([-1, 1]) * np.random.uniform(np.pi/6, np.pi/3)
                branch_length = np.random.uniform(0.1, 0.3)
                branch_width = np.random.uniform(0.03, 0.06)

                for i, coord in enumerate(self.dof_coords):
                    x, y = coord[0], coord[1]
                    x_rot = x * np.cos(root_angle) + y * np.sin(root_angle)
                    y_rot = -x * np.sin(root_angle) + y * np.cos(root_angle)

                    # 侧根坐标系
                    x_branch = (x_rot - branch_pos) * np.cos(branch_angle) + y_rot * np.sin(branch_angle)
                    y_branch = -(x_rot - branch_pos) * np.sin(branch_angle) + y_rot * np.cos(branch_angle)

                    if 0 < x_branch < branch_length and abs(y_branch) < branch_width:
                        sigma[i] = self.sigma_root

        return sigma

    def forward_solve(
        self,
        sigma: np.ndarray,
        noise_level: float = 0.0
    ) -> np.ndarray:
        """
        前向求解

        Args:
            sigma: (n_elems,) 电导率分布
            noise_level: 噪声水平

        Returns:
            voltage: (n_freq, n_meas) 边界电压
        """
        # 创建 EITImage
        img = EITImage(elem_data=sigma, fwd_model=self.eit_system.fwd_model)

        # 前向求解
        eit_data, _ = self.eit_system.fwd_model.fwd_solve(img)
        voltage = eit_data.meas.copy()

        # 添加噪声
        if noise_level > 0:
            noise = noise_level * np.std(voltage) * np.random.randn(len(voltage))
            voltage = voltage + noise

        # 扩展到多频率 (简化：假设频率响应相同)
        # 实际应用中可以根据频率调整电导率
        voltages = np.zeros((self.n_freq, len(voltage)))
        for i, freq in enumerate(self.frequencies):
            # 频率相关的缩放 (简化模型)
            freq_factor = 1.0 + 0.1 * np.log10(freq / 10000)
            voltages[i] = voltage * freq_factor

        return voltages

    def generate_sample(
        self,
        noise_level: float = 0.01,
        inclusion_type: str = 'random',
        seed: int = None
    ) -> Dict[str, np.ndarray]:
        """生成单个样本"""
        if seed is not None:
            np.random.seed(seed)

        # 随机背景电导率
        background = np.random.uniform(self.sigma_soil * 0.8, self.sigma_soil * 1.2)

        # 随机包含体数量
        n_inclusions = np.random.randint(1, 4)

        # 创建 phantom
        sigma = self.create_phantom(
            background=background,
            n_inclusions=n_inclusions,
            inclusion_type=inclusion_type
        )

        # 前向求解
        voltages = self.forward_solve(sigma, noise_level=noise_level)

        # 创建 mask (根区域标记)
        mask = (sigma > (self.sigma_soil * 1.5)).astype(np.float32)

        return {
            'voltages': voltages.astype(np.float32),
            'sigma': sigma.astype(np.float32),
            'mask': mask,
            'background': background,
            'n_inclusions': n_inclusions
        }

    def generate_dataset(
        self,
        n_samples: int,
        split: str = 'train',
        noise_range: Tuple[float, float] = (0.005, 0.02),
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        生成数据集

        Args:
            n_samples: 样本数
            split: 数据集划分 ('train', 'val', 'test')
            noise_range: 噪声范围
            seed: 随机种子

        Returns:
            dataset: 包含 voltages, sigmas, masks 的字典
        """
        print(f"\n生成 {split} 数据集 ({n_samples} 样本)...")

        np.random.seed(seed)

        # 预分配
        voltages = np.zeros((n_samples, self.n_freq, self.n_meas), dtype=np.float32)
        sigmas = np.zeros((n_samples, self.n_elems), dtype=np.float32)
        masks = np.zeros((n_samples, self.n_elems), dtype=np.float32)
        noise_levels = np.zeros(n_samples, dtype=np.float32)

        # 生成样本
        start_time = time.time()
        for i in range(n_samples):
            # 随机噪声水平
            noise = np.random.uniform(*noise_range)

            # 随机包含体类型
            inclusion_type = np.random.choice(['random', 'root'], p=[0.7, 0.3])

            sample = self.generate_sample(
                noise_level=noise,
                inclusion_type=inclusion_type
            )

            voltages[i] = sample['voltages']
            sigmas[i] = sample['sigma']
            masks[i] = sample['mask']
            noise_levels[i] = noise

            # 进度显示
            if (i + 1) % 100 == 0 or i == n_samples - 1:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (n_samples - i - 1)
                print(f"    进度: {i+1}/{n_samples} | "
                      f"耗时: {elapsed:.1f}s | ETA: {eta:.1f}s")

        return {
            'voltages': voltages,
            'sigmas': sigmas,
            'masks': masks,
            'noise_levels': noise_levels
        }

    def save_hdf5(
        self,
        output_path: str,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict
    ):
        """保存为 HDF5 格式（与现有训练流程兼容）"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n保存数据到: {output_path}")

        with h5py.File(output_path, 'w') as f:
            # 训练集
            train_grp = f.create_group('train')
            train_grp.create_dataset('voltages', data=train_data['voltages'], compression='gzip')
            train_grp.create_dataset('sigmas', data=train_data['sigmas'], compression='gzip')
            train_grp.create_dataset('masks', data=train_data['masks'], compression='gzip')
            train_grp.create_dataset('noise_db', data=train_data['noise_levels'])

            # 验证集
            val_grp = f.create_group('val')
            val_grp.create_dataset('voltages', data=val_data['voltages'], compression='gzip')
            val_grp.create_dataset('sigmas', data=val_data['sigmas'], compression='gzip')
            val_grp.create_dataset('masks', data=val_data['masks'], compression='gzip')
            val_grp.create_dataset('noise_db', data=val_data['noise_levels'])

            # 测试集
            test_grp = f.create_group('test')
            test_grp.create_dataset('voltages', data=test_data['voltages'], compression='gzip')
            test_grp.create_dataset('sigmas', data=test_data['sigmas'], compression='gzip')
            test_grp.create_dataset('masks', data=test_data['masks'], compression='gzip')
            test_grp.create_dataset('noise_db', data=test_data['noise_levels'])

            # 元数据
            meta_grp = f.create_group('metadata')
            meta_grp.create_dataset('frequencies', data=np.array(self.frequencies))
            meta_grp.create_dataset('dof_coords', data=self.dof_coords)
            meta_grp.attrs['n_elec'] = self.n_elec
            meta_grp.attrs['n_meas'] = self.n_meas
            meta_grp.attrs['n_elems'] = self.n_elems
            meta_grp.attrs['n_freq'] = self.n_freq
            meta_grp.attrs['domain_radius'] = self.domain_radius
            meta_grp.attrs['sigma_soil'] = self.sigma_soil
            meta_grp.attrs['sigma_root'] = self.sigma_root
            meta_grp.attrs['generator'] = 'PyEIDORS'

        print(f"✅ 数据保存成功")
        print(f"   文件大小: {output_path.stat().st_size / 1024**2:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='PyEIDORS 数据生成器')
    parser.add_argument('--config', type=str, default='config/mesh_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--n_train', type=int, default=10000,
                        help='训练样本数')
    parser.add_argument('--n_val', type=int, default=500,
                        help='验证样本数')
    parser.add_argument('--n_test', type=int, default=200,
                        help='测试样本数')
    parser.add_argument('--output', type=str, default='data/generated/pyeidors_dataset.h5',
                        help='输出路径')
    parser.add_argument('--noise_min', type=float, default=0.005,
                        help='最小噪声水平')
    parser.add_argument('--noise_max', type=float, default=0.02,
                        help='最大噪声水平')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 检查环境
    if not PYEIDORS_AVAILABLE:
        print("\n❌ PyEIDORS 未安装，请先安装:")
        print("   conda install -c conda-forge fenics")
        print("   pip install -e /path/to/PyEIDORS")
        sys.exit(1)

    # 创建生成器
    generator = PyEIDORSDataGenerator(config_path=args.config)

    # 生成训练集
    train_data = generator.generate_dataset(
        n_samples=args.n_train,
        split='train',
        noise_range=(args.noise_min, args.noise_max),
        seed=args.seed
    )

    # 生成验证集
    val_data = generator.generate_dataset(
        n_samples=args.n_val,
        split='val',
        noise_range=(args.noise_min, args.noise_max),
        seed=args.seed + 100
    )

    # 生成测试集
    test_data = generator.generate_dataset(
        n_samples=args.n_test,
        split='test',
        noise_range=(args.noise_min, args.noise_max),
        seed=args.seed + 200
    )

    # 保存数据
    generator.save_hdf5(args.output, train_data, val_data, test_data)

    print("\n" + "=" * 60)
    print("🎉 数据生成完成！")
    print("=" * 60)
    print(f"\n数据统计:")
    print(f"  训练集: {args.n_train} 样本")
    print(f"  验证集: {args.n_val} 样本")
    print(f"  测试集: {args.n_test} 样本")
    print(f"  输入维度: ({generator.n_freq}, {generator.n_meas})")
    print(f"  输出维度: ({generator.n_elems},)")
    print(f"\n下一步:")
    print(f"  python train.py --data {args.output}")


if __name__ == "__main__":
    main()
