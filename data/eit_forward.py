"""
pyEIT 正问题求解器（桶式 2D EIT）
====================================
硬件对应：圆柱形桶，单环16电极，2D截面成像。
封装了网格创建 → 电极配置 → 多频正向仿真 → 噪声注入 → 电压输出全流程。
"""

import os
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# pyEIT 核心模块
from pyeit.mesh import create, PyEITMesh
from pyeit.eit.fem import forward as fem_forward
from pyeit.eit.utils import eit_scan_lines


@dataclass
class EITMeasurement:
    """单次 EIT 测量数据容器"""
    voltage: np.ndarray          # (n_freq, n_measurements) 边界电压
    sigma: np.ndarray            # (n_elems,) 真实电导率分布 (ground truth, 仅仿真用)
    frequency: float             # 激励频率 (Hz)
    noise_db: float              # 实际注入噪声 (dB)
    mesh_nodes: np.ndarray       # (n_nodes, 2) 网格节点坐标
    mesh_elements: np.ndarray    # (n_elems, 3) 单元节点索引
    electrode_positions: np.ndarray  # (n_el, 2) 电极位置


class EITForwardSolver:
    """
    pyEIT 正问题求解器

    用法:
        solver = EITForwardSolver("config/mesh_config.yaml")
        # 从电导率分布生成边界电压
        voltage = solver.solve(sigma)                          # 单频
        voltages = solver.solve_multi_frequency(sigma)         # 多频
        # 批量生成
        dataset = solver.generate_dataset([sigma1, sigma2, ...])
    """

    def __init__(self, config_path: str = "config/mesh_config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        mesh_cfg = self.cfg['mesh']
        elec_cfg = self.cfg['electrodes']
        stim_cfg = self.cfg['stimulation']
        self.gt_cfg = self.cfg['ground_truth']

        self.frequencies = stim_cfg['frequencies']
        self.n_el = elec_cfg['count']
        self.amplitude = stim_cfg['amplitude']
        self.pattern = stim_cfg['pattern']

        # --- 1. 创建 2D 圆形网格（桶截面） ---
        self.mesh: PyEITMesh = self._create_mesh(mesh_cfg, elec_cfg)
        self.n_elems = self.mesh['element'].shape[0]
        self.n_nodes = self.mesh['node'].shape[0]

        # --- 2. 预计算扫描协议 ---
        # pyEIT 返回: (n_stim, n_el) 的激励-测量配对
        if self.pattern == 'adjacent':
            self.protocol = eit_scan_lines(self.n_el, 16)
        else:
            raise ValueError(f"不支持的激励模式: {self.pattern}")
        self.n_measurements = self.protocol.shape[0]

        # --- 3. 预计算 FEM 正问题算子（各频率独立） ---
        self._forward_ops = {}
        for f in self.frequencies:
            op = fem_forward(self.mesh, self.protocol, f)
            self._forward_ops[f] = op

        # --- 4. 缓存均匀场电导率的参考电压（用于差分EIT） ---
        self.sigma_uniform = np.full(self.n_elems, self.gt_cfg['conductivity_soil'])
        self.V_uniform = {}  # 各频率下的均匀场电压
        for f in self.frequencies:
            op = self._forward_ops[f]
            self.V_uniform[f] = op.solve(self.sigma_uniform)[1]

        print(f"[EITForwardSolver] 网格: {self.n_elems} 单元, {self.n_nodes} 节点")
        print(f"[EITForwardSolver] 电极: {self.n_el} | 测量: {self.n_measurements}")
        print(f"[EITForwardSolver] 频率: {self.frequencies} Hz")

    def _create_mesh(self, mesh_cfg: dict, elec_cfg: dict) -> PyEITMesh:
        """创建 2D 圆形网格（桶截面）"""
        if mesh_cfg['type'] == 'circle':
            mesh = create(
                n_el=elec_cfg['count'],
                fd=mesh_cfg.get('radius', 0.1),
                h0=mesh_cfg.get('mesh_resolution', 0.005)
            )
        elif mesh_cfg['type'] == 'rectangle':
            from pyeit.mesh import rectangle
            w = mesh_cfg.get('width', 0.2)
            h = mesh_cfg.get('height', 0.3)
            mesh = rectangle(n_el=elec_cfg['count'], w=w, h=h)
        else:
            raise ValueError(f"Unknown mesh type: {mesh_cfg['type']}")
        return mesh

    @property
    def element_centers(self) -> np.ndarray:
        """返回每个三角单元的中心坐标 (n_elems, 2)"""
        nodes = self.mesh['node']
        elems = self.mesh['element']
        return np.mean(nodes[elems], axis=1)

    def solve(self, sigma: np.ndarray, frequency: Optional[float] = None) -> np.ndarray:
        """
        求解 EIT 正问题

        参数:
            sigma: (n_elems,) 电导率分布
            frequency: 频率 (Hz)，默认使用第一个频率

        返回:
            voltage: (n_measurements,) 边界差分电压
        """
        if frequency is None:
            frequency = self.frequencies[0]

        op = self._forward_ops[frequency]
        # pyEIT fem_forward.solve 返回 (conductivity_map, voltage, ...)
        _, v = op.solve(sigma)

        # 差分电压：相对于均匀场
        v_diff = v - self.V_uniform[frequency]
        return v_diff

    def solve_current(self, sigma: np.ndarray, frequency: Optional[float] = None) -> np.ndarray:
        """
        返回绝对电压（非差分），部分方法需要
        """
        if frequency is None:
            frequency = self.frequencies[0]
        op = self._forward_ops[frequency]
        _, v = op.solve(sigma)
        return v

    def solve_multi_frequency(self, sigma: np.ndarray) -> np.ndarray:
        """
        多频率求解

        返回:
            voltages: (n_freq, n_measurements) 多频边界电压
        """
        V_list = []
        for f in self.frequencies:
            v = self.solve(sigma, frequency=f)
            V_list.append(v)
        return np.stack(V_list, axis=0)

    def add_noise(self, voltage: np.ndarray, noise_db: float = -30) -> np.ndarray:
        """
        向测量电压添加高斯白噪声

        参数:
            voltage: (n_measurements,) 或 (n_freq, n_measurements)
            noise_db: 信噪比 (dB), 如 -30 = 30dB SNR

        返回:
            加噪后的电压
        """
        signal_power = np.mean(voltage ** 2)
        noise_power = signal_power / (10 ** (-noise_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(*voltage.shape)
        return voltage + noise

    def simulate_contact_impedance_drift(self, sigma: np.ndarray, drift_scale: float = 0.02) -> np.ndarray:
        """
        模拟电极接触阻抗漂移（实际测量中常见问题）
        通过对特定电极附近的电导率加微小扰动来模拟
        """
        sigma_noisy = sigma.copy()
        # 在电极附近单元加随机扰动
        el_pos = np.array(self.mesh['el_pos'])  # 电极索引
        centers = self.element_centers

        for i, el_idx in enumerate(el_pos[:self.n_el]):
            node_pos = self.mesh['node'][el_idx]
            # 找该电极附近的单元
            dists = np.linalg.norm(centers - node_pos, axis=1)
            nearby = dists < 0.02
            sigma_noisy[nearby] *= (1 + drift_scale * np.random.randn())
        return sigma_noisy

    def generate_measurement(self, sigma: np.ndarray, noise_db: Optional[float] = None,
                             add_drift: bool = False) -> EITMeasurement:
        """
        生成单次完整测量

        返回:
            EITMeasurement 包含电压、电导率、网格信息等
        """
        if noise_db is None:
            noise_db = np.random.uniform(*self.cfg['data']['noise_level_db'])

        sigma_used = sigma.copy()
        if add_drift:
            sigma_used = self.simulate_contact_impedance_drift(sigma_used)

        V = self.solve_multi_frequency(sigma_used)
        V_noisy = self.add_noise(V, noise_db)

        return EITMeasurement(
            voltage=V_noisy,
            sigma=sigma,
            frequency=np.array(self.frequencies),
            noise_db=noise_db,
            mesh_nodes=self.mesh['node'],
            mesh_elements=self.mesh['element'],
            electrode_positions=self.mesh['node'][self.mesh['el_pos'][:self.n_el]]
        )

    def generate_dataset(self, sigma_list: List[np.ndarray],
                         noise_db_range: Tuple[float, float] = (-40, -20),
                         add_drift_prob: float = 0.0) -> List[EITMeasurement]:
        """
        批量生成数据集

        参数:
            sigma_list: 电导率分布列表
            noise_db_range: 噪声范围 (min_db, max_db)
            add_drift_prob: 模拟接触阻抗漂移的概率

        返回:
            measurements: EITMeasurement 列表
        """
        measurements = []
        for sigma in sigma_list:
            noise_db = np.random.uniform(*noise_db_range)
            add_drift = np.random.rand() < add_drift_prob
            meas = self.generate_measurement(sigma, noise_db, add_drift)
            measurements.append(meas)
        return measurements

    def get_jacobian(self, frequency: Optional[float] = None) -> np.ndarray:
        """
        获取灵敏度矩阵（雅可比矩阵）J = dV/dσ
        shape: (n_measurements, n_elems)
        """
        if frequency is None:
            frequency = self.frequencies[0]
        op = self._forward_ops[frequency]
        # pyEIT 提供 jacobian 计算
        from pyeit.eit.fem import jacobian
        J = jacobian(self.mesh, self.protocol, frequency)
        return J

    def get_reference_voltage(self, frequency: Optional[float] = None) -> np.ndarray:
        """获取均匀场参考电压"""
        if frequency is None:
            frequency = self.frequencies[0]
        return self.V_uniform[frequency]


if __name__ == "__main__":
    # 快速测试
    solver = EITForwardSolver("config/mesh_config.yaml")
    print(f"网格单元数: {solver.n_elems}")
    print(f"测量通道数: {solver.n_measurements}")

    # 生成一个简单根测试
    sigma = np.full(solver.n_elems, 0.01)  # 土壤背景
    centers = solver.element_centers
    # 在中心放一个高电导率区域模拟根
    dist = np.linalg.norm(centers, axis=1)
    sigma[dist < 0.02] = 0.05

    V = solver.solve_multi_frequency(sigma)
    print(f"电压 shape: {V.shape}")
    print(f"频率 {solver.frequencies[0]} Hz 电压范围: [{V[0].min():.4f}, {V[0].max():.4f}]")
    print("[测试通过] EITForwardSolver 工作正常")
