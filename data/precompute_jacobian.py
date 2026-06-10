"""
预计算雅可比矩阵（灵敏度矩阵）
用于加速无监督训练中的物理约束计算
=====================================
J = dV/dσ, shape: (n_freq, n_measurements, n_elems)

用法:
    python precompute_jacobian.py --config config/mesh_config.yaml --output data/generated/jacobian.npy
"""

import os
import sys
import yaml
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.eit_forward import EITForwardSolver


def precompute_jacobian(config_path: str = "config/mesh_config.yaml",
                        output_path: str = "data/generated/jacobian.npy"):
    """
    计算并保存雅可比矩阵。

    J[freq, m, e] = ∂V_m / ∂σ_e  在频率 freq 下的灵敏度
    """
    solver = EITForwardSolver(config_path)
    n_freq = len(solver.frequencies)
    n_meas = solver.n_measurements
    n_elems = solver.n_elems

    print(f"计算雅可比矩阵: {n_freq} 频率 × {n_meas} 测量 × {n_elems} 单元")

    jacobians = np.zeros((n_freq, n_meas, n_elems), dtype=np.float32)

    for fi, freq in enumerate(solver.frequencies):
        print(f"  频率 {freq/1000:.0f} kHz ...")
        J = solver.get_jacobian(frequency=freq)
        jacobians[fi] = J.astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, jacobians)

    print(f"雅可比矩阵已保存: {output_path}")
    print(f"  Shape: {jacobians.shape}")
    print(f"  Size: {jacobians.nbytes / 1e6:.1f} MB")

    return jacobians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/mesh_config.yaml")
    parser.add_argument("--output", type=str, default="data/generated/jacobian.npy")
    args = parser.parse_args()
    precompute_jacobian(args.config, args.output)
