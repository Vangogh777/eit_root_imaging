"""
正方形内含物 EIT 数据集生成
===========================
生成正方形内含物的纯净测试/训练数据。

特性：
  - 背景均匀 (0.01 S/m)
  - 内含物为正方向方形 (0.05 S/m)
  - 位置随机，完全在边界内
  - 大小随机 (边长 0.015~0.05m)
  - 角度随机 (0~360°)
"""

import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.eit_forward import EITForwardSolver

DOMAIN_RADIUS = 0.10
BG_SIGMA = 0.01
INC_SIGMA = 0.05


def point_in_domain(x, y, margin=0.005):
    """点是否在圆形域内（带边距）"""
    return np.sqrt(x**2 + y**2) < DOMAIN_RADIUS - margin


def generate_square_sample(centers, solver, seed):
    """生成一个正方形内含物样本"""
    rng = np.random.RandomState(seed)
    n_elems = len(centers)

    while True:
        # 正方形参数
        side = rng.uniform(0.025, 0.055)         # 边长 2.5~5.5cm
        angle = rng.uniform(0, np.pi)              # 旋转角 0~180°
        max_dist = DOMAIN_RADIUS - side * 0.8 - 0.01
        dist = rng.uniform(0, max_dist)
        theta = rng.uniform(0, 2 * np.pi)
        cx = dist * np.cos(theta)
        cy = dist * np.sin(theta)

        # 正方形的四个角（旋转前）
        h = side / 2
        corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])

        # 旋转
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        corners = corners @ rot.T + np.array([cx, cy])

        # 验证所有角都在域内
        if all(point_in_domain(cx, cy) for cx, cy in corners):
            break

    # 构建电导率（点在正方形内的判断）
    sigma = np.full(n_elems, BG_SIGMA, dtype=np.float32)
    mask = np.zeros(n_elems, dtype=np.float32)

    # 将单元中心转换到正方形的局部坐标系
    vec = centers - np.array([cx, cy])
    # 逆旋转
    inv_rot = np.array([[c, s], [-s, c]])
    local = vec @ inv_rot.T  # (n_elems, 2)

    inside = (np.abs(local[:, 0]) < h) & (np.abs(local[:, 1]) < h)
    sigma[inside] = INC_SIGMA
    mask[inside] = 1.0

    # 求解电压
    V = solver.solve_multi_frequency(sigma)
    noise_db = rng.uniform(-40, -20)
    V_noisy = solver.add_noise(V, noise_db)

    return sigma, mask, V_noisy.astype(np.float32)


def generate_dataset(config_path="config/mesh_config.yaml",
                     n_train=10000, n_val=500, n_test=200,
                     output_dir="data/generated", seed=42):
    """生成正方形数据集"""

    os.makedirs(output_dir, exist_ok=True)

    solver = EITForwardSolver(config_path)
    centers = solver.element_centers
    if centers.shape[1] > 2:
        centers = centers[:, :2]
    n_freq = len(solver.frequencies)
    n_meas = solver.n_measurements
    n_elems = solver.n_elems
    print(f"网格: {n_elems} 单元, 分辨率 {solver.cfg['mesh']['mesh_resolution']*1000:.1f}mm")

    import h5py

    def _gen_split(name, n, start_seed):
        print(f"生成 {name} 集 ({n} 样本)...")
        sigmas = np.zeros((n, n_elems), dtype=np.float32)
        masks = np.zeros((n, n_elems), dtype=np.float32)
        voltages = np.zeros((n, n_freq, n_meas), dtype=np.float32)

        for i in tqdm(range(n), desc=name):
            s, m, v = generate_square_sample(centers, solver, start_seed + i)
            sigmas[i] = s
            masks[i] = m
            voltages[i] = v
        return sigmas, masks, voltages

    train_s, train_m, train_v = _gen_split("train", n_train, seed)
    val_s, val_m, val_v = _gen_split("val", n_val, seed + n_train + 1000)
    test_s, test_m, test_v = _gen_split("test", n_test, seed + n_train + n_val + 2000)

    output_path = os.path.join(output_dir, "square_dataset.h5")
    print(f"保存到 {output_path} ...")

    with h5py.File(output_path, 'w') as f:
        for name, sigmas, masks, voltages in [
            ("train", train_s, train_m, train_v),
            ("val", val_s, val_m, val_v),
            ("test", test_s, test_m, test_v),
        ]:
            grp = f.create_group(name)
            grp.create_dataset('voltages', data=voltages, compression='gzip')
            grp.create_dataset('sigmas', data=sigmas, compression='gzip')
            grp.create_dataset('masks', data=masks, compression='gzip')
            grp.create_dataset('noise_db', data=np.zeros(len(sigmas), dtype=np.float32))

        meta = f.create_group('metadata')
        meta.create_dataset('mesh_nodes', data=solver.mesh.node)
        meta.create_dataset('mesh_elements', data=solver.mesh.element)
        meta.create_dataset('frequencies', data=np.array(solver.frequencies))
        meta.create_dataset('config', data=yaml.dump(solver.cfg).encode())

    print(f"完成! 训练: {n_train}, 验证: {n_val}, 测试: {n_test}")
    print(f"  电压: ({n_freq}, {n_meas}) → 电导率 ({n_elems})")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/mesh_config.yaml")
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_val", type=int, default=2)
    parser.add_argument("--n_test", type=int, default=2)
    parser.add_argument("--output", default="data/generated")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        config_path=args.config,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        output_dir=args.output,
        seed=args.seed,
    )
