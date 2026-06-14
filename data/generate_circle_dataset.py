"""
单圆 EIT 数据集生成
===================
生成纯净的单圆内含物测试/训练数据。

特性：
  - 背景均匀 (0.01 S/m)
  - 内含物为正圆形 (0.05 S/m)
  - 位置随机，完全在边界内
  - 半径随机 (1~3cm)
  - 可调整网格分辨率
"""

import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.eit_forward import EITForwardSolver


DOMAIN_RADIUS = 0.10   # 桶半径 10cm
BG_SIGMA = 0.01        # 土壤背景
INC_SIGMA = 0.05       # 内含物


def _worker_init(config_path):
    """每个 worker 初始化求解器"""
    global _solver
    _solver = EITForwardSolver(config_path)


def _generate_one(seed):
    """生成一个单圆样本"""
    global _solver
    rng = np.random.RandomState(seed)

    # 随机圆参数
    while True:
        r = rng.uniform(0.008, 0.030)          # 半径 0.8~3cm
        max_dist = DOMAIN_RADIUS - r - 0.005
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, max_dist)
        cx = dist * np.cos(angle)
        cy = dist * np.sin(angle)
        if np.sqrt(cx**2 + cy**2) + r < DOMAIN_RADIUS - 0.003:
            break

    # 构建电导率
    centers = _solver.element_centers
    n_elems = _solver.n_elems
    sigma = np.full(n_elems, BG_SIGMA, dtype=np.float32)
    mask = np.zeros(n_elems, dtype=np.float32)

    dist = np.sqrt((centers[:, 0] - cx)**2 + (centers[:, 1] - cy)**2)
    inside = dist < r
    sigma[inside] = INC_SIGMA
    mask[inside] = 1.0

    # 求解电压
    V = _solver.solve_multi_frequency(sigma)
    noise_db = rng.uniform(-40, -20)
    V_noisy = _solver.add_noise(V, noise_db)

    return {
        'sigma': sigma,
        'mask': mask,
        'voltage': V_noisy.astype(np.float32),
        'noise_db': noise_db,
        'cx': cx, 'cy': cy, 'r': r,
    }


def generate_dataset(config_path="config/mesh_config.yaml",
                     n_train=10000, n_val=500, n_test=200,
                     output_dir="data/generated",
                     workers=0, seed=42):
    """生成单圆数据集并保存为 HDF5"""

    os.makedirs(output_dir, exist_ok=True)

    # 初始化求解器（获取元数据）
    solver = EITForwardSolver(config_path)
    n_freq = len(solver.frequencies)
    n_meas = solver.n_measurements
    n_elems = solver.n_elems
    print(f"网格: {n_elems} 单元, 分辨率 {solver.cfg['mesh']['mesh_resolution']*1000:.1f}mm")

    import h5py

    def _gen_split(name, n, start_seed):
        print(f"生成 {name} 集 ({n} 样本)...")
        if workers > 1:
            n_proc = min(workers, cpu_count(), n)
            print(f"  使用 {n_proc} 进程...")
            with Pool(n_proc, initializer=_worker_init, initargs=(config_path,)) as pool:
                results = list(tqdm(pool.imap(_generate_one, range(start_seed, start_seed + n)),
                                    total=n))
        else:
            _worker_init(config_path)
            results = [_generate_one(s) for s in tqdm(range(start_seed, start_seed + n))]

        sigmas = np.stack([r['sigma'] for r in results])
        masks = np.stack([r['mask'] for r in results])
        voltages = np.stack([r['voltage'] for r in results])
        return sigmas, masks, voltages

    train_s, train_m, train_v = _gen_split("train", n_train, seed)
    val_s, val_m, val_v = _gen_split("val", n_val, seed + n_train + 1000)
    test_s, test_m, test_v = _gen_split("test", n_test, seed + n_train + n_val + 2000)

    # 保存 HDF5
    output_path = os.path.join(output_dir, "circle_dataset.h5")
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
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--output", default="data/generated")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        config_path=args.config,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        output_dir=args.output,
        workers=args.workers,
        seed=args.seed,
    )
