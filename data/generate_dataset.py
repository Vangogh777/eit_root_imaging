"""
批量生成 HDF5 训练/验证/测试数据集
====================================
用法:
    python generate_dataset.py                    # 使用默认配置
    python generate_dataset.py --n_train 5000     # 指定数量
    python generate_dataset.py --visualize        # 可视化样本
"""

import os
import sys
import h5py
import yaml
import argparse
import numpy as np
from tqdm import tqdm

# 添加项目根到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.eit_forward import EITForwardSolver
from data.root_simulator import RootSystemGenerator


def generate_dataset(config_path: str = "config/mesh_config.yaml",
                     n_train: int = 10000, n_val: int = 500, n_test: int = 200,
                     output_dir: str = "data/generated",
                     seed_start: int = 0, visualize: bool = False):
    """
    生成完整的 EIT 数据集并保存为 HDF5 格式。

    输出 HDF5 结构:
        /train
            /voltages    (n_train, n_freq, n_meas)
            /sigmas      (n_train, n_elems)
            /masks       (n_train, n_elems)  # 二值根标注
            /noise_db    (n_train,)
        /val (同 train)
        /test (同 train)
        /metadata
            /mesh_nodes      (n_nodes, 2)
            /mesh_elements   (n_elems, 3)
            /frequencies     (n_freq,)
            /electrode_positions (n_el, 2)
            /config_yaml     str
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 初始化正问题求解器
    print("[1/5] 初始化 EITForwardSolver ...")
    solver = EITForwardSolver(config_path)
    n_freq = len(solver.frequencies)
    n_meas = solver.n_measurements
    n_elems = solver.n_elems

    # 2. 初始化根生成器
    print("[2/5] 初始化 RootSystemGenerator ...")
    gen = RootSystemGenerator(
        solver.mesh['node'],
        solver.mesh['element'],
        domain_radius=solver.cfg['mesh']['radius'],
        conductivity_root=solver.gt_cfg['conductivity_root'],
        conductivity_soil=solver.gt_cfg['conductivity_soil']
    )

    # 3. 生成数据
    def _generate_one(seed: int):
        """生成一对 (sigma, mask, voltage)"""
        sigma, mask = gen.generate_with_label(seed=seed)
        V = solver.solve_multi_frequency(sigma)
        noise_db = np.random.uniform(*solver.cfg['data']['noise_db_range']
                                      if 'noise_db_range' in solver.cfg['data']
                                      else solver.cfg['data']['noise_level_db'])
        V_noisy = solver.add_noise(V, noise_db)
        return sigma, mask, V_noisy, noise_db

    def _generate_split(name: str, n: int, start_seed: int):
        """生成一个 split (train/val/test)"""
        print(f"[3/5] 生成 {name} 集 ({n} 样本) ...")
        sigmas = np.zeros((n, n_elems), dtype=np.float32)
        masks = np.zeros((n, n_elems), dtype=np.float32)
        voltages = np.zeros((n, n_freq, n_meas), dtype=np.float32)
        noise_dbs = np.zeros(n, dtype=np.float32)

        for i in tqdm(range(n), desc=f"{name}"):
            s, m, v, ndb = _generate_one(start_seed + i)
            sigmas[i] = s
            masks[i] = m
            voltages[i] = v
            noise_dbs[i] = ndb

        return sigmas, masks, voltages, noise_dbs

    train_sigmas, train_masks, train_voltages, train_noise = \
        _generate_split("train", n_train, seed_start)
    val_sigmas, val_masks, val_voltages, val_noise = \
        _generate_split("val", n_val, seed_start + n_train + 1000)
    test_sigmas, test_masks, test_voltages, test_noise = \
        _generate_split("test", n_test, seed_start + n_train + n_val + 2000)

    # 4. 保存 HDF5
    output_path = os.path.join(output_dir, "eit_dataset.h5")
    print(f"[4/5] 保存到 {output_path} ...")

    with h5py.File(output_path, 'w') as f:
        # 训练
        grp_train = f.create_group('train')
        grp_train.create_dataset('voltages', data=train_voltages, compression='gzip')
        grp_train.create_dataset('sigmas', data=train_sigmas, compression='gzip')
        grp_train.create_dataset('masks', data=train_masks, compression='gzip')
        grp_train.create_dataset('noise_db', data=train_noise)

        # 验证
        grp_val = f.create_group('val')
        grp_val.create_dataset('voltages', data=val_voltages, compression='gzip')
        grp_val.create_dataset('sigmas', data=val_sigmas, compression='gzip')
        grp_val.create_dataset('masks', data=val_masks, compression='gzip')
        grp_val.create_dataset('noise_db', data=val_noise)

        # 测试
        grp_test = f.create_group('test')
        grp_test.create_dataset('voltages', data=test_voltages, compression='gzip')
        grp_test.create_dataset('sigmas', data=test_sigmas, compression='gzip')
        grp_test.create_dataset('masks', data=test_masks, compression='gzip')
        grp_test.create_dataset('noise_db', data=test_noise)

        # 元数据
        grp_meta = f.create_group('metadata')
        grp_meta.create_dataset('mesh_nodes', data=solver.mesh['node'])
        grp_meta.create_dataset('mesh_elements', data=solver.mesh['element'])
        grp_meta.create_dataset('frequencies', data=np.array(solver.frequencies))
        grp_meta.create_dataset('electrode_positions',
                                 data=solver.mesh['node'][solver.mesh['el_pos'][:solver.n_el]])
        grp_meta.create_dataset('config_yaml',
                                 data=yaml.dump(solver.cfg).encode('utf-8'))

    print(f"[5/5] 完成！数据集大小:")
    print(f"  训练: {n_train} 样本 ({train_voltages.nbytes / 1e6:.1f} MB)")
    print(f"  验证: {n_val} 样本")
    print(f"  测试: {n_test} 样本")
    print(f"  每个样本: 电压 ({n_freq}×{n_meas}) → 电导率 ({n_elems})")

    # 可视化（可选）
    if visualize:
        _visualize_sample(solver, train_sigmas[0], train_voltages[0], output_dir)

    return output_path


def _visualize_sample(solver, sigma, voltages, output_dir):
    """可视化一个样本用于检查"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 电导率分布
    ax = axes[0]
    centers = solver.element_centers
    sc = ax.scatter(centers[:, 0], centers[:, 1], c=sigma, s=10,
                    cmap='viridis', vmin=0.005, vmax=0.055)
    plt.colorbar(sc, ax=ax)
    ax.set_title("电导率分布 σ")
    ax.set_aspect('equal')

    # 多频电压
    ax = axes[1]
    for i, f in enumerate(solver.frequencies):
        ax.plot(voltages[i], label=f"{f/1000:.0f} kHz", alpha=0.7)
    ax.set_title("边界电压 V")
    ax.set_xlabel("测量通道")
    ax.legend(fontsize=8)

    # 电压幅度谱
    ax = axes[2]
    v_rms = np.sqrt(np.mean(voltages**2, axis=1))
    ax.semilogx(solver.frequencies, v_rms, 'o-')
    ax.set_title("RMS 电压 vs 频率")
    ax.set_xlabel("频率 (Hz)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "sample_check.png")
    plt.savefig(save_path, dpi=150)
    print(f"  样本可视化已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 EIT 数据集")
    parser.add_argument("--config", type=str, default="config/mesh_config.yaml")
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/generated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    generate_dataset(
        config_path=args.config,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        output_dir=args.output,
        seed_start=args.seed,
        visualize=args.visualize
    )
