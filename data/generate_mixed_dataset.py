"""
多样化 EIT 数据集生成 (Phase 2.1)
=================================
生成多种形状内含物的 EIT 训练/测试数据。

形状类型 (等比例):
  - circle:  单圆 (25%)
  - ellipse: 椭圆 (25%)
  - double_circle: 双圆 (25%)
  - square: 正方形 (25%)

特性:
  - 背景均匀 (0.01 S/m)
  - 内含物随机位置、大小、对比度
  - 支持边缘/中心平衡采样
"""
import os, sys, argparse, yaml
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.eit_forward import EITForwardSolver

DOMAIN_RADIUS = 0.10
BG_SIGMA = 0.01
INC_SIGMA_BASE = 0.05  # 基础内含物电导率 (会被 contrast 缩放)

SHAPE_TYPES = ['circle', 'ellipse', 'double_circle', 'square']
SHAPE_WEIGHTS = [0.25, 0.25, 0.25, 0.25]


def _sample_position(rng, max_r):
    """在域内随机采样一个位置 (确保内含物完全在桶内)"""
    while True:
        angle = rng.uniform(0, 2 * np.pi)
        max_dist = DOMAIN_RADIUS - max_r - 0.003
        lo = min(getattr(_generate_one, 'edge_threshold', 0.05), max_dist * 0.95)
        if rng.random() < getattr(_generate_one, 'edge_ratio', 0.5):
            dist = rng.uniform(lo, max_dist)  # 边缘
        else:
            dist = rng.uniform(0, min(0.04, max_dist))  # 中心
        cx, cy = dist * np.cos(angle), dist * np.sin(angle)
        if np.sqrt(cx**2 + cy**2) + max_r < DOMAIN_RADIUS - 0.003:
            return cx, cy


def _gen_circle(rng, centers_np):
    """单圆"""
    r = rng.uniform(0.008, 0.030)
    cx, cy = _sample_position(rng, r)
    dist = np.sqrt((centers_np[:, 0] - cx)**2 + (centers_np[:, 1] - cy)**2)
    return dist < r, cx, cy, r, 'circle'


def _gen_ellipse(rng, centers_np):
    """椭圆"""
    a = rng.uniform(0.015, 0.040)  # 半长轴
    b = rng.uniform(0.008, 0.020)  # 半短轴
    max_r = max(a, b)
    cx, cy = _sample_position(rng, max_r)
    angle = rng.uniform(0, np.pi)
    dx = centers_np[:, 0] - cx
    dy = centers_np[:, 1] - cy
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    return (rx / a)**2 + (ry / b)**2 <= 1.0, cx, cy, max_r, 'ellipse'


def _gen_double_circle(rng, centers_np):
    """双圆 (两个不重叠的圆)"""
    mask = np.zeros(len(centers_np), dtype=bool)
    positions = []
    for _ in range(2):
        for attempt in range(20):
            r = rng.uniform(0.008, 0.025)
            cx, cy = _sample_position(rng, r)
            # 确保与已放置的圆不重叠
            ok = True
            for px, py, pr in positions:
                if np.sqrt((cx - px)**2 + (cy - py)**2) < pr + r + 0.002:
                    ok = False
                    break
            if ok:
                break
        positions.append((cx, cy, r))
        dist = np.sqrt((centers_np[:, 0] - cx)**2 + (centers_np[:, 1] - cy)**2)
        mask |= dist < r
    return mask, cx, cy, max(r for _, _, r in positions), 'double_circle'


def _gen_square(rng, centers_np):
    """正方形 (随机旋转)"""
    half = rng.uniform(0.01, 0.035)
    max_r = half * np.sqrt(2)
    cx, cy = _sample_position(rng, max_r)
    angle = rng.uniform(0, np.pi / 4)
    dx = centers_np[:, 0] - cx
    dy = centers_np[:, 1] - cy
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    return (np.abs(rx) <= half) & (np.abs(ry) <= half), cx, cy, max_r, 'square'


SHAPE_GENERATORS = {
    'circle': _gen_circle,
    'ellipse': _gen_ellipse,
    'double_circle': _gen_double_circle,
    'square': _gen_square,
}


def _worker_init(config_path):
    global _solver
    _solver = EITForwardSolver(config_path)


def _generate_one(seed):
    """生成一个随机形状样本"""
    global _solver
    rng = np.random.RandomState(seed)
    centers = _solver.element_centers
    n_elems = _solver.n_elems

    # 随机选形状类型
    shape = rng.choice(SHAPE_TYPES, p=SHAPE_WEIGHTS)
    gen_fn = SHAPE_GENERATORS[shape]
    mask, cx, cy, max_r, shape_name = gen_fn(rng, centers)

    # 随机对比度 (3x ~ 10x)
    contrast = rng.uniform(3.0, 10.0)
    inc_sigma = BG_SIGMA * contrast

    sigma = np.full(n_elems, BG_SIGMA, dtype=np.float32)
    sigma[mask] = inc_sigma

    # 求解电压
    V = _solver.solve_multi_frequency(sigma)
    noise_db = rng.uniform(-40, -20)
    V_noisy = _solver.add_noise(V, noise_db)

    return {
        'sigma': sigma,
        'mask': mask.astype(np.float32),
        'voltage': V_noisy.astype(np.float32),
        'noise_db': noise_db,
        'shape': shape_name,
        'contrast': contrast,
        'cx': float(cx), 'cy': float(cy),
    }


def generate_dataset(config_path="config/mesh_config.yaml",
                     n_train=10000, n_val=500, n_test=200,
                     output_dir="data/generated",
                     workers=0, seed=42,
                     edge_ratio=0.5, edge_threshold=0.05):
    """生成多样化数据集"""
    os.makedirs(output_dir, exist_ok=True)

    solver = EITForwardSolver(config_path)
    n_freq = len(solver.frequencies)
    n_meas = solver.n_measurements
    n_elems = solver.n_elems
    print(f"网格: {n_elems} 单元")

    import h5py
    _generate_one.edge_ratio = edge_ratio
    _generate_one.edge_threshold = edge_threshold
    print(f"形状: {SHAPE_TYPES} (权重: {SHAPE_WEIGHTS})")
    print(f"对比度: 3x ~ 10x (随机)")
    print(f"采样: {edge_ratio*100:.0f}%边缘/{100-edge_ratio*100:.0f}%中心 (阈值 {edge_threshold*100:.0f}cm)")

    def _gen_split(name, n, start_seed):
        print(f"生成 {name} 集 ({n} 样本)...")
        if workers > 1:
            n_proc = min(workers, cpu_count(), n)
            with Pool(n_proc, initializer=_worker_init, initargs=(config_path,)) as pool:
                results = list(tqdm(pool.imap(_generate_one, range(start_seed, start_seed + n)), total=n))
        else:
            _worker_init(config_path)
            results = [_generate_one(s) for s in tqdm(range(start_seed, start_seed + n))]

        sigmas = np.stack([r['sigma'] for r in results])
        masks = np.stack([r['mask'] for r in results])
        voltages = np.stack([r['voltage'] for r in results])
        shapes = [r['shape'] for r in results]
        contrasts = [r['contrast'] for r in results]
        return sigmas, masks, voltages, shapes, contrasts

    train_s, train_m, train_v, train_shapes, train_contrast = _gen_split("train", n_train, seed)
    val_s, val_m, val_v, val_shapes, val_contrast = _gen_split("val", n_val, seed + n_train + 1000)
    test_s, test_m, test_v, test_shapes, test_contrast = _gen_split("test", n_test, seed + n_train + n_val + 2000)

    output_path = os.path.join(output_dir, "mixed_dataset.h5")
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

    # 打印形状分布
    if train_shapes:
        from collections import Counter
        dist = Counter(train_shapes)
        print(f"训练集形状分布: {dict(dist)}")

    print(f"完成! 训练:{n_train}, 验证:{n_val}, 测试:{n_test}")
    return output_path


def generate_preview(output_dir="data/generated"):
    """生成少量样本用于可视化预览"""
    os.makedirs(os.path.join(output_dir, "preview"), exist_ok=True)
    solver = EITForwardSolver("config/mesh_config.yaml")
    centers = solver.element_centers[:, :2]
    elements = solver.mesh.element
    _worker_init("config/mesh_config.yaml")

    rng = np.random.RandomState(42)
    samples = []
    for i in range(20):
        result = _generate_one(42 + i)
        samples.append(result)

    np.savez(os.path.join(output_dir, "preview/preview_samples.npz"),
             sigmas=np.stack([s['sigma'] for s in samples]),
             masks=np.stack([s['mask'] for s in samples]),
             shapes=np.array([s['shape'] for s in samples]),
             contrasts=np.array([s['contrast'] for s in samples]),
             centers=centers, elements=elements)
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/mesh_config.yaml")
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--output", default="data/generated")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge_ratio", type=float, default=0.5)
    parser.add_argument("--edge_threshold", type=float, default=0.05)
    parser.add_argument("--preview", action="store_true",
                        help="只生成预览样本 (20个)")
    args = parser.parse_args()

    if args.preview:
        print("生成预览样本...")
        samples = generate_preview(args.output)
        print(f"生成了 {len(samples)} 个预览样本, 保存到 {args.output}/preview/")
    else:
        generate_dataset(
            config_path=args.config,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            output_dir=args.output,
            workers=args.workers,
            seed=args.seed,
            edge_ratio=args.edge_ratio,
            edge_threshold=args.edge_threshold,
        )
