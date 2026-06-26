"""
多样化 EIT 数据集生成 v4 (多背景增强版)
=================================
生成多种形状内含物的 EIT 训练/测试数据，支持多背景电导率域泛化。

v4 新增:
  - 多背景电导率支持 (0.005 ~ 0.3 S/m, 6档)
  - 对比度外推测试集 (contrast 6x~15x, OOD测试)
  - 背景电导率元数据追踪

v3 新增:
  - 环形内含物 (ring)
  - 近边界硬样本 (near-boundary)
  - 系统噪声测试集 (-30dB, -15dB fixed)
  - 元数据保存 (shape, contrast, noise_db)

形状类型:
  - circle:        单圆 (20%)
  - ellipse:       椭圆 (18%)
  - double_circle: 双圆 (18%)
  - square:        正方形 (14%)
  - ring:          环形 (15%)
  - near_boundary: 近边圆/椭圆 (15%, hardest)

特性:
  - 多背景电导率 (默认随机从6档采样, 可通过 --uniform_bg 固定为0.01)
  - 内含物随机位置、大小、对比度 (3x ~ 10x; 外推测试集 6x ~ 15x)
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
# Multi-background support for domain generalization
BACKGROUND_SIGMAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
BACKGROUND_WEIGHTS = [0.15, 0.25, 0.20, 0.20, 0.15, 0.05]  # weight toward common values

INC_SIGMA_BASE = 0.05  # 基础内含物电导率 (会被 contrast 缩放)

SHAPE_TYPES = ['circle', 'ellipse', 'double_circle', 'square', 'ring', 'near_boundary']
SHAPE_WEIGHTS = [0.20, 0.18, 0.18, 0.14, 0.15, 0.15]


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


def _sample_position_near_boundary(rng, max_r):
    """靠近边界的强制采样 (距边界 < 0.02m) — 对EIT最难的场景"""
    for _ in range(50):
        angle = rng.uniform(0, 2 * np.pi)
        # 强制靠近边界: 距边界 0.003 ~ 0.02m
        max_allowed = DOMAIN_RADIUS - max_r - 0.003
        min_allowed = max(0.0, DOMAIN_RADIUS - max_r - 0.025)
        if min_allowed >= max_allowed:
            min_allowed = max_allowed * 0.7
        dist = rng.uniform(min_allowed, max_allowed)
        cx, cy = dist * np.cos(angle), dist * np.sin(angle)
        edge_dist = DOMAIN_RADIUS - np.sqrt(cx**2 + cy**2) - max_r
        if 0.003 < edge_dist < 0.025:
            return cx, cy
    # fallback: 正常采样
    return _sample_position(rng, max_r)


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
    """双圆 (两个不重叠的圆, 最小半径0.012, 确保两个圆都可见)"""
    mask = np.zeros(len(centers_np), dtype=bool)
    positions = []
    for _ in range(2):
        for attempt in range(20):
            r = rng.uniform(0.012, 0.028)
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


def _gen_ring(rng, centers_np):
    """环形内含物 (空心圆环) — 测试边缘检测能力"""
    outer_r = rng.uniform(0.020, 0.040)
    inner_r = outer_r * rng.uniform(0.3, 0.6)  # 环宽: 40%~70% of outer
    cx, cy = _sample_position(rng, outer_r)
    dist = np.sqrt((centers_np[:, 0] - cx)**2 + (centers_np[:, 1] - cy)**2)
    mask = (dist < outer_r) & (dist > inner_r)
    return mask, cx, cy, outer_r, 'ring'


def _gen_near_boundary(rng, centers_np):
    """近边界内含物 (圆或椭圆，强制靠近桶壁) — 最难场景"""
    if rng.random() < 0.5:
        # 近边圆
        r = rng.uniform(0.008, 0.025)
        cx, cy = _sample_position_near_boundary(rng, r)
        dist = np.sqrt((centers_np[:, 0] - cx)**2 + (centers_np[:, 1] - cy)**2)
        mask = dist < r
        max_r = r
        shape_name = 'circle_near_boundary'
    else:
        # 近边椭圆
        a = rng.uniform(0.015, 0.035)
        b = rng.uniform(0.008, 0.018)
        max_r = max(a, b)
        cx, cy = _sample_position_near_boundary(rng, max_r)
        angle = rng.uniform(0, np.pi)
        dx = centers_np[:, 0] - cx
        dy = centers_np[:, 1] - cy
        rx = dx * np.cos(angle) + dy * np.sin(angle)
        ry = -dx * np.sin(angle) + dy * np.cos(angle)
        mask = (rx / a)**2 + (ry / b)**2 <= 1.0
        shape_name = 'ellipse_near_boundary'
    return mask, cx, cy, max_r, shape_name


SHAPE_GENERATORS = {
    'circle': _gen_circle,
    'ellipse': _gen_ellipse,
    'double_circle': _gen_double_circle,
    'square': _gen_square,
    'ring': _gen_ring,
    'near_boundary': _gen_near_boundary,
}


# ─── 扩散模型平滑 ───
_adj = None
_inv_deg = None
SMOOTH_ITERS = 1
SMOOTH_STRENGTH = 0.10


def _build_adjacency(elements, n_elems):
    """构建 FEM 网格邻接矩阵 (用于图 Laplacian 平滑)"""
    from scipy.sparse import coo_matrix
    rows, cols, data = [], [], []
    for i in range(n_elems):
        nbs = set()
        for tri in elements:
            if i in tri:
                for j in tri:
                    if j != i:
                        nbs.add(j)
        for nb in nbs:
            rows.append(i); cols.append(nb); data.append(1.0)
    adj = coo_matrix((data, (rows, cols)), shape=(n_elems, n_elems)).tocsr()
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    return adj, 1.0 / deg


def _apply_smooth(sigma):
    """图 Laplacian 平滑: 将硬边界变成连续过渡, 产生适合扩散模型训练的 σ 分布"""
    global _adj, _inv_deg
    s = sigma.copy().astype(np.float64)
    for _ in range(SMOOTH_ITERS):
        s = (1 - SMOOTH_STRENGTH) * s + SMOOTH_STRENGTH * (_adj.dot(s) * _inv_deg)
    return s.astype(np.float32)


def _worker_init(config_path):
    global _solver, _adj, _inv_deg
    _solver = EITForwardSolver(config_path)
    # 预构建邻接矩阵用于平滑
    _adj, _inv_deg = _build_adjacency(_solver.mesh.element, _solver.n_elems)


def _generate_one(seed, fixed_noise_db=None, bg_sigma=None, contrast_range=(3.0, 10.0)):
    """生成一个随机形状样本

    参数:
        seed: 随机种子
        fixed_noise_db: 若指定，使用固定噪声电平而非随机 (用于系统测试集)
        bg_sigma: 背景电导率, None时使用全局 BG_SIGMA (0.01)
        contrast_range: (min_contrast, max_contrast) 对比度范围
    """
    global _solver
    rng = np.random.RandomState(seed)
    centers = _solver.element_centers
    n_elems = _solver.n_elems
    if bg_sigma is None:
        bg_sigma = BG_SIGMA

    # 随机选形状类型
    shape = rng.choice(SHAPE_TYPES, p=SHAPE_WEIGHTS)
    gen_fn = SHAPE_GENERATORS[shape]
    mask, cx, cy, max_r, shape_name = gen_fn(rng, centers)

    # 随机对比度
    contrast = rng.uniform(contrast_range[0], contrast_range[1])
    inc_sigma = bg_sigma * contrast

    sigma = np.full(n_elems, bg_sigma, dtype=np.float32)
    sigma[mask] = inc_sigma

    # 扩散模型专用: 图 Laplacian 平滑 → 连续 σ 分布
    sigma = _apply_smooth(sigma)
    sigma = np.clip(sigma, bg_sigma * 0.95, inc_sigma * 1.02)

    # 求解绝对电压（非差分，模拟实际硬件采集）
    V_abs = _solver.solve_current(sigma)          # (n_meas,)
    # 复制到 n_freq 个频率（pyEIT 不区分频率）
    V = np.tile(V_abs, (len(_solver.frequencies), 1)).astype(np.float32)  # (n_freq, n_meas)
    if fixed_noise_db is not None:
        noise_db = float(fixed_noise_db)
    else:
        noise_db = rng.uniform(-40, -20)
    V_noisy = _solver.add_noise(V, noise_db)

    # 计算内含物质心到边界的距离 (用于 hard-case 分析)
    inc_centers = centers[mask]
    if len(inc_centers) > 0:
        inc_cx = float(np.mean(inc_centers[:, 0]))
        inc_cy = float(np.mean(inc_centers[:, 1]))
        edge_dist = float(DOMAIN_RADIUS - np.sqrt(inc_cx**2 + inc_cy**2))
    else:
        inc_cx, inc_cy, edge_dist = 0.0, 0.0, 1.0

    return {
        'sigma': sigma,
        'mask': mask.astype(np.float32),
        'voltage': V_noisy.astype(np.float32),
        'noise_db': noise_db,
        'shape': shape_name,
        'contrast': contrast,
        'cx': float(cx), 'cy': float(cy),
        'edge_dist': edge_dist,
        'bg_sigma': float(bg_sigma),
    }


def _generate_one_extrap(seed, fixed_noise_db=None, bg_sigma=None):
    """Generate sample with extrapolated contrast for OOD testing (contrast [6, 15])"""
    return _generate_one(seed, fixed_noise_db=fixed_noise_db, bg_sigma=bg_sigma,
                         contrast_range=(6.0, 15.0))


def generate_dataset(config_path="config/mesh_config.yaml",
                     n_train=10000, n_val=500, n_test=200,
                     output_dir="data/generated",
                     workers=0, seed=42,
                     edge_ratio=0.5, edge_threshold=0.05,
                     uniform_bg=False):
    """生成多样化数据集 (v4 多背景增强版)"""
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
    print(f"对比度: 3x ~ 10x (随机; 外推测试集: 6x ~ 15x)")
    print(f"背景电导率: {'固定 0.01' if uniform_bg else f'多档随机采样 {BACKGROUND_SIGMAS}'}")
    print(f"采样: {edge_ratio*100:.0f}%边缘/{100-edge_ratio*100:.0f}%中心 (阈值 {edge_threshold*100:.0f}cm)")
    print(f"噪声: 训练/验证随机 -40~-20dB, 测试集含固定噪声档位")

    def _gen_split(name, n, start_seed, fixed_noise_db=None, uniform_bg=False, use_extrap_contrast=False):
        label = f"{name}" + (f" (noise={fixed_noise_db}dB)" if fixed_noise_db else "")
        print(f"生成 {label} ({n} 样本)...")
        gen_fn = _generate_one_extrap if use_extrap_contrast else _generate_one
        # Sample background sigma for each sample
        bg_rng = np.random.RandomState(start_seed + 999999)
        bg_sigma_vals = []
        for _ in range(n):
            if uniform_bg:
                bg_sigma_vals.append(BG_SIGMA)
            else:
                bg_sigma_vals.append(float(bg_rng.choice(BACKGROUND_SIGMAS, p=BACKGROUND_WEIGHTS)))
        gen_args = [(s, fixed_noise_db, bg_sigma_vals[i])
                     for i, s in enumerate(range(start_seed, start_seed + n))]
        if workers > 1:
            n_proc = min(workers, cpu_count(), n)
            with Pool(n_proc, initializer=_worker_init, initargs=(config_path,)) as pool:
                results = list(tqdm(pool.starmap(gen_fn, gen_args), total=n))
        else:
            _worker_init(config_path)
            results = [gen_fn(s, fixed_noise_db=fixed_noise_db, bg_sigma=bg_sigma_vals[i])
                       for i, s in enumerate(tqdm(range(start_seed, start_seed + n)))]

        sigmas = np.stack([r['sigma'] for r in results])
        masks = np.stack([r['mask'] for r in results])
        voltages = np.stack([r['voltage'] for r in results])
        bg_sigmas = np.array([r['bg_sigma'] for r in results], dtype=np.float32)
        shapes = [r['shape'] for r in results]
        contrasts = [r['contrast'] for r in results]
        noise_dbs = [r['noise_db'] for r in results]
        edge_dists = [r['edge_dist'] for r in results]
        return sigmas, masks, voltages, bg_sigmas, shapes, contrasts, noise_dbs, edge_dists

    # 主数据集
    train_s, train_m, train_v, train_bg, train_shapes, _, _, _ = \
        _gen_split("train", n_train, seed, uniform_bg=uniform_bg)
    val_s, val_m, val_v, val_bg, val_shapes, _, _, _ = \
        _gen_split("val", n_val, seed + n_train + 1000, uniform_bg=uniform_bg)

    # 标准测试集 (随机噪声)
    test_s, test_m, test_v, test_bg, test_shapes, test_contrasts, test_noises, test_edges = \
        _gen_split("test", n_test, seed + n_train + n_val + 2000, uniform_bg=uniform_bg)

    # 固定噪声测试集 (用于系统评估)
    test_low_noise_s, test_low_noise_m, test_low_noise_v, test_low_bg, _, _, _, _ = \
        _gen_split("test_low_noise", n_test, seed + 100000, fixed_noise_db=-30, uniform_bg=uniform_bg)
    test_high_noise_s, test_high_noise_m, test_high_noise_v, test_high_bg, _, _, _, _ = \
        _gen_split("test_high_noise", n_test, seed + 200000, fixed_noise_db=-15, uniform_bg=uniform_bg)
    test_near_boundary_s, test_near_boundary_m, test_near_boundary_v, test_near_bg, _, _, _, _ = \
        _gen_split("test_near_boundary", n_test, seed + 300000, fixed_noise_db=-25, uniform_bg=uniform_bg)

    # 对比度外推测试集 (OOD测试，对比度6x~15x)
    test_extrap_s, test_extrap_m, test_extrap_v, test_extrap_bg, _, _, _, _ = \
        _gen_split("test_extrap", n_test, seed + 400000, fixed_noise_db=-25,
                   uniform_bg=uniform_bg, use_extrap_contrast=True)

    output_path = os.path.join(output_dir, "mixed_dataset.h5")
    print(f"保存到 {output_path} ...")

    with h5py.File(output_path, 'w') as f:
        # 标准 split
        for name, sigmas, masks, voltages, bg_sigmas in [
            ("train", train_s, train_m, train_v, train_bg),
            ("val", val_s, val_m, val_v, val_bg),
            ("test", test_s, test_m, test_v, test_bg),
        ]:
            grp = f.create_group(name)
            grp.create_dataset('voltages', data=voltages, compression='gzip')
            grp.create_dataset('sigmas', data=sigmas, compression='gzip')
            grp.create_dataset('masks', data=masks, compression='gzip')
            grp.create_dataset('bg_sigmas', data=bg_sigmas, compression='gzip')

        # 固定噪声测试集 及 外推测试集
        for name, sigmas, masks, voltages, bg_sigmas in [
            ("test_low_noise", test_low_noise_s, test_low_noise_m, test_low_noise_v, test_low_bg),
            ("test_high_noise", test_high_noise_s, test_high_noise_m, test_high_noise_v, test_high_bg),
            ("test_near_boundary", test_near_boundary_s, test_near_boundary_m, test_near_boundary_v, test_near_bg),
            ("test_extrap", test_extrap_s, test_extrap_m, test_extrap_v, test_extrap_bg),
        ]:
            grp = f.create_group(name)
            grp.create_dataset('voltages', data=voltages, compression='gzip')
            grp.create_dataset('sigmas', data=sigmas, compression='gzip')
            grp.create_dataset('masks', data=masks, compression='gzip')
            grp.create_dataset('bg_sigmas', data=bg_sigmas, compression='gzip')

        meta = f.create_group('metadata')
        meta.create_dataset('mesh_nodes', data=solver.mesh.node)
        meta.create_dataset('mesh_elements', data=solver.mesh.element)
        meta.create_dataset('frequencies', data=np.array(solver.frequencies))

    # 打印形状分布
    if train_shapes:
        from collections import Counter
        dist = Counter(train_shapes)
        print(f"训练集形状分布: {dict(dist)}")
        print(f"训练集形状分布 (%): { {k: f'{v/len(train_shapes)*100:.1f}%' for k, v in dist.most_common()} }")

    # 打印背景 sigma 分布
    if not uniform_bg:
        from collections import Counter
        bg_dist = Counter(train_bg.tolist())
        print(f"训练集背景sigma分布: {dict(sorted(bg_dist.items()))}")

    print(f"完成! 训练:{n_train}, 验证:{n_val}, 测试:{n_test}")
    print(f"附加测试: test_low_noise(-30dB), test_high_noise(-15dB), test_near_boundary(-25dB), test_extrap(对比度外推6x~15x)")
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
    parser.add_argument("--uniform_bg", action="store_true",
                        help="Use fixed background (0.01) instead of multi-background")
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
            uniform_bg=args.uniform_bg,
        )
