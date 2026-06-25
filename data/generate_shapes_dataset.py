"""
混合形状 EIT 数据集生成（自定义形状版）
=====================================
生成多种几何形状内含物的 EIT 训练/测试数据。

形状类型 (等比例):
  - circle:        单圆 (25%)
  - ellipse:       椭圆 (25%)
  - diamond:       菱形 (25%)
  - double_circle: 双圆 (25%)

特性:
  - 背景均匀 (0.01 S/m)
  - 内含物随机位置、大小、对比度
  - 支持边缘/中心平衡采样
  - 支持预览模式（小批量 + 可视化）
"""
import os, sys, argparse, yaml
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.eit_forward import EITForwardSolver
from models.two_stage_model import TraditionalReconstructor

DOMAIN_RADIUS = 0.10
BG_SIGMA = 0.1

SHAPE_TYPES = ['circle', 'ellipse', 'diamond', 'double_circle']
SHAPE_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

# ── 大小范围约束 ─────────────────────────────────────
# 保证内含物既不太小（可辨识）也不太大（不超出边界）
SIZE_RANGES = {
    'circle':        (0.012, 0.032),   # 半径 (cm)
    'ellipse_a':     (0.018, 0.040),   # 半长轴 (cm)
    'ellipse_b':     (0.010, 0.022),   # 半短轴 (cm)
    'diamond':       (0.022, 0.035),   # 半对角线长 (cm)
    'double_circle': (0.015, 0.025),   # 每个圆的半径 (cm)，避免太小
}
MIN_SEPARATION = 0.004  # 双圆之间的最小间距 (cm)
DOMAIN_MARGIN = 0.004   # 内含物距边界的余量 (cm)


def _sample_position(rng, max_r, edge_ratio=0.5, edge_threshold=0.05):
    """在域内随机采样一个位置（确保内含物完全在桶内）"""
    while True:
        angle = rng.uniform(0, 2 * np.pi)
        max_dist = DOMAIN_RADIUS - max_r - DOMAIN_MARGIN
        lo = min(edge_threshold, max_dist * 0.95)
        if rng.random() < edge_ratio:
            dist = rng.uniform(lo, max_dist)  # 边缘
        else:
            dist = rng.uniform(0, min(0.04, max_dist))  # 中心
        cx, cy = dist * np.cos(angle), dist * np.sin(angle)
        if np.sqrt(cx**2 + cy**2) + max_r < DOMAIN_RADIUS - DOMAIN_MARGIN:
            return cx, cy


def _gen_circle(rng, centers_np, edge_ratio=0.5, edge_threshold=0.05):
    """单圆"""
    r = rng.uniform(*SIZE_RANGES['circle'])
    cx, cy = _sample_position(rng, r, edge_ratio, edge_threshold)
    dist = np.sqrt((centers_np[:, 0] - cx)**2 + (centers_np[:, 1] - cy)**2)
    return dist < r, cx, cy, r, 'circle'


def _gen_ellipse(rng, centers_np, edge_ratio=0.5, edge_threshold=0.05):
    """椭圆（随机旋转）"""
    a = rng.uniform(*SIZE_RANGES['ellipse_a'])  # 半长轴
    b = rng.uniform(*SIZE_RANGES['ellipse_b'])  # 半短轴
    max_r = max(a, b)
    cx, cy = _sample_position(rng, max_r, edge_ratio, edge_threshold)
    angle = rng.uniform(0, np.pi)
    dx = centers_np[:, 0] - cx
    dy = centers_np[:, 1] - cy
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    return (rx / a)**2 + (ry / b)**2 <= 1.0, cx, cy, max_r, 'ellipse'


def _gen_diamond(rng, centers_np, edge_ratio=0.5, edge_threshold=0.05):
    """菱形（旋转 45° 的正方形 / 等边菱形，随机旋转）"""
    half_diag = rng.uniform(*SIZE_RANGES['diamond'])  # 半对角线长
    max_r = half_diag
    cx, cy = _sample_position(rng, max_r, edge_ratio, edge_threshold)
    angle = rng.uniform(0, np.pi / 2)

    dx = centers_np[:, 0] - cx
    dy = centers_np[:, 1] - cy
    # 旋转到菱形局部坐标
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    # 菱形: |x/half_diag| + |y/half_diag| <= 1
    # 也可以是非等边菱形: |x/a| + |y/b| <= 1
    # 这里使用等边菱形（正方形旋转45°），比例可调
    scale_x = half_diag
    scale_y = half_diag * rng.uniform(0.6, 1.0)  # 略微拉伸/压缩
    mask = np.abs(rx) / scale_x + np.abs(ry) / scale_y <= 1.0
    return mask, cx, cy, max_r, 'diamond'


def _gen_double_circle(rng, centers_np, edge_ratio=0.5, edge_threshold=0.05):
    """双圆（两个不重叠的圆）"""
    mask = np.zeros(len(centers_np), dtype=bool)
    positions = []
    for _ in range(2):
        for attempt in range(30):
            r = rng.uniform(*SIZE_RANGES['double_circle'])
            cx, cy = _sample_position(rng, r, edge_ratio, edge_threshold)
            # 确保与已放置的圆不重叠
            ok = True
            for px, py, pr in positions:
                if np.sqrt((cx - px)**2 + (cy - py)**2) < pr + r + MIN_SEPARATION:
                    ok = False
                    break
            if ok:
                break
        positions.append((cx, cy, r))
        dist = np.sqrt((centers_np[:, 0] - cx)**2 + (centers_np[:, 1] - cy)**2)
        mask |= dist < r
    return mask, cx, cy, max(r for _, _, r in positions), 'double_circle'


SHAPE_GENERATORS = {
    'circle': _gen_circle,
    'ellipse': _gen_ellipse,
    'diamond': _gen_diamond,
    'double_circle': _gen_double_circle,
}


def _worker_init(config_path):
    global _solver
    _solver = EITForwardSolver(config_path)


def _generate_one(args):
    """生成一个随机形状样本"""
    seed, config_path, edge_ratio, edge_threshold = args
    global _solver
    rng = np.random.RandomState(seed)
    centers = _solver.element_centers
    n_elems = _solver.n_elems

    # 随机选形状类型
    shape = rng.choice(SHAPE_TYPES, p=SHAPE_WEIGHTS)
    gen_fn = SHAPE_GENERATORS[shape]
    mask, cx, cy, max_r, shape_name = gen_fn(rng, centers, edge_ratio, edge_threshold)

    # 固定对比度 (5x, 内含物 0.5 S/m)
    contrast = 5.0
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


def generate_dataset(config_path="config/mesh_fine_config.yaml",
                     n_train=10000, n_val=500, n_test=200,
                     output_dir="data/generated/shapes_dataset",
                     workers=0, seed=42,
                     edge_ratio=0.5, edge_threshold=0.05):
    """生成多样化形状数据集"""
    os.makedirs(output_dir, exist_ok=True)

    solver = EITForwardSolver(config_path)
    n_freq = len(solver.frequencies)
    n_meas = solver.n_measurements
    n_elems = solver.n_elems
    print(f"网格: {n_elems} 单元")
    print(f"形状: {SHAPE_TYPES} (等权重: {SHAPE_WEIGHTS})")
    print(f"大小范围: {SIZE_RANGES}")
    print(f"对比度: 固定 5x (内含物 {BG_SIGMA*5:.1f} S/m)")
    print(f"采样: {edge_ratio*100:.0f}%边缘 / {100-edge_ratio*100:.0f}%中心")
    print(f"噪声: -40 ~ -20 dB (随机)")

    import h5py

    def _gen_split(name, n, start_seed):
        print(f"生成 {name} 集 ({n} 样本)...")
        gen_args = [(s, config_path, edge_ratio, edge_threshold)
                    for s in range(start_seed, start_seed + n)]
        if workers > 1:
            n_proc = min(workers, cpu_count(), n)
            with Pool(n_proc, initializer=_worker_init, initargs=(config_path,)) as pool:
                results = list(tqdm(pool.imap(_generate_one, gen_args), total=n))
        else:
            _worker_init(config_path)
            results = [_generate_one(a) for a in tqdm(gen_args)]

        sigmas = np.stack([r['sigma'] for r in results])
        masks = np.stack([r['mask'] for r in results])
        voltages = np.stack([r['voltage'] for r in results])
        shapes = [r['shape'] for r in results]
        contrasts = [r['contrast'] for r in results]
        return sigmas, masks, voltages, shapes, contrasts

    train_s, train_m, train_v, train_shapes, _ = _gen_split(
        "train", n_train, seed)
    val_s, val_m, val_v, val_shapes, _ = _gen_split(
        "val", n_val, seed + n_train + 1000)
    test_s, test_m, test_v, test_shapes, _ = _gen_split(
        "test", n_test, seed + n_train + n_val + 2000)

    output_path = os.path.join(output_dir, "shapes_dataset.h5")
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


def generate_preview(config_path="config/mesh_fine_config.yaml",
                     output_dir="data/generated/shapes_dataset",
                     n_samples=40):
    """生成少量样本用于可视化预览（含 BP 传统重建）"""
    preview_dir = os.path.join(output_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)

    solver = EITForwardSolver(config_path)
    centers = solver.element_centers[:, :2]
    elements = solver.mesh.element
    _worker_init(config_path)

    # 初始化 BP 重建器（传统算法）
    bp_reconstructor = TraditionalReconstructor(solver, method='bp')

    rng = np.random.RandomState(42)
    samples = []
    for i in range(n_samples):
        seed = 42 + i
        gen_args = (seed, config_path, 0.5, 0.05)
        result = _generate_one(gen_args)
        # BP 重建：差分电压 → 加回均匀场 → 传统反演
        v_diff = result['voltage'][0]  # (208,) 差分电压
        v_abs = v_diff + solver.V_uniform  # 绝对电压
        sigma_bp = bp_reconstructor.reconstruct(v_abs)
        result['sigma_bp'] = sigma_bp.astype(np.float32)
        samples.append(result)

    # 保存为 npz
    np.savez(os.path.join(preview_dir, "preview_samples.npz"),
             sigmas=np.stack([s['sigma'] for s in samples]),
             sigmas_bp=np.stack([s['sigma_bp'] for s in samples]),
             masks=np.stack([s['mask'] for s in samples]),
             shapes=np.array([s['shape'] for s in samples]),
             contrasts=np.array([s['contrast'] for s in samples]),
             centers=centers, elements=elements)

    # 生成可视化图片
    _visualize_preview(samples, centers, elements, preview_dir)

    return samples


def _visualize_preview(samples, centers, elements, output_dir):
    """生成 FEM 电导率分布 + 电压曲线的可视化预览图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return

    global _solver
    try:
        nodes = _solver.mesh.node
    except Exception:
        print("无法获取网格节点坐标，跳过可视化")
        return

    n_samples = len(samples)
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols

    triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    vmin, vmax = BG_SIGMA, BG_SIGMA * 6   # 0.1~0.6 S/m (5x + 余量)
    cmap_sigma = 'viridis'

    # ── 总览大图: sigma map + colorbar ──
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = axes.flatten()
    last_im = None
    for i in range(n_samples):
        ax = axes[i]
        sigma = samples[i]['sigma']
        shape_name = samples[i]['shape']
        contrast = samples[i]['contrast']
        im = ax.tripcolor(triang, facecolors=sigma, cmap=cmap_sigma,
                          vmin=vmin, vmax=vmax, shading='flat')
        last_im = im
        ax.set_xlim(-0.105, 0.105)
        ax.set_ylim(-0.105, 0.105)
        ax.set_aspect('equal')
        ax.set_title(f"{shape_name} c={contrast:.1f}x", fontsize=7)
        ax.axis('off')
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    # 全局 colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label('Conductivity (S/m)', fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    save_path = os.path.join(output_dir, "preview_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"预览总览图: {save_path}")

    # ── 每类形状: sigma_gt (左) | sigma_bp (中) | voltage (右) ──
    for shape in SHAPE_TYPES:
        indices = [i for i, s in enumerate(samples) if s['shape'] == shape]
        if not indices:
            continue

        n_sub = len(indices)
        n_cols_sub = min(3, n_sub)  # 每行3组
        n_rows_sub = (n_sub + n_cols_sub - 1) // n_cols_sub

        # 每个样本占 1 行 3 列: gt | bp | voltage
        fig2, axes2 = plt.subplots(n_rows_sub, n_cols_sub * 3,
                                   figsize=(n_cols_sub * 7.5, n_rows_sub * 2.5))
        axes2 = axes2.flatten()

        for j, idx in enumerate(indices):
            col_gt  = j * 3
            col_bp  = j * 3 + 1
            col_vol = j * 3 + 2

            sigma = samples[idx]['sigma']
            sigma_bp = samples[idx]['sigma_bp']
            voltage_diff = samples[idx]['voltage'].flatten()
            voltage_abs = voltage_diff + _solver.V_uniform
            contrast = samples[idx]['contrast']

            # 左: ground truth sigma
            ax_s = axes2[col_gt]
            im = ax_s.tripcolor(triang, facecolors=sigma, cmap=cmap_sigma,
                                vmin=vmin, vmax=vmax, shading='flat')
            ax_s.set_xlim(-0.105, 0.105)
            ax_s.set_ylim(-0.105, 0.105)
            ax_s.set_aspect('equal')
            ax_s.set_title(f"GT σ  c={contrast:.1f}x", fontsize=7)
            ax_s.axis('off')
            plt.colorbar(im, ax=ax_s, fraction=0.08, pad=0.04)

            # 中: BP reconstruction
            ax_bp = axes2[col_bp]
            im_bp = ax_bp.tripcolor(triang, facecolors=sigma_bp, cmap=cmap_sigma,
                                    vmin=vmin, vmax=vmax, shading='flat')
            ax_bp.set_xlim(-0.105, 0.105)
            ax_bp.set_ylim(-0.105, 0.105)
            ax_bp.set_aspect('equal')
            ax_bp.set_title(f"BP recon", fontsize=7)
            ax_bp.axis('off')
            plt.colorbar(im_bp, ax=ax_bp, fraction=0.08, pad=0.04)

            # 右: voltage curve (U型 - 绝对电压)
            ax_v = axes2[col_vol]
            ax_v.plot(voltage_abs, 'b-', linewidth=0.8, alpha=0.7)
            n_meas_per_exc = 13
            for e in range(1, 16):
                x = e * n_meas_per_exc
                ax_v.axvline(x, color='gray', linestyle='--', linewidth=0.3, alpha=0.4)
            ax_v.set_xlim(0, 208)
            ax_v.set_title(f"Absolute Voltage", fontsize=7)
            ax_v.set_xlabel('Measurement', fontsize=6)
            ax_v.set_ylabel('V', fontsize=6)
            ax_v.tick_params(labelsize=5)
            ax_v.grid(True, alpha=0.15)

        # 隐藏多余子图
        for j in range(n_sub * 2, len(axes2)):
            axes2[j].axis('off')

        plt.tight_layout()
        save_path2 = os.path.join(output_dir, f"preview_{shape}.png")
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"形状 '{shape}' 预览图: {save_path2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="生成混合形状 EIT 数据集（圆/椭圆/菱形/双圆）")
    parser.add_argument("--config", default="config/mesh_fine_config.yaml")
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--output", default="data/generated/shapes_dataset")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge_ratio", type=float, default=0.5,
                        help="边缘采样比例 (0-1)")
    parser.add_argument("--edge_threshold", type=float, default=0.05,
                        help="边缘/中心分界阈值 (m)")
    parser.add_argument("--preview", action="store_true",
                        help="只生成预览样本 (默认40个)")
    parser.add_argument("--preview_n", type=int, default=40,
                        help="预览样本数量")
    args = parser.parse_args()

    if args.preview:
        print(f"生成预览样本 ({args.preview_n} 个)...")
        print(f"配置: {args.config}")
        samples = generate_preview(args.config, args.output, args.preview_n)
        print(f"生成了 {len(samples)} 个预览样本")
        print(f"预览文件: {args.output}/preview/")
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
