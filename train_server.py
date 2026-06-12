"""
服务器训练脚本
==============
完整训练配置，用于GPU服务器

用法:
    python train_server.py                    # 默认配置 (20000样本)
    python train_server.py --n_train 50000    # 自定义样本数
    python train_server.py --model physics    # 使用物理信息增强模型
    python train_server.py --generate         # 强制重新生成数据
    python train_server.py --workers 8        # 使用8进程并行生成数据
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.eit_forward import EITForwardSolver
from models.simple_model import SimpleSFSBLC
from models.universal_eit import PhysicsInformedEIT, UniversalPhantomGenerator
from models.eit_gnn_model import EITModelGNN, EITModelSimple

def get_cache_path(n_train, n_val, n_elems):
    """获取数据缓存路径"""
    cache_dir = "data/generated"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"server_data_{n_train}_{n_val}_{n_elems}.npz")

def check_cached_data(cache_path):
    """检查缓存数据是否存在"""
    return os.path.exists(cache_path)

def load_cached_data(cache_path):
    """加载缓存数据"""
    data = np.load(cache_path)
    return (
        data['train_sigmas'],
        data['train_voltages'],
        data['val_sigmas'],
        data['val_voltages'],
    )

def save_cached_data(cache_path, train_sigmas, train_voltages, val_sigmas, val_voltages):
    """保存数据到缓存"""
    np.savez_compressed(
        cache_path,
        train_sigmas=train_sigmas,
        train_voltages=train_voltages,
        val_sigmas=val_sigmas,
        val_voltages=val_voltages,
    )
    print(f"  数据已缓存: {cache_path}")


# ============ 多进程数据生成 ============

# 全局变量（用于多进程共享，避免重复初始化）
_solver = None
_phantom_gen = None
_n_freq = None
_n_meas = None

def init_worker(config_path):
    """初始化worker进程"""
    global _solver, _phantom_gen, _n_freq, _n_meas
    _solver = EITForwardSolver(config_path)
    _n_freq = len(_solver.frequencies)
    _n_meas = _solver.n_measurements
    _phantom_gen = UniversalPhantomGenerator(
        _solver.mesh.node,
        _solver.mesh.element,
        domain_radius=_solver.cfg['mesh']['radius'],
        sigma_background=0.01,
        sigma_inclusion=0.05
    )

def generate_single_sample(seed):
    """生成单个样本（用于多进程）"""
    global _solver, _phantom_gen, _n_freq, _n_meas

    sigma = _phantom_gen.generate_random(seed=seed)
    V = _solver.solve_multi_frequency(sigma)

    if np.isnan(V).any():
        V = np.random.randn(_n_freq, _n_meas).astype(np.float32) * 1e-6

    V_noisy = _solver.add_noise(V, noise_db=np.random.uniform(-40, -20))
    return sigma.astype(np.float32), V_noisy.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="服务器训练脚本")
    parser.add_argument("--n_train", type=int, default=20000, help="训练样本数")
    parser.add_argument("--n_val", type=int, default=1000, help="验证样本数")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="隐藏层维度")
    parser.add_argument("--model", type=str, default="gnn", choices=["simple", "physics", "gnn"], help="模型类型")
    parser.add_argument("--output", type=str, default="checkpoints/server_model.pt", help="输出路径")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb 日志")
    parser.add_argument("--wandb_project", type=str, default="eit-root-imaging", help="wandb 项目名")
    parser.add_argument("--generate", action="store_true", help="强制重新生成数据")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数（默认CPU核心数）")
    args = parser.parse_args()

    # 设置并行进程数
    n_workers = args.workers if args.workers else cpu_count()

    # ============ wandb 初始化 ============
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config={
                    "n_train": args.n_train,
                    "n_val": args.n_val,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "hidden_dim": args.hidden_dim,
                    "model": args.model,
                }
            )
            print(f"[wandb] 已连接: {wandb.run.url}")
        except ImportError:
            print("[WARN] wandb 未安装，跳过。运行: pip install wandb")
            use_wandb = False

    print("=" * 60)
    print("🖥️  服务器训练 (高分辨率版)")
    print("=" * 60)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    if device.type == 'cuda':
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ============ 配置 ============
    print(f"\n配置:")
    print(f"  训练样本: {args.n_train}")
    print(f"  验证样本: {args.n_val}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  隐藏层维度: {args.hidden_dim}")
    print(f"  模型: {args.model}")

    # ============ 1. 初始化求解器 ============
    print("\n[1/5] 初始化 EIT 求解器...")
    solver = EITForwardSolver("config/mesh_config.yaml")
    n_elems = solver.n_elems
    n_freq = len(solver.frequencies)
    n_meas = solver.n_measurements
    print(f"  网格: {n_elems} 单元, {n_freq} 频率, {n_meas} 测量通道")
    print(f"  ⚠️  目标单元数: ~8000 (当前: {n_elems})")

    # ============ 2. 生成数据 ============
    print("\n[2/5] 生成数据集...")

    # 检查缓存
    cache_path = get_cache_path(args.n_train, args.n_val, n_elems)
    use_cache = not args.generate and check_cached_data(cache_path)

    if use_cache:
        print(f"  发现缓存数据，加载中...")
        train_sigmas, train_voltages, val_sigmas, val_voltages = load_cached_data(cache_path)
        print(f"  训练: {train_sigmas.shape}, 验证: {val_sigmas.shape}")
    else:
        if args.generate:
            print(f"  强制重新生成数据...")
        else:
            print(f"  未找到缓存，生成新数据...")

        print(f"  使用 {n_workers} 个进程并行生成...")

        # 使用多进程并行生成
        config_path = "config/mesh_config.yaml"

        with Pool(n_workers, initializer=init_worker, initargs=(config_path,)) as pool:
            # 生成训练数据
            print(f"  生成 {args.n_train} 训练样本...")
            train_seeds = list(range(args.n_train))
            results = list(tqdm(pool.imap(generate_single_sample, train_seeds),
                                total=args.n_train, desc="  训练数据"))
            train_sigmas = np.stack([r[0] for r in results])
            train_voltages = np.stack([r[1] for r in results])

            # 生成验证数据
            print(f"  生成 {args.n_val} 验证样本...")
            val_seeds = list(range(args.n_train, args.n_train + args.n_val))
            results = list(tqdm(pool.imap(generate_single_sample, val_seeds),
                                total=args.n_val, desc="  验证数据"))
            val_sigmas = np.stack([r[0] for r in results])
            val_voltages = np.stack([r[1] for r in results])

        # 保存缓存
        save_cached_data(cache_path, train_sigmas, train_voltages, val_sigmas, val_voltages)

    print(f"  电导率范围: [{train_sigmas.min():.4f}, {train_sigmas.max():.4f}]")

    # ============ 3. 构建模型 ============
    print("\n[3/5] 构建模型...")

    # 获取网格信息（GNN模型需要）
    centers = np.mean(solver.mesh.node[solver.mesh.element], axis=1)
    elements = solver.mesh.element

    if args.model == "simple":
        model = SimpleSFSBLC(
            input_dim=n_meas,
            hidden_dim=args.hidden_dim,
            n_frequencies=n_freq,
            n_elems=n_elems,
        ).to(device)
    elif args.model == "gnn":
        model = EITModelGNN(
            input_dim=n_meas,
            n_frequencies=n_freq,
            n_elems=n_elems,
            hidden_dim=args.hidden_dim,
            n_res_blocks=6,
            gnn_layers=4,
            gnn_heads=4,
            use_jacobian=True,
        ).to(device)
        # 设置网格结构
        model.setup_mesh(centers, elements)
        print(f"  GNN 网格已设置: {n_elems} 单元")
    else:  # physics
        model = PhysicsInformedEIT(
            input_dim=n_meas,
            hidden_dim=args.hidden_dim,
            n_frequencies=n_freq,
            n_elems=n_elems,
            use_jacobian_prior=True,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ============ 4. 训练 ============
    print("\n[4/5] 开始训练...")

    train_V = torch.from_numpy(train_voltages).float().to(device)
    train_S = torch.from_numpy(train_sigmas).float().to(device)
    val_V = torch.from_numpy(val_voltages).float().to(device)
    val_S = torch.from_numpy(val_sigmas).float().to(device)

    best_val_re = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(args.n_train)

        epoch_loss = 0.0
        n_batches = (args.n_train + args.batch_size - 1) // args.batch_size

        for b in range(n_batches):
            idx = perm[b*args.batch_size:(b+1)*args.batch_size]
            V_batch = train_V[idx]
            S_batch = train_S[idx]

            optimizer.zero_grad()

            out = model(V_batch)
            loss = torch.nn.functional.mse_loss(out['sigma'], S_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= n_batches
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_out = model(val_V)

            val_loss = torch.nn.functional.mse_loss(val_out['sigma'], val_S).item()
            re = torch.norm(val_out['sigma'] - val_S, dim=-1) / (torch.norm(val_S, dim=-1) + 1e-8)
            val_re = re.mean().item()

        history.append({'epoch': epoch, 'loss': epoch_loss, 'val_loss': val_loss, 'val_re': val_re})

        # 打印
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {epoch_loss:.6f} | "
              f"Val: {val_loss:.6f} | RE: {val_re:.4f} | LR: {lr:.6f}")

        # wandb 日志
        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": epoch_loss,
                "val/loss": val_loss,
                "val/RE": val_re,
                "train/lr": lr,
            })

        # 保存最佳模型
        if val_re < best_val_re:
            best_val_re = val_re
            best_state = model.state_dict().copy()
            best_epoch = epoch

    # 加载最佳模型
    model.load_state_dict(best_state)

    # ============ 5. 测试 ============
    print("\n[5/5] 测试重建质量...")
    model.eval()

    with torch.no_grad():
        out = model(val_V[:8])

        for i in range(8):
            re_i = torch.norm(out['sigma'][i] - val_S[i]) / (torch.norm(val_S[i]) + 1e-8)
            print(f"  样本 {i}: RE = {re_i.item():.4f}")

    # ============ 完成 ============
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"📊 最佳验证相对误差: {best_val_re:.4f} (Epoch {best_epoch})")
    print(f"📊 模型参数量: {total_params:,}")
    print(f"📊 网格单元数: {n_elems}")

    # 保存模型
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model,
        'config': {
            'n_elems': n_elems,
            'n_freq': n_freq,
            'n_meas': n_meas,
            'hidden_dim': args.hidden_dim,
        },
        'history': history,
        'best_val_re': best_val_re,
        'best_epoch': best_epoch,
    }, args.output)
    print(f"💾 模型已保存: {args.output}")

    # wandb 结束
    if use_wandb:
        import wandb
        wandb.finish()
        print("[wandb] 日志已同步")

if __name__ == "__main__":
    main()
