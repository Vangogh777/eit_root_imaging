"""
两阶段 EIT 训练脚本
==================
第一阶段: 传统反演生成粗略电导率
第二阶段: 神经网络精调

用法:
    python train_two_stage.py --wandb
    python train_two_stage.py --refine_type graph --hidden_dim 128
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
from models.two_stage_model import TwoStageEITModel, TraditionalReconstructor
from models.universal_eit import UniversalPhantomGenerator


def get_cache_path(n_train, n_val, n_elems):
    """获取数据缓存路径"""
    cache_dir = "data/generated"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"two_stage_data_{n_train}_{n_val}_{n_elems}.npz")


def main():
    parser = argparse.ArgumentParser(description="两阶段 EIT 训练")
    parser.add_argument("--n_train", type=int, default=20000, help="训练样本数")
    parser.add_argument("--n_val", type=int, default=1000, help="验证样本数")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--refine_type", type=str, default="unet", choices=["unet", "graph"], help="精调网络类型")
    parser.add_argument("--output", type=str, default="checkpoints/two_stage_model.pt", help="输出路径")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb 日志")
    parser.add_argument("--wandb_project", type=str, default="eit-root-imaging", help="wandb 项目名")
    parser.add_argument("--generate", action="store_true", help="强制重新生成数据")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数")
    args = parser.parse_args()

    n_workers = args.workers if args.workers else cpu_count()

    # wandb 初始化
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
                    "refine_type": args.refine_type,
                    "model": "two_stage",
                }
            )
            print(f"[wandb] 已连接: {wandb.run.url}")
        except ImportError:
            print("[WARN] wandb 未安装，跳过")
            use_wandb = False

    print("=" * 60)
    print("🔄 两阶段 EIT 训练")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    if device.type == 'cuda':
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n配置:")
    print(f"  训练样本: {args.n_train}")
    print(f"  验证样本: {args.n_val}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  精调网络: {args.refine_type}")
    print(f"  隐藏层维度: {args.hidden_dim}")

    # ============ 1. 初始化求解器 ============
    print("\n[1/6] 初始化 EIT 求解器...")
    solver = EITForwardSolver("config/mesh_config.yaml")
    n_elems = solver.n_elems
    n_meas = solver.n_measurements
    print(f"  网格: {n_elems} 单元, {n_meas} 测量通道")

    # ============ 2. 创建传统反演器 ============
    print("\n[2/6] 创建传统反演器...")
    reconstructor = TraditionalReconstructor(solver, method='jac')
    print("  使用 Gauss-Newton (JAC) 反演")

    # ============ 3. 生成/加载数据 ============
    print("\n[3/6] 准备数据集...")

    cache_path = get_cache_path(args.n_train, args.n_val, n_elems)
    use_cache = not args.generate and os.path.exists(cache_path)

    if use_cache:
        print(f"  发现缓存数据，加载中...")
        data = np.load(cache_path)
        train_coarse = data['train_coarse']
        train_target = data['train_target']
        val_coarse = data['val_coarse']
        val_target = data['val_target']
        print(f"  训练: {train_coarse.shape}, 验证: {val_coarse.shape}")
    else:
        print(f"  生成新数据...")

        # 加载已有电压数据（如果有）
        voltage_cache = f"data/generated/server_data_{args.n_train}_{args.n_val}_{n_elems}.npz"
        if os.path.exists(voltage_cache) and not args.generate:
            print(f"  从电压缓存加载: {voltage_cache}")
            vdata = np.load(voltage_cache)
            train_voltages = vdata['train_voltages']
            val_voltages = vdata['val_voltages']
            train_target = vdata['train_sigmas']
            val_target = vdata['val_sigmas']
        else:
            print("  需要先生成电压数据，运行: python train_server.py --generate")
            return

        # 第一阶段: 传统反演
        print("  执行传统反演...")
        print(f"  训练集反演...")
        train_coarse = reconstructor.batch_reconstruct(train_voltages.squeeze(1))
        print(f"  验证集反演...")
        val_coarse = reconstructor.batch_reconstruct(val_voltages.squeeze(1))

        # 保存缓存
        np.savez_compressed(
            cache_path,
            train_coarse=train_coarse,
            train_target=train_target,
            val_coarse=val_coarse,
            val_target=val_target,
        )
        print(f"  数据已缓存: {cache_path}")

    print(f"  粗略电导率范围: [{train_coarse.min():.4f}, {train_coarse.max():.4f}]")
    print(f"  目标电导率范围: [{train_target.min():.4f}, {train_target.max():.4f}]")

    # 计算初始误差（反演质量）
    init_re = np.linalg.norm(train_coarse - train_target, axis=1) / (np.linalg.norm(train_target, axis=1) + 1e-8)
    print(f"  传统反演初始 RE: {init_re.mean():.4f} ± {init_re.std():.4f}")

    # ============ 4. 构建模型 ============
    print("\n[4/6] 构建精调模型...")

    centers = np.mean(solver.mesh.node[solver.mesh.element], axis=1)
    if centers.shape[1] > 2:
        centers = centers[:, :2]
    elements = solver.mesh.element

    model = TwoStageEITModel(
        n_elems=n_elems,
        refine_type=args.refine_type,
        hidden_dim=args.hidden_dim,
    ).to(device)

    if args.refine_type == 'graph':
        model.setup_mesh(centers, elements)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ============ 5. 训练 ============
    print("\n[5/6] 开始训练...")

    train_coarse_t = torch.from_numpy(train_coarse).float().to(device)
    train_target_t = torch.from_numpy(train_target).float().to(device)
    val_coarse_t = torch.from_numpy(val_coarse).float().to(device)
    val_target_t = torch.from_numpy(val_target).float().to(device)

    best_val_re = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(args.n_train)

        epoch_loss = 0.0
        n_batches = (args.n_train + args.batch_size - 1) // args.batch_size

        for b in range(n_batches):
            idx = perm[b*args.batch_size:(b+1)*args.batch_size]
            coarse_batch = train_coarse_t[idx]
            target_batch = train_target_t[idx]

            optimizer.zero_grad()

            out = model(coarse_batch, target_batch)

            # 损失: 相对MSE
            loss = ((out['sigma'] - target_batch) ** 2 / (target_batch ** 2 + 1e-6)).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= n_batches
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_re_sum = 0.0
            n_val_batches = (args.n_val + args.batch_size - 1) // args.batch_size

            for b in range(n_val_batches):
                idx_start = b * args.batch_size
                idx_end = min((b + 1) * args.batch_size, args.n_val)
                coarse_batch = val_coarse_t[idx_start:idx_end]
                target_batch = val_target_t[idx_start:idx_end]

                val_out = model(coarse_batch, target_batch)
                val_loss_sum += ((val_out['sigma'] - target_batch) ** 2 / (target_batch ** 2 + 1e-6)).mean().item()

                re = torch.norm(val_out['sigma'] - target_batch, dim=-1) / (torch.norm(target_batch, dim=-1) + 1e-8)
                val_re_sum += re.sum().item()

            val_loss = val_loss_sum / n_val_batches
            val_re = val_re_sum / args.n_val

        history.append({'epoch': epoch, 'loss': epoch_loss, 'val_loss': val_loss, 'val_re': val_re})

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {epoch_loss:.6f} | "
              f"Val: {val_loss:.6f} | RE: {val_re:.4f} | LR: {lr:.6f}")

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": epoch_loss,
                "val/loss": val_loss,
                "val/RE": val_re,
                "train/lr": lr,
            })

        if val_re < best_val_re:
            best_val_re = val_re
            best_state = model.state_dict().copy()
            best_epoch = epoch

    model.load_state_dict(best_state)

    # ============ 6. 测试 ============
    print("\n[6/6] 测试精调效果...")
    model.eval()

    with torch.no_grad():
        out = model(val_coarse_t[:8], val_target_t[:8])

        for i in range(8):
            # 粗略误差
            coarse_re = torch.norm(val_coarse_t[i] - val_target_t[i]) / (torch.norm(val_target_t[i]) + 1e-8)
            # 精调误差
            refined_re = torch.norm(out['sigma'][i] - val_target_t[i]) / (torch.norm(val_target_t[i]) + 1e-8)
            print(f"  样本 {i}: 粗略 RE = {coarse_re.item():.4f} → 精调 RE = {refined_re.item():.4f}")

    # ============ 完成 ============
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"📊 最佳验证相对误差: {best_val_re:.4f} (Epoch {best_epoch})")
    print(f"📊 模型参数量: {total_params:,}")
    print(f"📊 精调网络: {args.refine_type}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'two_stage',
        'refine_type': args.refine_type,
        'config': {
            'n_elems': n_elems,
            'hidden_dim': args.hidden_dim,
        },
        'history': history,
        'best_val_re': best_val_re,
        'best_epoch': best_epoch,
    }, args.output)
    print(f"💾 模型已保存: {args.output}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
