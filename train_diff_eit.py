"""
DiffEIT v3 训练脚本
====================
训练改进版扩散模型用于 EIT 重建。

Phase 1: 无条件预训练 (100 epoch) — 只需 σ 样本，不需配对 V
Phase 2: 条件微调 (150 epoch) — 使用 V-σ 对, x₀-prediction + 灵敏度特征 + warm-start

v3 改进:
  - 余弦噪声调度 (默认)
  - x₀-prediction 训练目标
  - 灵敏度特征 (J^T·V, J_energy)
  - 线性 warm-start 支持
  - 全层 FiLM 条件注入

用法:
    python train_diff_eit.py                          # 完整训练
    python train_diff_eit.py --phase unconditional    # 仅无条件
    python train_diff_eit.py --phase conditional      # 仅条件微调
    python train_diff_eit.py --resume <ckpt>          # 恢复训练
    python train_diff_eit.py --warm_start             # 启用 warm-start 残差扩散
"""
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.eit_forward import EITForwardSolver
from data.datasets.eit_dataset import MemoryEITDataset
from models.diff_eit import DiffEIT
from models.diffusion_utils import DiffusionProcess
from models.mesh_pooling import build_hierarchy
from training.recorder import TrainingRecorder


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['unconditional', 'conditional', 'both'],
                        default='both')
    parser.add_argument('--mesh_config', default='config/mesh_config.yaml')
    parser.add_argument('--data_path', default='data/generated/mixed_dataset.h5')
    parser.add_argument('--jacobian_path', default='data/generated/jacobian.npy')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs_uncond', type=int, default=100)
    parser.add_argument('--epochs_cond', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--schedule', choices=['linear', 'cosine'], default='cosine')
    parser.add_argument('--warm_start', action='store_true',
                        help='使用线性最小二乘 warm-start (残差扩散)')
    parser.add_argument('--no_warm_start', action='store_true',
                        help='强制不使用 warm-start')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    use_warm_start = args.warm_start and not args.no_warm_start

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    if device.type == 'cuda':
        print(f'  {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
        torch.set_float32_matmul_precision('high')

    # ========== 1. 加载网格和 Jacobian ==========
    print('\n[1/6] 加载网格和 Jacobian...')
    solver = EITForwardSolver(args.mesh_config)
    centers = solver.element_centers[:, :2].copy()
    elements = solver.mesh.element.copy()
    n_elems = solver.n_elems
    print(f'  网格: {n_elems} 元素')

    # 加载 Jacobian
    try:
        jacobian = np.load(args.jacobian_path)
        print(f'  Jacobian: {jacobian.shape}')
    except FileNotFoundError:
        print(f'  ⚠️ Jacobian 未找到: {args.jacobian_path}')
        print(f'  请先运行: python data/precompute_jacobian.py')
        sys.exit(1)

    # ========== 2. 预计算层次图 ==========
    print('\n[2/6] 构建多尺度图层次...')
    t0 = time.time()
    hierarchy = build_hierarchy(centers, elements, n_levels=3, k_neighbors=8)
    for i, h in enumerate(hierarchy):
        print(f'  Level {i}: {h["nodes"]:5d} nodes, {h["edges"].shape[1]:6d} edges')
    print(f'  耗时: {time.time()-t0:.1f}s')

    # ========== 3. 加载数据 ==========
    print('\n[3/6] 加载数据...')
    try:
        train_ds = MemoryEITDataset(args.data_path, split='train', voltage_mask_ratio=0.0)
        val_ds = MemoryEITDataset(args.data_path, split='val', voltage_mask_ratio=0.0)
    except Exception as e:
        print(f'  加载失败: {e}')
        print(f'  提示: 请先运行 python data/generate_mixed_dataset.py 生成数据')
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f'  训练: {len(train_ds)}, 验证: {len(val_ds)}')

    # ========== 4. 创建模型 ==========
    print('\n[4/6] 创建模型 (v3: FiLM + sensitivity + cosine schedule)...')
    model = DiffEIT(
        n_elems=n_elems,
        n_meas=208,
        hidden_dim=args.hidden_dim,
        time_dim=256,
        voltage_dim=512,
        pos_dim=35,
        T=args.T,
        n_levels=3,
        dropout=0.1,
        sigma_min=0.005,
        sigma_max=0.1,
        schedule=args.schedule,
    )
    model.setup_mesh(centers, elements, jacobian, hierarchy, sigma_ref=0.01)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  参数量: {n_params:,}')
    print(f'  噪声调度: {args.schedule}')
    print(f'  Warm-start: {"启用 (残差扩散)" if use_warm_start else "禁用 (全量扩散)"}')

    # ========== 5. 训练准备 ==========
    phase_label = f'{"ws" if use_warm_start else "full"}_{args.schedule}_hd{args.hidden_dim}'
    recorder = TrainingRecorder(name=f'diff_eit_v3_{args.phase}_{phase_label}')
    recorder.save_meta({
        'version': 'v3',
        'n_elems': n_elems, 'hidden_dim': args.hidden_dim, 'T': args.T,
        'n_params': n_params, 'phase': args.phase, 'lr': args.lr,
        'schedule': args.schedule, 'warm_start': use_warm_start,
        'model_type': 'x0_prediction',
    })

    ckpt_dir = f'checkpoints/{recorder.run_id}'
    os.makedirs(ckpt_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_uncond + args.epochs_cond)

    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    best_val_loss = float('inf')

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(f'  从 epoch {start_epoch} 恢复')

    total_epochs = args.epochs_uncond if args.phase != 'conditional' else args.epochs_cond
    if args.phase == 'both':
        total_epochs = args.epochs_uncond + args.epochs_cond
    use_voltage = (args.phase == 'conditional')

    print(f'\n[5/6] 开始训练 ({total_epochs} epochs)...')
    print(f'  条件模式: {use_voltage}')
    print(f'  x₀-prediction 目标, {args.schedule} 噪声调度')

    for epoch in range(start_epoch + 1, total_epochs + 1):
        # Phase 切换
        if args.phase == 'both' and epoch == args.epochs_uncond + 1:
            print('\n=== Phase 2: 条件微调 (x₀-prediction) ===')
            use_voltage = True
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs_cond)

        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}'):
            sigma_0 = batch['sigmas'].to(device)
            V = batch['voltages'].to(device) if use_voltage else None

            t = torch.randint(0, args.T, (sigma_0.shape[0],), device=device)

            # Warm-start
            sigma_warm = None
            if use_warm_start and V is not None:
                with torch.no_grad():
                    sigma_warm = model.compute_warm_start(V)

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                sigma_0_pred, sigma_0_true = model(sigma_0, t, V, sigma_warm=sigma_warm)
                loss = criterion(sigma_0_pred, sigma_0_true)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sigma_0 = batch['sigmas'].to(device)
                V = batch['voltages'].to(device) if use_voltage else None

                t = torch.randint(0, args.T, (sigma_0.shape[0],), device=device)

                sigma_warm = None
                if use_warm_start and V is not None:
                    sigma_warm = model.compute_warm_start(V)

                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    sigma_0_pred, sigma_0_true = model(sigma_0, t, V, sigma_warm=sigma_warm)
                    val_loss += criterion(sigma_0_pred, sigma_0_true).item()

        val_loss /= max(len(val_loader), 1)
        lr_now = optimizer.param_groups[0]['lr']

        print(f'  Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}')
        recorder.log_epoch(phase='diffusion', epoch=epoch, loss=avg_loss,
                          val_loss=val_loss, lr=lr_now)

        # 保存最佳
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'n_elems': n_elems,
                'hidden_dim': args.hidden_dim,
                'T': args.T,
                'schedule': args.schedule,
                'warm_start': use_warm_start,
                'run_id': recorder.run_id,
                'version': 'v3',
            }, f'{ckpt_dir}/best.pt')
            recorder.log_event("best_model_saved", epoch=epoch, val_loss=val_loss)
            print(f'    → 保存最佳模型 (loss={val_loss:.6f})')

        # 定期保存
        if epoch % 50 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'run_id': recorder.run_id,
                'version': 'v3',
            }, f'{ckpt_dir}/epoch{epoch}.pt')
            recorder.log_event("checkpoint_saved", epoch=epoch)

    # 最终保存
    torch.save({
        'model': model.state_dict(),
        'epoch': total_epochs,
        'run_id': recorder.run_id,
        'version': 'v3',
    }, f'{ckpt_dir}/final.pt')
    recorder.log_event("training_completed", epoch=total_epochs)

    print(f'\n✅ 训练完成! 模型保存到 {ckpt_dir}/')
    recorder.set_status('completed')


if __name__ == '__main__':
    train()
