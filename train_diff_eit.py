"""
DiffEIT v4 训练脚本
====================
单阶段条件扩散模型训练 (标准化 DDPM + 物理一致性).

v4 改进 (2026-06-26):
  - 移除阶段切换 (单阶段条件训练)
  - 移除 warm-start (架构简化)
  - 标准化扩散空间 (N(0,1) 替代原始 σ 空间)
  - 两阶段归一化: [0,1] → N(0,1)
  - 加权 MSE 损失 (异常区域权重更高)
  - 物理一致性损失 (via Jacobian)

用法:
    python train_diff_eit.py                          # 默认训练
    python train_diff_eit.py --epochs 200             # 自定义 epoch 数
    python train_diff_eit.py --resume <ckpt>          # 恢复训练
"""
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.eit_forward import EITForwardSolver
from data.datasets.eit_dataset import MemoryEITDataset
from models.diff_eit import DiffEIT
from models.diffusion_utils import DiffusionProcess
from models.mesh_pooling import build_hierarchy
from training.recorder import TrainingRecorder


def simple_mse_loss(pred, target):
    """Simple MSE — all elements equal weight. Smooth data doesn't need anomaly weighting."""
    return F.mse_loss(pred, target)


def physics_consistency_loss(sigma_pred_std, sigma_mean, sigma_std, sigma_min, sigma_range,
                              V_measured, J_T, sigma_ref):
    """Measurement consistency: ||V - J·(σ_pred - σ_ref)||² / ||V||² (normalized)"""
    # Inverse transform: standardized -> physical
    sigma_pred_norm = sigma_pred_std.squeeze(-1) * sigma_std + sigma_mean
    sigma_pred_phys = sigma_pred_norm.clamp(0, 1) * sigma_range + sigma_min
    delta_sigma = sigma_pred_phys - sigma_ref
    V_pred = delta_sigma @ J_T  # (B, n_elems) @ (n_elems, 208) = (B, 208)
    # Use all frequencies: take mean across freq dim if multi-frequency
    if V_measured.dim() == 3:
        V_measured_mean = V_measured.mean(dim=1)  # (B, 208) average over frequencies
    else:
        V_measured_mean = V_measured
    # Normalize by measured voltage magnitude for scale invariance
    v_norm = V_measured_mean.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return F.mse_loss(V_pred / v_norm, V_measured_mean / v_norm)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_config', default='config/mesh_config.yaml')
    parser.add_argument('--data_path', default='data/generated/mixed_dataset.h5')
    parser.add_argument('--jacobian_path', default='data/generated/jacobian.npy')
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--schedule', choices=['linear', 'cosine'], default='cosine')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    if device.type == 'cuda':
        print(f'  {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

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

    # ========== 3.5/6. 计算 sigma 统计量 ==========
    print('\n[3.5/6] Computing sigma statistics...')
    all_sigmas = []
    for batch in train_loader:
        all_sigmas.append(batch['sigmas'].numpy().ravel())
    all_sigmas = np.concatenate(all_sigmas)
    print(f'  sigma: [{all_sigmas.min():.4f}, {all_sigmas.max():.4f}], {len(all_sigmas):,} values')

    # RankGauss: build lookup table forcing data to N(0,1)
    from scipy.special import erfinv
    n_quantiles = min(10000, len(all_sigmas))
    sorted_vals = np.sort(np.random.choice(all_sigmas, size=n_quantiles, replace=False))
    quantiles = np.linspace(0, 1, n_quantiles)
    quantiles_clipped = np.clip(quantiles, 1e-7, 1 - 1e-7)
    gauss_vals = np.sqrt(2.0) * erfinv(2.0 * quantiles_clipped - 1.0)
    print(f'  RankGauss: {n_quantiles} quantiles, gauss range [{gauss_vals[0]:.2f}, {gauss_vals[-1]:.2f}]')
    print(f'    → data is now truly N(0,1) in diffusion space')

    # ========== 4. 创建模型 ==========
    print('\n[4/6] 创建模型 (v4: standardized DDPM + physics consistency)...')
    model = DiffEIT(
        n_elems=n_elems,
        n_meas=208,
        hidden_dim=args.hidden_dim,
        T=args.T,
        schedule=args.schedule,
    )
    model.configure_rankgauss(sorted_vals, gauss_vals)
    model.setup_mesh(centers, elements, jacobian, hierarchy, sigma_ref=0.01)
    model.to(device)

    # torch.compile for ~20-30% speedup (only if PyTorch >= 2.0)
    if hasattr(torch, 'compile'):
        try:
            model.denoiser = torch.compile(model.denoiser, mode='reduce-overhead')
            print(f'  torch.compile: enabled (reduce-overhead)')
        except Exception as e:
            print(f'  torch.compile: failed ({e}), skipping')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  参数量: {n_params:,}')
    print(f'  噪声调度: {args.schedule}')

    # ========== 5. 训练准备 ==========
    recorder = TrainingRecorder(name=f'diff_eit_v4_{args.schedule}_hd{args.hidden_dim}')
    recorder.save_meta({
        'version': 'v4',
        'n_elems': n_elems, 'hidden_dim': args.hidden_dim, 'T': args.T,
        'n_params': n_params, 'lr': args.lr,
        'schedule': args.schedule,
        'model_type': 'x0_prediction_std',
    })

    ckpt_dir = f'checkpoints/{recorder.run_id}'
    os.makedirs(ckpt_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best_val_loss = float('inf')

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(f'  从 epoch {start_epoch} 恢复')

    total_epochs = args.epochs
    sigma_ref = float(sorted_vals[0])  # background conductivity for physics reference

    print(f'\n[5/6] 开始训练 ({total_epochs} epochs)...')
    print(f'  x₀-prediction 目标 (N(0,1) 空间), {args.schedule} 噪声调度')
    print(f'  损失: Simple MSE + λ=0.3 物理一致性 (smooth data)')

    for epoch in range(start_epoch + 1, total_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_loss_mse = 0.0
        epoch_loss_phys = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}'):
            sigma_0 = batch['sigmas'].to(device)
            V = batch['voltages'].to(device)

            t = torch.randint(0, args.T, (sigma_0.shape[0],), device=device)

            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                sigma_0_pred, sigma_0_true = model(sigma_0, t, V)

                # Simple MSE in standardized space (RankGauss: true N(0,1))
                loss_mse = simple_mse_loss(sigma_0_pred, sigma_0_true)

                # Physics consistency (RankGauss inverse → physical → Jacobian)
                sigma_pred_phys = model._std_to_phys(sigma_0_pred.squeeze(-1))
                delta_sigma = sigma_pred_phys - sigma_ref
                V_pred = delta_sigma @ model.J_T
                if V.dim() == 3:
                    V_mean = V.mean(dim=1)
                else:
                    V_mean = V
                v_norm = V_mean.norm(dim=1, keepdim=True).clamp(min=1e-8)
                loss_phys = F.mse_loss(V_pred / v_norm, V_mean / v_norm)

                loss = loss_mse + 0.3 * loss_phys

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            epoch_loss_mse += loss_mse.item()
            epoch_loss_phys += loss_phys.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        avg_mse = epoch_loss_mse / max(len(train_loader), 1)
        avg_phys = epoch_loss_phys / max(len(train_loader), 1)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sigma_0 = batch['sigmas'].to(device)
                V = batch['voltages'].to(device)

                t = torch.randint(0, args.T, (sigma_0.shape[0],), device=device)

                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    sigma_0_pred, sigma_0_true = model(sigma_0, t, V)
                    val_loss += simple_mse_loss(sigma_0_pred, sigma_0_true).item()

        val_loss /= max(len(val_loader), 1)
        lr_now = optimizer.param_groups[0]['lr']

        # val_RE monitoring every 10 epochs (compute BEFORE logging)
        val_re = None
        if epoch % 10 == 0 and epoch > 0:
            model.eval()
            val_re_sum = 0.0
            n_val_re = min(5, len(val_loader))
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= n_val_re:
                        break
                    V_val = batch['voltages'][0].to(device)
                    sigma_gt = batch['sigmas'][0].to(device)
                    sigma_pred = model.sample(V_val, n_steps=50, cfg_scale=2.0)
                    re = torch.norm(sigma_pred - sigma_gt) / (torch.norm(sigma_gt) + 1e-8)
                    val_re_sum += re.item()
            val_re = val_re_sum / n_val_re
            recorder.log_event('val_RE', epoch=epoch, val_RE=val_re)
            print(f'  Val RE (DDIM 50steps): {val_re:.4f}')
            model.train()

        print(f'  Epoch {epoch:3d} | Loss: {avg_loss:.6f} (MSE: {avg_mse:.6f} Phys: {avg_phys:.6f}) | Val: {val_loss:.6f} | LR: {lr_now:.2e}')
        epoch_kwargs = dict(phase='diffusion', epoch=epoch, loss=avg_loss,
                           val_loss=val_loss, lr=lr_now)
        if val_re is not None:
            epoch_kwargs['re'] = val_re
        recorder.log_epoch(**epoch_kwargs)

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
                'sigma_min': model.sigma_min,
                'sigma_max': model.sigma_max,
                'sigma_mean': model.sigma_mean,
                'sigma_std': model.sigma_std,
                'run_id': recorder.run_id,
                'version': 'v4',
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
                'sigma_min': model.sigma_min,
                'sigma_max': model.sigma_max,
                'sigma_mean': model.sigma_mean,
                'sigma_std': model.sigma_std,
                'T': args.T,
                'run_id': recorder.run_id,
                'version': 'v4',
            }, f'{ckpt_dir}/epoch{epoch}.pt')
            recorder.log_event("checkpoint_saved", epoch=epoch)

            model.train()

    # 最终保存
    torch.save({
        'model': model.state_dict(),
        'epoch': total_epochs,
        'sigma_min': model.sigma_min,
        'sigma_max': model.sigma_max,
        'sigma_mean': model.sigma_mean,
        'sigma_std': model.sigma_std,
        'T': args.T,
        'run_id': recorder.run_id,
        'version': 'v4',
    }, f'{ckpt_dir}/final.pt')
    recorder.log_event("training_completed", epoch=total_epochs)

    print(f'\n✅ 训练完成! 模型保存到 {ckpt_dir}/')
    recorder.set_status('completed')


if __name__ == '__main__':
    train()
