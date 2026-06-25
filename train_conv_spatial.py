"""
Conv-Spatial EIT 训练脚本
==========================
两阶段训练:
  阶段1: 有监督 MSE 预训练（快速收敛）
  阶段2: 无监督物理约束精调

用法:
  python train_conv_spatial.py                          # 完整训练
  python train_conv_spatial.py --epochs_sup 50          # 有监督 50 epoch
  python train_conv_spatial.py --mode unsupervised      # 仅无监督
  python train_conv_spatial.py --generate               # 重新生成数据

后台运行（防止 SSH 断开导致训练中断）:
  nohup python train_conv_spatial.py [参数] > train.log 2>&1 &
  tail -f train.log                                     # 查看实时输出

或用 tmux:
  tmux new -s eit
  python train_conv_spatial.py [参数]
  Ctrl+B, D 脱离; tmux attach -t eit 重新连回

注意:
  - 历史最佳模型 RE=0.103（hidden=512, 2026-06-17）
  - Checkpoint 按运行隔离: checkpoints/{run_id}/
  - 恢复训练: python train_conv_spatial.py --resume checkpoints/{run_id}/best.pt
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from multiprocessing import cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.conv_spatial_eit import ConvSpatialEIT
from data.datasets.eit_dataset import MemoryEITDataset
from data.eit_forward import EITForwardSolver
from training.recorder import TrainingRecorder
from training.loss import edge_weighted_mse, AdaptiveLossWeighter


def get_mesh_data(config_path):
    """从 mesh_config 获取网格信息"""
    solver = EITForwardSolver(config_path)
    centers = solver.element_centers
    if centers.shape[1] > 2:
        centers = centers[:, :2]
    return centers, solver.mesh.element, solver.n_elems, solver


def voltage_masking(V, mask_ratio=0.15):
    """电压掩码数据增强: 以 mask_ratio 概率随机遮盖电压通道"""
    if mask_ratio > 0:
        mask = torch.rand_like(V) > mask_ratio
        return V * mask
    return V


def extract_model_state(ckpt):
    """兼容 raw state_dict、{'model': ...}、{'model_state_dict': ...} 三种格式。"""
    if isinstance(ckpt, dict):
        for key in ("model", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def reset_ema_to_model(ema_model, model):
    ema_model.module.load_state_dict(model.state_dict())
    if hasattr(ema_model, "n_averaged"):
        ema_model.n_averaged.zero_()


def update_ema(ema_model, model, decay):
    """手动 EMA，同时同步 BN 等 buffers，避免验证时使用初始统计量。"""
    ema_module = ema_model.module
    with torch.no_grad():
        for ema_p, p in zip(ema_module.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)
        for ema_b, b in zip(ema_module.buffers(), model.buffers()):
            if torch.is_floating_point(ema_b):
                ema_b.data.mul_(decay).add_(b.data, alpha=1 - decay)
            else:
                ema_b.data.copy_(b.data)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--mesh_config", default="config/mesh_config.yaml")
    parser.add_argument("--epochs_sup", type=int, default=80, help="有监督预训练轮数")
    parser.add_argument("--epochs_unsup", type=int, default=200, help="无监督精调轮数")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                        help="梯度累积步数；显存不够时用小 batch 模拟大 batch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.99,
                        help="EMA 衰减系数；短训建议 0.99，长训可调到 0.999")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--gnn_layers", type=int, default=4)
    parser.add_argument("--no_gat", action="store_true", default=True,
                        help="关闭 GATv2，使用 SimpleGNN（默认开启）")
    parser.add_argument("--use_gat", action="store_true",
                        help="使用 GATv2 注意力（显存占用大，默认关闭）")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="GATv2 attention heads 数量")
    parser.add_argument("--mode", choices=["supervised", "unsupervised", "both"],
                        default="both", help="训练模式")
    parser.add_argument("--generate", action="store_true", help="强制重新生成数据")
    parser.add_argument("--resume", type=str, default=None, help="恢复 checkpoint")
    parser.add_argument("--workers", type=int, default=0, help="数据生成并行数")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb")
    parser.add_argument("--edge_ratio", type=float, default=0.5,
                        help="边缘样本比例 (默认0.5=50%%)")
    parser.add_argument("--edge_threshold", type=float, default=0.05,
                        help="边缘判定阈值 (米, 默认0.05)")
    parser.add_argument("--use_model_jacobian", action="store_true",
                        help="在模型内部启用 Jᵀr 残差校正（默认关闭，因为 J 病态时有害）")
    parser.add_argument("--voltage_mask_ratio", type=float, default=0.0,
                        help="训练时随机遮盖电压通道比例；短训排障默认关闭")
    parser.add_argument("--mcl_mode", choices=["jacobian", "full_fem"],
                        default="full_fem",
                        help="测量一致性损失模式: jacobian(线性近似,省显存) / full_fem(精确FEM, 默认)")
    # ── 无监督阶段专用参数 ──
    parser.add_argument("--unsup_lr", type=float, default=1e-5,
                        help="无监督精调学习率（默认1e-5，比预训练小10倍）")
    parser.add_argument("--use_fixed_weights", action="store_true",
                        help="使用固定损失权重代替自适应权重（推荐）")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="验证RE连续恶化N个epoch后提前停止（0=关闭）")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"设备: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision('high')
            print("  TF32 矩阵乘法加速: 已启用")
    else:
        print(f"设备: CPU")

    # ============ 1. 数据 ============
    centers, elements, n_elems, solver = get_mesh_data(args.mesh_config)
    print(f"网格: {n_elems} 单元")

    # 检查/生成数据
    h5_path = "data/generated/mixed_dataset.h5"
    if args.generate or not os.path.exists(h5_path):
        print("生成多样化数据集...")
        from data.generate_mixed_dataset import generate_dataset
        generate_dataset(
            config_path=args.mesh_config,
            output_dir="data/generated",
            workers=args.workers or cpu_count(),
            edge_ratio=getattr(args, 'edge_ratio', 0.5),
            edge_threshold=getattr(args, 'edge_threshold', 0.05),
        )

    train_ds = MemoryEITDataset(h5_path, split='train', voltage_mask_ratio=0.0)
    val_ds = MemoryEITDataset(h5_path, split='val', voltage_mask_ratio=0.0)
    print(f"训练: {len(train_ds)}, 验证: {len(val_ds)}")
    if args.grad_accum_steps > 1:
        print(f"梯度累积: batch_size={args.batch_size} × {args.grad_accum_steps} "
              f"= 等效 batch_size {args.batch_size * args.grad_accum_steps}")

    # ── 分层采样器：每个 batch 一半边缘、一半中心 ──
    class BalancedBatchSampler(torch.utils.data.Sampler):
        """
        确保每个 batch 中边缘样本比例 ≈ args.edge_ratio。
        边缘判定: 样本中根区域重心距中心 > edge_threshold。
        """
        def __init__(self, dataset, batch_size, edge_ratio=0.5, edge_threshold=0.05):
            self.batch_size = batch_size
            self.edge_ratio = edge_ratio
            self.edge_threshold = edge_threshold
            self.n_samples = len(dataset)

            # 预计算每个样本的边缘标签（向量化，比逐样本循环快几百倍）
            print("  计算样本边缘标签...")
            centers_np = centers  # (n_elems, 2)

            # 从 MemoryEITDataset 直接获取内存中的 sigmas
            all_sigmas = dataset.sigmas  # (n_samples, n_elems), already in memory

            root_mask = all_sigmas > 0.015  # (n_samples, n_elems)
            n_root = root_mask.sum(axis=1)  # (n_samples,)

            # 向量化计算质心: 每个样本取根区域中心坐标的平均值
            # centers_np: (n_elems, 2) -> (1, n_elems, 2)
            # root_mask: (n_samples, n_elems) -> (n_samples, n_elems, 1)
            # sum over elems gives (n_samples, 2)
            centroid_x = (centers_np[:, 0] * root_mask).sum(axis=1) / np.maximum(n_root, 1)
            centroid_y = (centers_np[:, 1] * root_mask).sum(axis=1) / np.maximum(n_root, 1)
            dist = np.sqrt(centroid_x**2 + centroid_y**2)
            self.labels = np.where(n_root > 0, (dist > edge_threshold).astype(np.int64), 0)

            self.edge_idx = np.where(self.labels == 1)[0]
            self.center_idx = np.where(self.labels == 0)[0]
            print(f"  边缘: {len(self.edge_idx)}, 中心: {len(self.center_idx)}")

        def __iter__(self):
            edge_idx = self.edge_idx.copy()
            center_idx = self.center_idx.copy()
            np.random.shuffle(edge_idx)
            np.random.shuffle(center_idx)

            # 目标: 每个 batch 中 edge_ratio 比例的边缘样本
            n_edge_per = max(1, int(self.batch_size * self.edge_ratio))
            n_center_per = self.batch_size - n_edge_per

            for i in range(0, max(len(edge_idx), len(center_idx)),
                           max(n_edge_per, n_center_per)):
                # 使用 np.take(mode='wrap') 确保每批恰好取 n_edge_per + n_center_per 个元素
                edge_batch = np.take(edge_idx, range(i, i + n_edge_per), mode='wrap')
                center_batch = np.take(center_idx, range(i, i + n_center_per), mode='wrap')
                batch = np.concatenate([edge_batch, center_batch])
                yield batch

        def __len__(self):
            n_edge_per = max(1, int(self.batch_size * self.edge_ratio))
            n_center_per = self.batch_size - n_edge_per
            step = max(n_edge_per, n_center_per)
            return int(np.ceil(max(len(self.edge_idx), len(self.center_idx)) / step))

    sampler = BalancedBatchSampler(
        train_ds, batch_size=args.batch_size,
        edge_ratio=getattr(args, 'edge_ratio', 0.5),
        edge_threshold=getattr(args, 'edge_threshold', 0.05),
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)

    # ============ 2. 模型 ============
    use_gat_flag = args.use_gat and not args.no_gat  # no_gat 默认 True
    model = ConvSpatialEIT(
        n_frequencies=6,
        n_meas=208,
        n_elems=n_elems,
        hidden_dim=args.hidden_dim,
        gnn_hidden=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        use_gat=use_gat_flag,
        n_heads=args.n_heads,
    )
    # v3: Jacobian 校正默认关闭（J 病态时有害）
    jac_path = "data/generated/jacobian.npy"
    model_jacobian = None
    if args.use_model_jacobian and os.path.exists(jac_path):
        model_jacobian = np.load(jac_path)
        if model_jacobian.ndim == 3:
            model_jacobian = model_jacobian[0]
        print(f"模型 Jᵀr 校正: 已启用 ({model_jacobian.shape})")
    else:
        print("模型 Jᵀr 校正: 已禁用 (性能更稳定)")
    model.setup_mesh(centers, elements, jacobian=model_jacobian)
    model = model.to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练记录器
    recorder = TrainingRecorder(
        name=f"v2_{args.mode}_hd{args.hidden_dim}",
    )
    n_params = sum(p.numel() for p in model.parameters())
    recorder.save_meta({
        "hidden_dim": args.hidden_dim,
        "gnn_hidden": args.hidden_dim,
        "gnn_layers": args.gnn_layers,
        "batch_size": args.batch_size,
        "mode": args.mode,
        "epochs_sup": args.epochs_sup,
        "epochs_unsup": args.epochs_unsup,
        "model_params": n_params,
        "lr": args.lr,
        "use_model_jacobian": args.use_model_jacobian,
        "use_gat": use_gat_flag,
        "n_heads": args.n_heads,
        "mcl_mode": args.mcl_mode,
        "use_linear_output": True,
        "loss": "combined_v3",
    })
    # torch.compile 暂不兼容(位置编码buffer跨设备问题), 后续适配
    # model = torch.compile(model)

    # ── 每运行独立的 checkpoint 路径 ──
    ckpt_dir = f"checkpoints/{recorder.run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_sup_ckpt_path = f"{ckpt_dir}/best_supervised.pt"
    best_unsup_ckpt_path = f"{ckpt_dir}/best_unsupervised.pt"
    final_ckpt_path = f"{ckpt_dir}/final.pt"

    # v3: 使用旧版 loss（不引入不稳定因素），但保留模型/ckpt 改进
    criterion = edge_weighted_mse

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    # 有监督阶段使用普通余弦退火（稳定，不重启）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_sup, eta_min=1e-6)

    # ── EMA 指数移动平均 ──
    ema_model = torch.optim.swa_utils.AveragedModel(model).to(device)
    ema_decay = args.ema_decay
    reset_ema_to_model(ema_model, model)
    adaptive_weighter = None  # 无监督阶段使用

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(extract_model_state(ckpt))
        if isinstance(ckpt, dict) and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"  恢复优化器状态")
        if isinstance(ckpt, dict) and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            print(f"  恢复调度器状态")
        if isinstance(ckpt, dict) and 'ema_model' in ckpt:
            ema_model.load_state_dict(ckpt['ema_model'])
            print(f"  恢复EMA模型权重")
        else:
            reset_ema_to_model(ema_model, model)
        run_info = ckpt.get('run_id', '?')
        epoch_info = ckpt.get('epoch', '?')
        re_info = ckpt.get('best_re', '?')
        print(f"恢复: {args.resume} (run={run_info}, epoch={epoch_info}, best_re={re_info})")

    # ============ 3. 有监督预训练 ============
    if args.mode in ("supervised", "both"):
        print("\n" + "=" * 50)
        print("阶段 1: 有监督预训练 (v3 loss)")
        print("=" * 50)

        best_re = float('inf')
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

        for epoch in range(1, args.epochs_sup + 1):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(tqdm(train_loader, desc=f"Sup Epoch {epoch}"), start=1):
                V = batch['voltages'].to(device)
                S = batch['sigmas'].to(device)
                if not torch.isfinite(V).all() or not torch.isfinite(S).all():
                    continue
                B = V.shape[0]
                V_img = V.view(B, 6, 13, 16)
                V_img = voltage_masking(V_img, mask_ratio=args.voltage_mask_ratio)

                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    out = model(V_img)
                    loss = criterion(out['sigma'], S)

                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1.0:
                    continue

                scaler.scale(loss / args.grad_accum_steps).backward()
                should_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))
                if should_step:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                    # 梯度爆炸预警
                    if grad_norm > 2.0:
                        print(f"  ⚠ 大梯度: {grad_norm:.2f} | loss={loss.item():.6f}")

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema_model, model, ema_decay)
                epoch_loss += loss.item()

            scheduler.step()

            # 验证：同时看 raw 和 EMA。短训时 EMA 可能滞后，不能只看 EMA。
            model.eval()
            ema_model.eval()
            raw_val_loss = 0.0
            ema_val_loss = 0.0
            raw_pred, ema_pred, all_gt = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    V = batch['voltages'].to(device).view(-1, 6, 13, 16)
                    S = batch['sigmas'].to(device)
                    raw_out = model(V)
                    ema_out = ema_model(V)
                    raw_val_loss += criterion(raw_out['sigma'], S).item()
                    ema_val_loss += criterion(ema_out['sigma'], S).item()
                    raw_pred.append(raw_out['sigma'].cpu())
                    ema_pred.append(ema_out['sigma'].cpu())
                    all_gt.append(S.cpu())

            raw_pred = torch.cat(raw_pred)
            ema_pred = torch.cat(ema_pred)
            all_gt = torch.cat(all_gt)
            raw_re = torch.norm(raw_pred - all_gt, dim=-1).mean() / \
                 (torch.norm(all_gt, dim=-1).mean() + 1e-8)
            ema_re = torch.norm(ema_pred - all_gt, dim=-1).mean() / \
                 (torch.norm(all_gt, dim=-1).mean() + 1e-8)
            use_ema_best = ema_re <= raw_re
            re = ema_re if use_ema_best else raw_re
            val_loss = ema_val_loss if use_ema_best else raw_val_loss
            best_source = "ema" if use_ema_best else "raw"

            print(f"  Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.6f}"
                  f" | Raw Val: {raw_val_loss/len(val_loader):.6f} | Raw RE: {raw_re:.4f}"
                  f" | EMA Val: {ema_val_loss/len(val_loader):.6f} | EMA RE: {ema_re:.4f}"
                  f" | Best: {best_source}")
            recorder.log_epoch(phase="supervised", epoch=epoch,
                               loss=epoch_loss/len(train_loader),
                               val_loss=val_loss/len(val_loader),
                               re=re.item(),
                               raw_re=raw_re.item(),
                               ema_re=ema_re.item())

            if re < best_re:
                best_re = re
                os.makedirs(ckpt_dir, exist_ok=True)
                save_ema_model = ema_model
                if not use_ema_best:
                    save_ema_model = torch.optim.swa_utils.AveragedModel(model).to(device)
                    reset_ema_to_model(save_ema_model, model)
                torch.save({
                    'model': model.state_dict(),
                    'ema_model': save_ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'n_elems': n_elems,
                    'hidden_dim': args.hidden_dim,
                    'gnn_hidden': args.hidden_dim,
                    'gnn_layers': args.gnn_layers,
                    'use_gat': use_gat_flag,
                    'n_heads': args.n_heads,
                    'best_source': best_source,
                    'epoch': epoch,
                    'best_re': best_re.item(),
                    'run_id': recorder.run_id,
                    'phase': 'supervised',
                }, best_sup_ckpt_path)
                print(f"  → 保存有监督最佳 ({best_source}, RE={best_re:.4f}) 到 {best_sup_ckpt_path}")
                recorder.log_event("best_supervised_saved", re=best_re.item(),
                                   source=best_source, epoch=epoch)

            if re < 0.03:
                print(f"  ✓ 预训练收敛 (RE={re:.4f} < 0.03)")
                break

    # ============ 4. 无监督精调 ============
    if args.mode in ("unsupervised", "both") and args.mode != "supervised":
        print("\n" + "=" * 50)
        print("阶段 2: 无监督物理约束精调")
        print("=" * 50)

        # 加载最佳有监督模型
        if os.path.exists(best_sup_ckpt_path):
            best_ckpt = torch.load(best_sup_ckpt_path, map_location=device)
            if 'ema_model' in best_ckpt:
                ema_model.load_state_dict(best_ckpt['ema_model'])
                model.load_state_dict(ema_model.module.state_dict())
            else:
                model.load_state_dict(best_ckpt.get('model', best_ckpt))
                reset_ema_to_model(ema_model, model)
            best_re = best_ckpt.get('best_re', '?')
            print(f"加载有监督预训练权重: {best_sup_ckpt_path} (best_re={best_re})")
        else:
            print(f"⚠ 未找到有监督 ckpt: {best_sup_ckpt_path}，使用当前权重")
            best_re = float('inf')

        if device.type == 'cuda' and args.wandb:
            import wandb
            wandb.init(project="conv-spatial-eit", config=vars(args))

        from training.loss import MeasurementConsistencyLoss, TVRegularizationLoss
        from training.loss import SigmaDeviationLoss

        # 预计算 Jacobian（可选）
        jacobian = None
        jac_path = "data/generated/jacobian.npy"
        if os.path.exists(jac_path):
            jacobian = torch.from_numpy(np.load(jac_path)).float().to(device)
            print(f"加载 Jacobian: {jacobian.shape}")

        # 损失函数
        mcl = MeasurementConsistencyLoss(
            mode=args.mcl_mode,
            jacobian=jacobian,
            sigma_ref_value=0.01,
            forward_solver=lambda s: solver.solve_multi_frequency(s),
            fem_interval=10,        # 更频繁的FEM求解（从20改为10）
            fem_subset_size=4,      # 每次FEM只算4个样本（加速关键）
        )
        tvl = TVRegularizationLoss(
            element_centers=torch.from_numpy(centers).float(),
            mesh_elements=torch.from_numpy(elements).long(),
            mesh_nodes=torch.from_numpy(solver.mesh.node[:, :2]).float(),
        )
        sdl = SigmaDeviationLoss(sigma_ref_value=0.01)

        # ── 无监督阶段：重新创建优化器和调度器 ──
        unsup_optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.unsup_lr, weight_decay=1e-6)
        unsup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            unsup_optimizer, T_max=args.epochs_unsup, eta_min=1e-7)
        print(f"无监督学习率: {args.unsup_lr}")

        # 固定损失权重（v3: 更保守，增加监督锚点，减少物理约束）
        loss_weights = {
            "loss_sup": 0.8,   # 有监督锚点 (↑ 防止漂移)
            "loss_m": 0.3,     # 物理约束 (↓ 减少 Jacobian 近似误差)
            "loss_t": 0.05,    # TV正则
            "loss_d": 0.05,    # 偏离约束 (↓)
        }
        print(f"损失权重: {loss_weights}")

        # Early stopping
        patience_counter = 0
        unsup_best_re = float('inf')

        for epoch in range(1, args.epochs_unsup + 1):
            model.train()
            epoch_loss = 0.0
            epoch_losses = {"sup": 0.0, "m": 0.0, "t": 0.0, "d": 0.0}
            unsup_optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(tqdm(train_loader, desc=f"Unsup Epoch {epoch}"), start=1):
                V = batch['voltages'].to(device).view(-1, 6, 13, 16)
                V = voltage_masking(V, mask_ratio=args.voltage_mask_ratio)
                S_gt = batch['sigmas'].to(device)  # GT 用于半监督锚点
                if not torch.isfinite(V).all() or not torch.isfinite(S_gt).all():
                    print(f"  ⚠ 跳过非有限数据 batch: idx={batch.get('idx')}")
                    continue

                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    out = model(V)
                    sp = out['sigma']

                    loss_m = mcl(sp, batch['voltages'].to(device))
                    loss_t = tvl(sp)
                    loss_d = sdl(sp)
                    loss_sup = criterion(sp, S_gt)  # 半监督锚点

                    # 固定权重加权（稳定，推荐）
                    total = (loss_weights["loss_sup"] * loss_sup +
                             loss_weights["loss_m"] * loss_m +
                             loss_weights["loss_t"] * loss_t +
                             loss_weights["loss_d"] * loss_d)

                # 跳过异常 batch
                if torch.isnan(total) or torch.isinf(total) or total.item() > 10.0:
                    print(f"  ⚠ 跳过异常 batch: total_loss={total.item():.4f}")
                    continue

                (total / args.grad_accum_steps).backward()
                should_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))
                if should_step:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 更严格的梯度裁剪
                    if grad_norm > 0.5:
                        print(f"  ⚠ 大梯度: {grad_norm:.2f} | loss={total.item():.4f}")
                    unsup_optimizer.step()
                    unsup_optimizer.zero_grad(set_to_none=True)
                    update_ema(ema_model, model, ema_decay)

                epoch_loss += total.item()
                epoch_losses["sup"] += loss_sup.item()
                epoch_losses["m"] += loss_m.item()
                epoch_losses["t"] += loss_t.item()
                epoch_losses["d"] += loss_d.item()

            unsup_scheduler.step()

            # ── 验证监控 ──
            model.eval()
            ema_model.eval()
            raw_pred, ema_pred, all_gt = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    V = batch['voltages'].to(device).view(-1, 6, 13, 16)
                    S = batch['sigmas'].to(device)
                    raw_out = model(V)
                    ema_out = ema_model(V)
                    raw_pred.append(raw_out['sigma'].cpu())
                    ema_pred.append(ema_out['sigma'].cpu())
                    all_gt.append(S.cpu())

            raw_pred = torch.cat(raw_pred)
            ema_pred = torch.cat(ema_pred)
            all_gt = torch.cat(all_gt)
            raw_re = torch.norm(raw_pred - all_gt, dim=-1).mean() / \
                 (torch.norm(all_gt, dim=-1).mean() + 1e-8)
            ema_re = torch.norm(ema_pred - all_gt, dim=-1).mean() / \
                 (torch.norm(all_gt, dim=-1).mean() + 1e-8)
            use_ema = ema_re <= raw_re
            val_re = ema_re if use_ema else raw_re

            n_batches = len(train_loader)
            print(f"  Unsup Epoch {epoch:2d} | Loss: {epoch_loss/n_batches:.4f} "
                  f"(sup:{epoch_losses['sup']/n_batches:.4f}, m:{epoch_losses['m']/n_batches:.4f}, "
                  f"t:{epoch_losses['t']/n_batches:.4f}, d:{epoch_losses['d']/n_batches:.4f})")
            print(f"    Val RE: {val_re:.4f} ({'EMA' if use_ema else 'Raw'}) | LR: {unsup_optimizer.param_groups[0]['lr']:.2e}")

            recorder.log_epoch(phase="unsupervised", epoch=epoch,
                               loss=epoch_loss/n_batches, val_re=val_re.item())

            # ── Early Stopping & 保存最佳无监督模型 ──
            if val_re < unsup_best_re:
                unsup_best_re = val_re
                patience_counter = 0
                torch.save({
                    'model': model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': unsup_optimizer.state_dict(),
                    'scheduler': unsup_scheduler.state_dict(),
                    'epoch': epoch,
                    'val_re': val_re.item(),
                    'n_elems': n_elems,
                    'hidden_dim': args.hidden_dim,
                    'gnn_hidden': args.hidden_dim,
                    'gnn_layers': args.gnn_layers,
                    'use_gat': use_gat_flag,
                    'n_heads': args.n_heads,
                    'mode': 'unsupervised',
                    'run_id': recorder.run_id,
                    'phase': 'unsupervised',
                }, best_unsup_ckpt_path)
                print(f"    → 保存无监督最佳 (RE={val_re:.4f})")
                recorder.log_event("best_unsupervised_saved", re=val_re.item(), epoch=epoch)
            else:
                patience_counter += 1
                if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                    print(f"    ⚠ 验证RE连续{patience_counter}轮未改善，提前停止")
                    break

            # 定期保存 checkpoint
            if epoch % 20 == 0:
                ckpt_path = f"{ckpt_dir}/unsup_epoch{epoch}.pt"
                torch.save({
                    'model': model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': unsup_optimizer.state_dict(),
                    'scheduler': unsup_scheduler.state_dict(),
                    'epoch': epoch,
                    'val_re': val_re.item(),
                    'n_elems': n_elems,
                    'hidden_dim': args.hidden_dim,
                    'gnn_hidden': args.hidden_dim,
                    'gnn_layers': args.gnn_layers,
                    'use_gat': use_gat_flag,
                    'n_heads': args.n_heads,
                    'mode': 'unsupervised',
                    'run_id': recorder.run_id,
                }, ckpt_path)
                print(f"    → 已保存: {ckpt_path}")
                recorder.log_event("checkpoint_saved", path=ckpt_path, epoch=epoch)

    # ============ 5. 保存最终模型 ============
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'ema_model': ema_model.state_dict(),
        'n_elems': n_elems,
        'hidden_dim': args.hidden_dim,
        'gnn_hidden': args.hidden_dim,
        'gnn_layers': args.gnn_layers,
        'use_gat': use_gat_flag,
        'n_heads': args.n_heads,
        'run_id': recorder.run_id,
    }, final_ckpt_path)
    print(f"\n✅ 模型已保存: {final_ckpt_path}")
    recorder.log_event("training_completed", final_model=final_ckpt_path)
    recorder.set_status("completed")


if __name__ == "__main__":
    train()
