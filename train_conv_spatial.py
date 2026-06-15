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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.conv_spatial_eit import ConvSpatialEIT
from data.datasets.eit_dataset import EITDataset
from data.eit_forward import EITForwardSolver


def get_mesh_data(config_path):
    """从 mesh_config 获取网格信息"""
    solver = EITForwardSolver(config_path)
    centers = solver.element_centers
    if centers.shape[1] > 2:
        centers = centers[:, :2]
    return centers, solver.mesh.element, solver.n_elems, solver


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--mesh_config", default="config/mesh_config.yaml")
    parser.add_argument("--epochs_sup", type=int, default=50, help="有监督预训练轮数")
    parser.add_argument("--epochs_unsup", type=int, default=200, help="无监督精调轮数")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=4)
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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"设备: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        torch.set_float32_matmul_precision('high')
        print("  TF32 矩阵乘法加速: 已启用")
    else:
        print(f"设备: CPU")

    # ============ 1. 数据 ============
    centers, elements, n_elems, solver = get_mesh_data(args.mesh_config)
    print(f"网格: {n_elems} 单元")

    # 检查/生成数据
    h5_path = "data/generated/circle_dataset.h5"
    if args.generate or not os.path.exists(h5_path):
        print("生成单圆数据集...")
        from data.generate_circle_dataset import generate_dataset
        generate_dataset(
            config_path=args.mesh_config,
            output_dir="data/generated",
            workers=args.workers or cpu_count(),
            edge_ratio=getattr(args, 'edge_ratio', 0.5),
            edge_threshold=getattr(args, 'edge_threshold', 0.05),
        )

    train_ds = EITDataset(h5_path, split='train', voltage_mask_ratio=0.0)
    val_ds = EITDataset(h5_path, split='val', voltage_mask_ratio=0.0)
    print(f"训练: {len(train_ds)}, 验证: {len(val_ds)}")

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

            # 直接从 HDF5 一次性加载所有 sigmas，避免 Dataset 逐个读取的开销
            import h5py
            h5_path = dataset.h5_path
            split = dataset.split
            with h5py.File(h5_path, 'r') as f:
                all_sigmas = f[f'{split}/sigmas'][:]  # (n_samples, n_elems)

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
                batch = np.concatenate([
                    edge_idx[i % len(edge_idx):(i + n_edge_per) % len(edge_idx)],
                    center_idx[i % len(center_idx):(i + n_center_per) % len(center_idx)],
                ])
                if len(batch) > self.batch_size:
                    batch = batch[:self.batch_size]
                yield batch

        def __len__(self):
            return len(self.edge_idx) // max(1, int(self.batch_size * self.edge_ratio))

    sampler = BalancedBatchSampler(
        train_ds, batch_size=args.batch_size,
        edge_ratio=getattr(args, 'edge_ratio', 0.5),
        edge_threshold=getattr(args, 'edge_threshold', 0.05),
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ============ 2. 模型 ============
    model = ConvSpatialEIT(
        n_frequencies=6,
        n_meas=208,
        n_elems=n_elems,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
    ).to(device)
    model.setup_mesh(centers, elements)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    model = torch.compile(model)
    print("  torch.compile: 已启用")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-6, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_sup + args.epochs_unsup)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"  恢复优化器状态")
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            print(f"  恢复调度器状态")
        print(f"恢复: {args.resume}")

    # ============ 3. 有监督预训练 ============
    if args.mode in ("supervised", "both"):
        print("\n" + "=" * 50)
        print("阶段 1: 有监督 MSE 预训练")
        print("=" * 50)

        criterion = nn.MSELoss()
        best_re = float('inf')
        scaler = torch.amp.GradScaler()

        for epoch in range(1, args.epochs_sup + 1):
            model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Sup Epoch {epoch}"):
                V = batch['voltages'].to(device)  # (B, 6, 208)
                S = batch['sigmas'].to(device)     # (B, n_elems)
                B = V.shape[0]
                V_img = V.view(B, 6, 13, 16)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    out = model(V_img)
                    loss = criterion(out['sigma'], S)

                # 跳过 loss 异常的 batch（防止梯度爆炸）
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1.0:
                    print(f"  ⚠ 跳过异常 batch: loss={loss.item():.4f}")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                # 梯度爆炸预警
                if grad_norm > 2.0:
                    print(f"  ⚠ 大梯度: {grad_norm:.2f} | loss={loss.item():.6f}")

                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

            scheduler.step()

            # 验证
            model.eval()
            val_loss = 0.0
            all_pred, all_gt = [], []
            with torch.no_grad():
                for batch in val_loader:
                    V = batch['voltages'].to(device).view(-1, 6, 13, 16)
                    S = batch['sigmas'].to(device)
                    out = model(V)
                    val_loss += criterion(out['sigma'], S).item()
                    all_pred.append(out['sigma'].cpu())
                    all_gt.append(S.cpu())

            all_pred = torch.cat(all_pred)
            all_gt = torch.cat(all_gt)
            re = torch.norm(all_pred - all_gt, dim=-1).mean() / \
                 (torch.norm(all_gt, dim=-1).mean() + 1e-8)

            print(f"  Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.6f}"
                  f" | Val: {val_loss/len(val_loader):.6f} | RE: {re:.4f}")

            if re < best_re:
                best_re = re
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), "checkpoints/conv_spatial_best.pt")
                print(f"  → 保存最佳模型 (RE={best_re:.4f})")

            if re < 0.03:
                print(f"  ✓ 预训练收敛 (RE={re:.4f} < 0.03)")
                break

    # ============ 4. 无监督精调 ============
    if args.mode in ("unsupervised", "both") and args.mode != "supervised":
        print("\n" + "=" * 50)
        print("阶段 2: 无监督物理约束精调")
        print("=" * 50)

        # 加载最佳有监督模型
        ckpt_path = "checkpoints/conv_spatial_best.pt"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"加载有监督预训练权重: {ckpt_path}")

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
            mode='full_fem',
            jacobian=jacobian,
            sigma_ref_value=0.01,
            forward_solver=lambda s: solver.solve_multi_frequency(s),
            fem_interval=5,
        )
        tvl = TVRegularizationLoss(
            element_centers=torch.from_numpy(centers).float(),
            mesh_elements=torch.from_numpy(elements).long(),
            mesh_nodes=torch.from_numpy(solver.mesh.node[:, :2]).float(),
        )
        sdl = SigmaDeviationLoss(sigma_ref_value=0.01)

        for epoch in range(1, args.epochs_unsup + 1):
            model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Unsup Epoch {epoch}"):
                V = batch['voltages'].to(device).view(-1, 6, 13, 16)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    out = model(V)
                    sp = out['sigma']

                    loss_m = mcl(sp, batch['voltages'].to(device))
                    loss_t = tvl(sp)
                    loss_d = sdl(sp)
                    total = loss_m + 0.05 * loss_t + 0.01 * loss_d

                # 跳过异常 batch
                if torch.isnan(total) or torch.isinf(total) or total.item() > 10.0:
                    print(f"  ⚠ 跳过异常 batch: total_loss={total.item():.4f}")
                    continue

                total.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if grad_norm > 2.0:
                    print(f"  ⚠ 大梯度: {grad_norm:.2f} | loss={total.item():.4f}")
                optimizer.step()
                epoch_loss += total.item()

            scheduler.step()
            print(f"  Unsup Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f}")

            # 每 20 epoch 保存一次 checkpoint
            if epoch % 20 == 0:
                ckpt_path = f"checkpoints/conv_spatial_unsup_epoch{epoch}.pt"
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': epoch_loss / len(train_loader),
                    'n_elems': n_elems,
                    'hidden_dim': args.hidden_dim,
                    'gnn_layers': args.gnn_layers,
                }, ckpt_path)
                print(f"  → 已保存: {ckpt_path}")

    # ============ 5. 保存最终模型 ============
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/conv_spatial_final.pt"
    torch.save({
        'model': model.state_dict(),
        'n_elems': n_elems,
        'hidden_dim': args.hidden_dim,
        'gnn_layers': args.gnn_layers,
    }, save_path)
    print(f"\n✅ 模型已保存: {save_path}")


if __name__ == "__main__":
    train()
