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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=4)
    parser.add_argument("--mode", choices=["supervised", "unsupervised", "both"],
                        default="both", help="训练模式")
    parser.add_argument("--generate", action="store_true", help="强制重新生成数据")
    parser.add_argument("--resume", type=str, default=None, help="恢复 checkpoint")
    parser.add_argument("--workers", type=int, default=0, help="数据生成并行数")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

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
        )

    train_ds = EITDataset(h5_path, split='train', voltage_mask_ratio=0.0)
    val_ds = EITDataset(h5_path, split='val', voltage_mask_ratio=0.0)
    print(f"训练: {len(train_ds)}, 验证: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_sup + args.epochs_unsup)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
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
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        if args.mode == "both":
            ckpt_path = "checkpoints/conv_spatial_best.pt"
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"加载有监督预训练权重: {ckpt_path}")

        if device.type == 'cuda' and args.wandb:
            import wandb
            wandb.init(project="conv-spatial-eit", config=vars(args))

        from training.loss import MeasurementConsistencyLoss, TVRegularizationLoss
        from training.loss import SmoothnessLoss, SigmaDeviationLoss

        # 预计算 Jacobian（可选）
        jacobian = None
        jac_path = "data/generated/jacobian.npy"
        if os.path.exists(jac_path):
            jacobian = torch.from_numpy(np.load(jac_path)).float().to(device)
            print(f"加载 Jacobian: {jacobian.shape}")

        # 损失函数
        mcl = MeasurementConsistencyLoss(
            mode='jacobian' if jacobian is not None else 'full_fem',
            jacobian=jacobian,
            sigma_ref_value=0.01,
            forward_solver=lambda s: solver.solve_multi_frequency(s),
        )
        tvl = TVRegularizationLoss(
            element_centers=torch.from_numpy(centers).float(),
            mesh_elements=torch.from_numpy(elements).long(),
            mesh_nodes=torch.from_numpy(solver.mesh.node[:, :2]).float(),
        )
        sdl = SigmaDeviationLoss(sigma_ref_value=0.01)
        sml = SmoothnessLoss()

        for epoch in range(1, args.epochs_unsup + 1):
            model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Unsup Epoch {epoch}"):
                V = batch['voltages'].to(device).view(-1, 6, 13, 16)

                optimizer.zero_grad()
                out = model(V)
                sp = out['sigma']

                loss_m = mcl(sp, batch['voltages'].to(device))
                loss_t = tvl(sp)
                loss_d = sdl(sp)
                loss_s = sml(sp)
                total = loss_m + 0.05 * loss_t + 0.1 * loss_d + 0.02 * loss_s

                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += total.item()

            scheduler.step()
            print(f"  Unsup Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f}")

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
