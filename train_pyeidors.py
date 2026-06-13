#!/usr/bin/env python3
"""
使用 PyEIDORS 数据训练 EIT 重建网络
===================================

工作流程:
  1. 在服务器上用 PyEIDORS 生成数据:
     python data/pyeidors_data_generator.py --n_train 20000 --n_val 1000

  2. 训练模型:
     python train_pyeidors.py

  3. 评估:
     python evaluation/evaluate.py --checkpoint checkpoints/...pt
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 项目路径
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from torch.utils.data import Dataset, DataLoader
import h5py


class PyEIDORSDataset(Dataset):
    """PyEIDORS 数据集加载器"""

    def __init__(self, h5_path: str, split: str = 'train'):
        """
        Args:
            h5_path: HDF5 数据文件路径
            split: 'train', 'val', 或 'test'
        """
        self.h5_path = h5_path
        self.split = split

        # 读取数据集信息
        with h5py.File(h5_path, 'r') as f:
            grp = f[split]
            self.n_samples = grp['voltages'].shape[0]
            self.n_freq = grp['voltages'].shape[1]
            self.n_meas = grp['voltages'].shape[2]
            self.n_elems = grp['sigmas'].shape[1]

        print(f"  [{split}] {self.n_samples} 样本, "
              f"电压维度: ({self.n_freq}, {self.n_meas}), "
              f"电导率维度: {self.n_elems}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[self.split]
            voltages = grp['voltages'][idx]  # (n_freq, n_meas)
            sigma = grp['sigmas'][idx]       # (n_elems,)
            mask = grp['masks'][idx]         # (n_elems,)

        return {
            'voltages': torch.from_numpy(voltages).float(),
            'sigma': torch.from_numpy(sigma).float(),
            'mask': torch.from_numpy(mask).float(),
        }


def create_model(model_type: str, n_meas: int, n_freq: int, n_elems: int,
                 hidden_dim: int = 512, device: str = 'cuda'):
    """创建模型"""
    print(f"\n创建模型: {model_type}")

    if model_type == 'simple':
        from models.simple_model import SimpleSFSBLC
        model = SimpleSFSBLC(
            n_freq=n_freq,
            n_meas=n_meas,
            n_elems=n_elems,
            hidden_dim=hidden_dim
        )
    elif model_type == 'deep':
        from models.linear_model import DeepEITModel
        model = DeepEITModel(
            n_freq=n_freq,
            n_meas=n_meas,
            n_elems=n_elems,
            hidden_dim=hidden_dim,
            n_layers=8
        )
    elif model_type == 'physics':
        from models.universal_eit import PhysicsInformedEIT
        model = PhysicsInformedEIT(
            n_freq=n_freq,
            n_meas=n_meas,
            n_elems=n_elems,
            hidden_dim=hidden_dim
        )
    elif model_type == 'gnn':
        from models.improved_gnn_model import ImprovedEITModelGNN
        model = ImprovedEITModelGNN(
            n_freq=n_freq,
            n_meas=n_meas,
            n_elems=n_elems,
            hidden_dim=hidden_dim,
            use_jacobian=False,
            use_attention=False
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    return model


def compute_loss(pred: torch.Tensor, target: torch.Tensor,
                 weights: dict = None) -> dict:
    """计算损失"""
    if weights is None:
        weights = {'mse': 1.0, 'tv': 0.01}

    # MSE 损失
    mse_loss = torch.nn.functional.mse_loss(pred, target)

    # TV 正则化 (简化版)
    tv_loss = torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))

    # 总损失
    total_loss = weights.get('mse', 1.0) * mse_loss + weights.get('tv', 0.01) * tv_loss

    return {
        'total': total_loss,
        'mse': mse_loss.item(),
        'tv': tv_loss.item()
    }


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """计算评估指标"""
    # 相对误差
    re = torch.norm(pred - target) / torch.norm(target)

    # 相关系数
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    corr = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]

    return {
        're': re.item(),
        'corr': corr.item() if not torch.isnan(corr) else 0.0
    }


def train_epoch(model, dataloader, optimizer, device, epoch, config):
    """训练一个 epoch"""
    model.train()

    total_loss = 0.0
    total_re = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        voltages = batch['voltages'].to(device)    # (B, n_freq, n_meas)
        sigma_target = batch['sigma'].to(device)   # (B, n_elems)

        # 前向传播
        optimizer.zero_grad()

        if hasattr(model, 'forward'):
            # 检查模型接口
            try:
                output = model(voltages)
                if isinstance(output, dict):
                    sigma_pred = output.get('sigma', output.get('reconstruction'))
                else:
                    sigma_pred = output
            except Exception as e:
                print(f"  前向传播错误: {e}")
                continue

        # 计算损失
        loss_dict = compute_loss(sigma_pred, sigma_target,
                                 weights={'mse': 1.0, 'tv': 0.01})
        loss = loss_dict['total']

        # 反向传播
        loss.backward()

        # 梯度裁剪
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        # 统计
        total_loss += loss.item()
        with torch.no_grad():
            metrics = compute_metrics(sigma_pred, sigma_target)
            total_re += metrics['re']
        n_batches += 1

        # 打印进度
        if (batch_idx + 1) % config.get('log_interval', 20) == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | RE: {metrics['re']:.4f}")

    return {
        'loss': total_loss / max(n_batches, 1),
        're': total_re / max(n_batches, 1)
    }


def validate(model, dataloader, device):
    """验证"""
    model.eval()

    total_re = 0.0
    total_corr = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            voltages = batch['voltages'].to(device)
            sigma_target = batch['sigma'].to(device)

            try:
                output = model(voltages)
                if isinstance(output, dict):
                    sigma_pred = output.get('sigma', output.get('reconstruction'))
                else:
                    sigma_pred = output
            except Exception as e:
                continue

            metrics = compute_metrics(sigma_pred, sigma_target)
            total_re += metrics['re']
            total_corr += metrics['corr']
            n_batches += 1

    return {
        're': total_re / max(n_batches, 1),
        'corr': total_corr / max(n_batches, 1)
    }


def main():
    parser = argparse.ArgumentParser(description='PyEIDORS 数据训练')
    parser.add_argument('--data', type=str, default='data/generated/pyeidors_dataset.h5',
                        help='PyEIDORS 数据路径')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'deep', 'physics', 'gnn'],
                        help='模型类型')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='隐藏层维度')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--output_dir', type=str, default='checkpoints/pyeidors',
                        help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的 checkpoint')

    args = parser.parse_args()

    # 检查数据
    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        print("\n请先生成数据:")
        print("  python data/pyeidors_data_generator.py --n_train 20000 --n_val 1000")
        return

    # 设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 加载数据集
    print(f"\n加载数据: {args.data}")
    train_dataset = PyEIDORSDataset(args.data, split='train')
    val_dataset = PyEIDORSDataset(args.data, split='val')

    # 获取维度
    n_freq = train_dataset.n_freq
    n_meas = train_dataset.n_meas
    n_elems = train_dataset.n_elems

    print(f"\n数据维度:")
    print(f"  频率数: {n_freq}")
    print(f"  测量数: {n_meas}")
    print(f"  网格单元数: {n_elems}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # 创建模型
    model = create_model(
        args.model, n_meas, n_freq, n_elems,
        hidden_dim=args.hidden_dim, device=device
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"恢复训练 from epoch {start_epoch}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练配置
    config = {
        'grad_clip': 1.0,
        'log_interval': 20
    }

    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    best_re = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch + 1, config
        )

        # 验证
        val_metrics = validate(model, val_loader, device)

        # 更新学习率
        scheduler.step()

        # 打印结果
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, RE: {train_metrics['re']:.4f}")
        print(f"  Val   - RE: {val_metrics['re']:.4f}, Corr: {val_metrics['corr']:.4f}")

        # 保存最佳模型
        if val_metrics['re'] < best_re:
            best_re = val_metrics['re']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_re': val_metrics['re'],
                'n_elems': n_elems,
                'n_meas': n_meas,
                'n_freq': n_freq,
            }, output_dir / 'best_model.pt')
            print(f"  ✅ 保存最佳模型 (RE: {best_re:.4f})")

        # 定期保存
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')

    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print("=" * 60)
    print(f"最佳验证 RE: {best_re:.4f}")
    print(f"模型保存: {output_dir}")


if __name__ == "__main__":
    main()
