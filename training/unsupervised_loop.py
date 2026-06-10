"""
无监督训练循环
===============
核心训练逻辑：物理约束驱动的无监督学习。

每步:
  1. 模型前向 → 预测电导率 σ_pred
  2. 物理损失: ||F(σ_pred) - V||² + TV(σ_pred)
  3. 正则化损失: 频率交叉一致性 + BLC校正约束
  4. 梯度回传 → 更新模型参数

在整个过程中，从未使用真实 σ_gt 作为监督信号！
(虽然有 σ_gt 时可用于验证集评估)
"""

import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from collections import defaultdict

# 项目模块 — 将项目根加入 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.sf_sblc import SFSBLC
from training.loss import (
    MeasurementConsistencyLoss,
    TVRegularizationLoss,
    FrequencyCrossConsistencyLoss,
    BLCCorrectionLoss,
    SmoothnessLoss,
    AdaptiveLossWeighter,
)
from training.optimizer import build_optimizer, build_scheduler
from training.logger import TrainLogger
from data.datasets.eit_dataset import EITDataset, EITDataModule


class UnsupervisedTrainer:
    """
    无监督训练器

    用法:
        trainer = UnsupervisedTrainer(config_path="config/train_config.yaml")
        trainer.train()

    或手动:
        trainer.setup(model, datamodule)
        trainer.run_one_epoch()
    """

    def __init__(self, config_path: str = "config/train_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备: {self.device}")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dm = None
        self.loss_fns = {}
        self.logger = None
        self.current_epoch = 0
        self.global_step = 0

    def setup(self, model: Optional[nn.Module] = None,
              datamodule: Optional[object] = None):
        """设置模型、数据和训练组件"""
        model_cfg = self.cfg['model']
        train_cfg = self.cfg['training']

        # --- 模型 ---
        if model is None:
            self.model = SFSBLC(
                input_dim=model_cfg['input_dim'],
                hidden_dim=model_cfg['hidden_dim'],
                n_frequencies=model_cfg['n_frequencies'],
                n_elems=model_cfg.get('n_elems', 1500),
                n_res_blocks=model_cfg.get('n_res_blocks', 8),
                dropout=model_cfg.get('dropout', 0.1),
                use_attention=model_cfg.get('use_attention', True),
            ).to(self.device)
        else:
            self.model = model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数: {total_params:,}")

        # --- 数据 ---
        if datamodule is None:
            data_cfg = self.cfg['data']
            self.dm = EITDataModule(
                h5_path=data_cfg['dataset_path'],
                batch_size=train_cfg['batch_size'],
                num_workers=data_cfg.get('num_workers', 4),
                voltage_mask_ratio=train_cfg['voltage_sampling']['mask_ratio'],
                jacobian_path=data_cfg.get('jacobian_path', None),
            )
        else:
            self.dm = datamodule

        # --- 损失函数 ---
        weights = train_cfg['loss_weights']
        self.loss_fns = {
            'meas': MeasurementConsistencyLoss(
                use_jacobian=True,
                jacobian=self._load_jacobian(),
            ),
            'tv': TVRegularizationLoss(
                element_centers=self._get_element_centers(),
                mesh_elements=self._get_mesh_elements(),
                mesh_nodes=self._get_mesh_nodes(),
            ),
            'freq': FrequencyCrossConsistencyLoss(),
            'blc': BLCCorrectionLoss(),
            'smooth': SmoothnessLoss(),
        }
        self.loss_weights = {
            'meas': weights['measurement_consistency'],
            'tv': weights['tv_regularization'],
            'freq': weights['frequency_cross'],
            'blc': weights['blc_correction'],
            'smooth': weights['smoothness'],
        }

        # 自适应权重（可选）
        self.adaptive_weighter = None
        if train_cfg.get('use_adaptive_weights', False):
            self.adaptive_weighter = AdaptiveLossWeighter(n_losses=5).to(self.device)

        # --- 优化器 ---
        self.optimizer = build_optimizer(self.model, self.cfg)

        # 必须先构建 DataLoader 才能知道 steps_per_epoch
        train_loader = self.dm.train_dataloader()
        self.scheduler = build_scheduler(self.optimizer, self.cfg,
                                          steps_per_epoch=len(train_loader))

        # --- 日志 ---
        self.logger = TrainLogger(
            log_dir=self.cfg.get('logging', {}).get('log_dir', 'logs'),
            use_tensorboard=self.cfg.get('logging', {}).get('use_tensorboard', True),
        )
        self.logger.save_config(self.cfg)

        print("设置完成，准备训练！")

    def _load_jacobian(self) -> Optional[torch.Tensor]:
        """加载预计算雅可比矩阵"""
        jacobian_path = self.cfg['data'].get('jacobian_path', None)
        if jacobian_path and os.path.exists(jacobian_path):
            J = np.load(jacobian_path)
            print(f"  加载雅可比: {J.shape}")
            return torch.from_numpy(J).float().to(self.device)
        return None

    def _get_element_centers(self):
        """从数据集获取单元中心"""
        ds = self.dm.train_dataset
        nodes = ds.mesh_nodes
        elems = ds.mesh_elements
        centers = np.mean(nodes[elems], axis=1)
        return torch.from_numpy(centers).float()

    def _get_mesh_elements(self):
        ds = self.dm.train_dataset
        return torch.from_numpy(ds.mesh_elements).long()

    def _get_mesh_nodes(self):
        ds = self.dm.train_dataset
        return torch.from_numpy(ds.mesh_nodes).float()

    def train(self):
        """完整训练流程"""
        train_cfg = self.cfg['training']
        log_cfg = self.cfg.get('logging', {})
        n_epochs = train_cfg['epochs']
        log_interval = log_cfg.get('log_interval', 10)
        save_interval = log_cfg.get('save_interval', 10)
        vis_interval = log_cfg.get('vis_interval', 20)

        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()

        for epoch in range(1, n_epochs + 1):
            self.current_epoch = epoch

            # --- 训练 ---
            train_metrics = self._train_one_epoch(train_loader, epoch)

            # --- 验证 ---
            val_metrics = self._validate(val_loader, epoch)

            # --- Logging ---
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                lr=self.optimizer.param_groups[0]['lr'],
                extra_metrics={
                    'train_meas': train_metrics.get('meas', 0),
                    'train_tv': train_metrics.get('tv', 0),
                    'val_re': val_metrics.get('re', 0),
                    'val_cc': val_metrics.get('cc', 0),
                    'val_ssim': val_metrics.get('ssim', 0),
                }
            )

            # --- 保存 checkpoint ---
            if epoch % save_interval == 0:
                self.logger.save_model(self.model, self.optimizer, epoch)

            # --- 可视化 ---
            if epoch % vis_interval == 0:
                self._visualize_reconstruction(val_loader, epoch)

        print(f"\n训练完成！ {n_epochs} epochs")
        # 保存最终模型
        self.logger.save_model(self.model, self.optimizer, 'final')
        self.logger.close()

    def _train_one_epoch(self, train_loader, epoch: int) -> Dict:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        metrics_sum = defaultdict(float)
        n_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # 移到设备
            voltages = batch['voltages'].to(self.device)  # (B, F, M)

            # --- 前向 ---
            out = self.model(voltages)
            sigma_pred = out['sigma']

            # --- 无监督损失计算 ---
            losses = {}
            # 测量一致性 (核心)
            losses['meas'] = self.loss_fns['meas'](
                sigma_pred, voltages
            )
            # TV 正则化
            losses['tv'] = self.loss_fns['tv'](sigma_pred)
            # 频率交叉一致性
            losses['freq'] = self.loss_fns['freq'](
                out.get('freq_weights'), sigma_pred,
                out.get('base_map')
            )
            # BLC 校正
            if out.get('blc_gates') is not None:
                losses['blc'] = self.loss_fns['blc'](out['blc_gates'])
            else:
                losses['blc'] = torch.tensor(0.0, device=self.device)
            # 平滑度
            losses['smooth'] = self.loss_fns['smooth'](sigma_pred)

            # --- 加权总损失 ---
            if self.adaptive_weighter:
                total = self.adaptive_weighter(losses)
            else:
                total = sum(
                    self.loss_weights[k] * losses[k]
                    for k in losses
                )

            # --- 反向传播 ---
            self.optimizer.zero_grad()
            total.backward()

            # 梯度裁剪
            grad_clip = self.cfg['training'].get('grad_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            # --- 统计 ---
            total_loss += total.item()
            for k, v in losses.items():
                metrics_sum[k] += v.item()
            self.global_step += 1

            # --- 日志 (每 N 步) ---
            if batch_idx % log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                losses_dict = {k: v.item() for k, v in losses.items()}
                self.logger.log_batch(self.global_step, losses_dict, lr)

            pbar.set_postfix({'loss': f"{total.item():.4f}"})

        avg_loss = total_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
        avg_metrics['loss'] = avg_loss

        return avg_metrics

    def _validate(self, val_loader, epoch: int) -> Dict:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                voltages = batch['voltages'].to(self.device)
                sigmas_gt = batch['sigmas'].to(self.device)

                out = self.model(voltages)
                sigma_pred = out['sigma']

                # 无监督损失（只用测量一致性）
                loss = self.loss_fns['meas'](sigma_pred, voltages)
                total_loss += loss.item()

                all_preds.append(sigma_pred.cpu())
                all_targets.append(sigmas_gt.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算评估指标
        re = self._relative_error(all_preds, all_targets)
        cc = self._correlation_coefficient(all_preds, all_targets)

        return {
            'loss': total_loss / len(val_loader),
            're': re,
            'cc': cc,
            'ssim': 0.0,  # 需要在网格上重建成图像后计算
        }

    def _relative_error(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """相对误差 RE = ||pred - target|| / ||target||"""
        error = torch.norm(pred - target, dim=-1)
        norm = torch.norm(target, dim=-1) + 1e-8
        return (error / norm).mean().item()

    def _correlation_coefficient(self, pred: torch.Tensor,
                                  target: torch.Tensor) -> float:
        """相关系数 CC"""
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        target_centered = target - target.mean(dim=-1, keepdim=True)
        cov = (pred_centered * target_centered).sum(dim=-1)
        std_pred = torch.sqrt((pred_centered ** 2).sum(dim=-1) + 1e-8)
        std_target = torch.sqrt((target_centered ** 2).sum(dim=-1) + 1e-8)
        cc = cov / (std_pred * std_target + 1e-8)
        return cc.mean().item()

    def _visualize_reconstruction(self, val_loader, epoch: int):
        """保存重建可视化"""
        import matplotlib.pyplot as plt

        self.model.eval()
        batch = next(iter(val_loader))

        with torch.no_grad():
            voltages = batch['voltages'][:4].to(self.device)
            sigmas_gt = batch['sigmas'][:4].cpu().numpy()

            out = self.model(voltages)
            sigmas_pred = out['sigma'].cpu().numpy()

        fig, axes = plt.subplots(4, 3, figsize=(9, 12))

        # 获取单元中心（用于散点图）
        ds = self.dm.train_dataset
        centers = np.mean(ds.mesh_nodes[ds.mesh_elements], axis=1)

        for i in range(4):
            # Ground Truth
            ax = axes[i, 0]
            sc = ax.scatter(centers[:, 0], centers[:, 1], c=sigmas_gt[i],
                            s=5, cmap='viridis', vmin=0.008, vmax=0.052)
            ax.set_title(f"GT [{i}]")
            ax.set_aspect('equal')

            # Prediction
            ax = axes[i, 1]
            ax.scatter(centers[:, 0], centers[:, 1], c=sigmas_pred[i],
                       s=5, cmap='viridis', vmin=0.008, vmax=0.052)
            ax.set_title(f"Pred [{i}]")
            ax.set_aspect('equal')

            # Error
            ax = axes[i, 2]
            error = np.abs(sigmas_pred[i] - sigmas_gt[i])
            ax.scatter(centers[:, 0], centers[:, 1], c=error,
                       s=5, cmap='hot', vmin=0, vmax=0.01)
            ax.set_title(f"Error [{i}]")
            ax.set_aspect('equal')

        plt.tight_layout()
        save_path = os.path.join(self.logger.experiment_dir,
                                  f"recon_epoch_{epoch}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  重建可视化: {save_path}")


if __name__ == "__main__":
    trainer = UnsupervisedTrainer("config/train_config.yaml")
    trainer.setup()
    trainer.train()
