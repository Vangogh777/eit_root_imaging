"""
训练日志与可视化工具
支持 TensorBoard + Weights & Biases (wandb)
"""

import os
import time
import json
import torch
import numpy as np
from typing import Dict, Optional, List, Any
from collections import defaultdict


class TrainLogger:
    """
    训练日志记录
    支持:
        - CSV 日志
        - TensorBoard
        - Weights & Biases (wandb)
        - 控制台输出
        - 重建可视化
    """

    def __init__(self, log_dir: str = "logs", use_tensorboard: bool = True,
                 use_wandb: bool = False, wandb_project: str = "eit-root-imaging",
                 wandb_entity: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 config: Optional[Dict] = None):
        os.makedirs(log_dir, exist_ok=True)

        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_name = f"eit_{timestamp}"

        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # CSV 文件
        self.csv_path = os.path.join(self.experiment_dir, "metrics.csv")
        self.csv_file = open(self.csv_path, 'w', encoding='utf-8')
        self.csv_writer = None
        self.csv_initialized = False

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.experiment_dir)
            except ImportError:
                print("[WARN] tensorboard not installed, skipping")
                self.use_tensorboard = False

        # Weights & Biases
        self.use_wandb = use_wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_name,
                    dir=log_dir,
                    config=config or {},
                    sync_tensorboard=True,  # 同步 TensorBoard 日志
                )
                self.wandb_run = wandb.run
                print(f"[wandb] 已连接: {wandb.run.url}")
            except ImportError:
                print("[WARN] wandb not installed, skipping. Run: pip install wandb")
                self.use_wandb = False
            except Exception as e:
                print(f"[WARN] wandb 初始化失败: {e}")
                self.use_wandb = False

        # 内存缓存
        self.epoch_metrics = defaultdict(list)
        self.batch_metrics = defaultdict(list)

        # 计时
        self.start_time = time.time()
        self.epoch_start = None

    def log_batch(self, step: int, losses: Dict[str, float],
                  lr: float, phase: str = "train"):
        """记录一个 batch 的指标"""
        for k, v in losses.items():
            key = f"{phase}/{k}"
            self.batch_metrics[key].append(v)

            if self.tb_writer:
                self.tb_writer.add_scalar(f"batch/{k}", v, step)

        if self.tb_writer:
            self.tb_writer.add_scalar("batch/lr", lr, step)

        # wandb 日志
        if self.use_wandb:
            import wandb
            log_dict = {f"batch/{k}": v for k, v in losses.items()}
            log_dict["batch/lr"] = lr
            log_dict["step"] = step
            wandb.log(log_dict)

    def log_epoch(self, epoch: int, train_loss: float,
                  val_loss: Optional[float] = None,
                  lr: Optional[float] = None,
                  extra_metrics: Optional[Dict] = None):
        """记录一个 epoch 的汇总指标"""
        now = time.time()
        elapsed = now - self.start_time
        epoch_time = now - (self.epoch_start or self.start_time)
        self.epoch_start = now

        # 更新汇总
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss or 0.0,
            'lr': lr or 0.0,
            'elapsed_sec': elapsed,
            'epoch_time_sec': epoch_time,
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        self.epoch_metrics['epoch'].append(epoch)

        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("epoch/train_loss", train_loss, epoch)
            if val_loss is not None:
                self.tb_writer.add_scalar("epoch/val_loss", val_loss, epoch)
            if lr is not None:
                self.tb_writer.add_scalar("epoch/lr", lr, epoch)
            if extra_metrics:
                for k, v in extra_metrics.items():
                    self.tb_writer.add_scalar(f"epoch/{k}", v, epoch)

        # wandb 日志
        if self.use_wandb:
            import wandb
            log_dict = {
                "epoch": epoch,
                "epoch/train_loss": train_loss,
                "epoch/time": epoch_time,
            }
            if val_loss is not None:
                log_dict["epoch/val_loss"] = val_loss
            if lr is not None:
                log_dict["epoch/lr"] = lr
            if extra_metrics:
                for k, v in extra_metrics.items():
                    log_dict[f"epoch/{k}"] = v
            wandb.log(log_dict)

        # CSV
        self._write_csv(metrics)

        # 控制台输出
        self._print_epoch(epoch, train_loss, val_loss, elapsed, epoch_time, extra_metrics)

    def _print_epoch(self, epoch: int, train_loss: float,
                     val_loss: Optional[float], elapsed: float,
                     epoch_time: float, extra: Optional[Dict]):
        """格式化控制台输出"""
        msg = (f"[Epoch {epoch:3d}] train={train_loss:.6f}")
        if val_loss is not None:
            msg += f" | val={val_loss:.6f}"
        msg += f" | time={epoch_time:.1f}s | total={elapsed:.0f}s"
        if extra:
            for k, v in extra.items():
                msg += f" | {k}={v:.4f}"
        print(msg)

    def _write_csv(self, metrics: Dict):
        """写入 CSV"""
        import csv
        if not self.csv_initialized:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=metrics.keys())
            self.csv_writer.writeheader()
            self.csv_initialized = True
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()

    def save_config(self, config: dict):
        """保存训练配置"""
        config_path = os.path.join(self.experiment_dir, "config.yaml")
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        print(f"  配置已保存: {config_path}")

    def save_model(self, model, optimizer, epoch: int, path: Optional[str] = None):
        """保存 checkpoint"""
        if path is None:
            path = os.path.join(self.experiment_dir, f"checkpoint_epoch_{epoch}.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print(f"  模型已保存: {path}")

        # wandb 保存 artifact
        if self.use_wandb:
            import wandb
            artifact = wandb.Artifact(
                name=f"model-{self.experiment_name}",
                type="model",
                metadata={"epoch": epoch}
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def load_checkpoint(self, model, optimizer, path: str) -> int:
        """加载 checkpoint"""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def log_image(self, image_path: str, caption: Optional[str] = None):
        """记录图像到 wandb"""
        if self.use_wandb:
            import wandb
            wandb.log({"images": wandb.Image(image_path, caption=caption)})

    def log_images(self, images: Dict[str, Any]):
        """批量记录图像到 wandb"""
        if self.use_wandb:
            import wandb
            wandb.log(images)

    def close(self):
        """清理"""
        if self.csv_file:
            self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()
        print(f"日志已保存: {self.experiment_dir}")
