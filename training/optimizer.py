"""
优化器与调度器配置
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math


def build_optimizer(model: nn.Module, config: dict) -> Optimizer:
    """
    构建优化器

    参数:
        model: 模型
        config: 训练配置字典

    返回:
        optimizer
    """
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    # 不同层组用不同学习率（可选）
    # 编码器层用较低学习率，解码器较高
    if 'layer_wise_lr' in config['training'] and config['training']['layer_wise_lr']:
        params = []
        for name, param in model.named_parameters():
            if 'encoder' in name:
                params.append({'params': param, 'lr': lr * 0.5})
            elif 'backbone' in name:
                params.append({'params': param, 'lr': lr * 0.8})
            else:
                params.append({'params': param, 'lr': lr})
        optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

    return optimizer


def build_scheduler(optimizer: Optimizer, config: dict,
                    steps_per_epoch: int) -> _LRScheduler:
    """
    构建学习率调度器

    支持: cosine / step / plateau
    """
    scheduler_cfg = config['training']
    sched_type = scheduler_cfg.get('lr_scheduler', 'cosine')
    epochs = scheduler_cfg['epochs']

    if sched_type == 'cosine':
        # 余弦退火 + warmup
        warmup_epochs = scheduler_cfg.get('warmup_epochs', 10)
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched_type == 'step':
        step_size = epochs // 3
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=0.1
        )

    elif sched_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10,
            min_lr=1e-6, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")

    return scheduler
