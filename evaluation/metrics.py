"""
评估指标
========
EIT 重建质量的综合评价指标体系。
"""

import numpy as np
import torch
from typing import Union, Optional

ArrayLike = Union[np.ndarray, torch.Tensor]


def relative_error(pred: ArrayLike, target: ArrayLike) -> float:
    """
    相对误差 RE = ||pred - target|| / ||target||

    范围: [0, +∞) — 越接近 0 越好
    """
    if isinstance(pred, torch.Tensor):
        error = torch.norm(pred - target, dim=-1)
        norm = torch.norm(target, dim=-1) + 1e-8
        return (error / norm).mean().item()
    else:
        error = np.linalg.norm(pred - target, axis=-1)
        norm = np.linalg.norm(target, axis=-1) + 1e-8
        return float(np.mean(error / norm))


def correlation_coefficient(pred: ArrayLike, target: ArrayLike) -> float:
    """
    相关系数 CC

    范围: [-1, 1] — 越接近 1 越好
    """
    if isinstance(pred, torch.Tensor):
        p = pred - pred.mean(dim=-1, keepdim=True)
        t = target - target.mean(dim=-1, keepdim=True)
        cov = (p * t).sum(dim=-1)
        std_p = torch.sqrt((p ** 2).sum(dim=-1) + 1e-8)
        std_t = torch.sqrt((t ** 2).sum(dim=-1) + 1e-8)
        return (cov / (std_p * std_t + 1e-8)).mean().item()
    else:
        p = pred - pred.mean(axis=-1, keepdims=True)
        t = target - target.mean(axis=-1, keepdims=True)
        cov = (p * t).sum(axis=-1)
        std_p = np.sqrt((p ** 2).sum(axis=-1) + 1e-8)
        std_t = np.sqrt((t ** 2).sum(axis=-1) + 1e-8)
        return float(np.mean(cov / (std_p * std_t + 1e-8)))


def peak_snr(pred: ArrayLike, target: ArrayLike) -> float:
    """
    峰值信噪比 PSNR (dB)

    范围: [0, +∞) — 越大越好
    """
    if isinstance(pred, torch.Tensor):
        mse = torch.mean((pred - target) ** 2, dim=-1)
        max_val = target.max(dim=-1).values - target.min(dim=-1).values
        psnr = 20 * torch.log10(max_val / (torch.sqrt(mse) + 1e-8))
        return psnr.mean().item()
    else:
        mse = np.mean((pred - target) ** 2, axis=-1)
        max_val = target.max(axis=-1) - target.min(axis=-1)
        psnr = 20 * np.log10(max_val / (np.sqrt(mse) + 1e-8))
        return float(np.mean(psnr))


def structural_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    """
    结构相似性 SSIM
    需要将单元级数据先映射到图像网格

    范围: [-1, 1] — 越接近 1 越好
    """
    from skimage.metrics import structural_similarity as ssim

    # 映射到图像
    pred_img = _map_to_image(pred)
    target_img = _map_to_image(target)

    ssim_vals = []
    for i in range(min(pred_img.shape[0], 8)):  # 取前8个
        val = ssim(pred_img[i], target_img[i],
                   data_range=target_img[i].max() - target_img[i].min())
        ssim_vals.append(val)

    return float(np.mean(ssim_vals))


def _map_to_image(sigma: np.ndarray,
                  grid_size: int = 64) -> np.ndarray:
    """
    将单元级电导率映射到规整图像网格
    用于 SSIM 等基于图像的指标计算

    参数:
        sigma: (B, n_elems) 或 (n_elems,)
        grid_size: 图像分辨率

    返回:
        img: (B, grid_size, grid_size) 或 (grid_size, grid_size)
    """
    from scipy.interpolate import griddata

    if sigma.ndim == 1:
        sigma = sigma[np.newaxis, :]

    # 需要网格节点坐标（这里假设已通过外部方式提供）
    # 简化：用 2D 插值到规整网格
    # 注意: 实际使用时需要 element_centers 作为插值基点
    return sigma  # 暂用 sigma 替代，实际需插值


def detection_accuracy(mask_pred: np.ndarray, mask_gt: np.ndarray) -> dict:
    """
    根检测精度
    将重建电导率二值化后与真实根位置对比

    返回:
        {'precision': ..., 'recall': ..., 'f1': ..., 'iou': ...}
    """
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.cpu().numpy()
    if isinstance(mask_gt, torch.Tensor):
        mask_gt = mask_gt.cpu().numpy()

    # 二值化
    binary_pred = (mask_pred > 0.5).astype(np.float32)
    binary_gt = (mask_gt > 0.5).astype(np.float32)

    tp = (binary_pred * binary_gt).sum()
    fp = (binary_pred * (1 - binary_gt)).sum()
    fn = ((1 - binary_pred) * binary_gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
    }


def compute_all_metrics(pred: ArrayLike, target: ArrayLike,
                        mask_pred: Optional[ArrayLike] = None,
                        mask_gt: Optional[ArrayLike] = None) -> dict:
    """计算所有指标"""
    metrics = {
        'RE': relative_error(pred, target),
        'CC': correlation_coefficient(pred, target),
        'PSNR': peak_snr(pred, target),
    }

    if mask_pred is not None and mask_gt is not None:
        det_metrics = detection_accuracy(mask_pred, mask_gt)
        metrics.update(det_metrics)

    return metrics
