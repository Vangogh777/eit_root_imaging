"""
PyTorch Dataset 封装
从 HDF5 文件加载 EIT 数据
支持:
  - 多频率电压 → 电导率 (有监督/无监督)
  - 随机电压掩码 (训练时增强)
  - 可选的雅可比矩阵预加载
"""

import os
import h5py
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple


class EITDataset(Dataset):
    """
    EIT 数据集

    每个样本:
        - voltages: (n_freq, n_measurements) 边界电压
        - sigmas: (n_elems,) 电导率分布 (ground truth，评估用)
        - masks: (n_elems,) 二值根位置掩码
        - noise_db: 标量

    用法:
        dataset = EITDataset("data/generated/eit_dataset.h5", split="train")
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, h5_path: str, split: str = "train",
                 load_sigmas: bool = True, load_masks: bool = True,
                 voltage_mask_ratio: float = 0.0,
                 jacobian_path: Optional[str] = None):
        """
        参数:
            h5_path: HDF5 数据集路径
            split: "train" / "val" / "test"
            load_sigmas: 是否加载真实电导率（评估时需要）
            load_masks: 是否加载根掩码
            voltage_mask_ratio: >0 时在训练中随机遮掩部分测量通道
            jacobian_path: 雅可比矩阵路径（可选，用于无监督物理约束）
        """
        self.h5_path = h5_path
        self.split = split
        self.load_sigmas = load_sigmas
        self.load_masks = load_masks
        self.voltage_mask_ratio = voltage_mask_ratio

        # 打开 HDF5 读取元数据
        with h5py.File(h5_path, 'r') as f:
            grp = f[split]
            self.n_samples = grp['voltages'].shape[0]
            self.n_freq = grp['voltages'].shape[1]
            self.n_meas = grp['voltages'].shape[2]
            self.n_elems = grp['sigmas'].shape[1]

            # 缓存网格元数据
            meta = f['metadata']
            self.mesh_nodes = meta['mesh_nodes'][:]
            self.mesh_elements = meta['mesh_elements'][:]
            self.frequencies = meta['frequencies'][:]

        # 加载雅可比矩阵
        self.jacobian = None
        if jacobian_path and os.path.exists(jacobian_path):
            self.jacobian = torch.from_numpy(np.load(jacobian_path)).float()

        # 缓存配置
        self._cache = {}

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """按需从 HDF5 读取（内存高效）"""
        # 使用缓存避免多次读取同一个 idx
        if idx in self._cache:
            return self._cache[idx]

        with h5py.File(self.h5_path, 'r') as f:
            grp = f[self.split]
            V = torch.from_numpy(grp['voltages'][idx]).float()
            sigma = torch.from_numpy(grp['sigmas'][idx]).float()
            mask = torch.from_numpy(grp['masks'][idx]).float()
            noise_db = torch.tensor(grp['noise_db'][idx], dtype=torch.float32)

        # --- 电压掩码增强 ---
        if self.voltage_mask_ratio > 0 and self.split == 'train':
            # 随机遮掩部分测量通道
            n_meas = V.shape[-1]
            n_mask = max(1, int(n_meas * self.voltage_mask_ratio))
            mask_indices = torch.randperm(n_meas)[:n_mask]
            V[..., mask_indices] = 0.0

        sample = {
            'voltages': V,              # (n_freq, n_meas)
            'sigmas': sigma,             # (n_elems,)
            'masks': mask,               # (n_elems,)
            'noise_db': noise_db,
            'idx': idx,
        }

        # 缓存（限制缓存大小）
        if len(self._cache) < 1000:
            self._cache[idx] = sample

        return sample


class EITDataModule:
    """
    统一数据管理
    封装 train/val/test 的 Dataset 和 DataLoader
    """

    def __init__(self, h5_path: str, batch_size: int = 32,
                 val_batch_size: Optional[int] = None,
                 num_workers: int = 4, prefetch_factor: int = 2,
                 voltage_mask_ratio: float = 0.3,
                 jacobian_path: Optional[str] = None):

        self.train_dataset = EITDataset(
            h5_path, split='train',
            voltage_mask_ratio=voltage_mask_ratio,
            jacobian_path=jacobian_path
        )
        self.val_dataset = EITDataset(
            h5_path, split='val',
            voltage_mask_ratio=0.0,
            jacobian_path=jacobian_path
        )
        self.test_dataset = EITDataset(
            h5_path, split='test',
            voltage_mask_ratio=0.0,
            jacobian_path=jacobian_path
        )

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size * 2
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == "__main__":
    # 快速测试
    dataset = EITDataset("data/generated/eit_dataset.h5", split="train")
    print(f"数据集大小: {len(dataset)}")
    sample = dataset[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} [{v.dtype}]")
        else:
            print(f"  {k}: {v}")
    print("[测试通过] EITDataset 工作正常")
