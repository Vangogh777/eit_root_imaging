"""
Data collation functions
"""
import torch
import numpy as np
from typing import Dict, List


def collate_eit(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    EIT 批处理函数
    将单个样本堆叠为 batch
    """
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == 'idx':
            out[k] = torch.tensor([b[k] for b in batch])
        elif k == 'noise_db':
            out[k] = torch.tensor([b[k] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
