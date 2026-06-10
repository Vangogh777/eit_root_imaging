"""
重建结果可视化工具
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional, List


class EITVisualizer:
    """
    EIT 重建结果可视化

    支持:
        - 单样本对比 (GT vs Pred vs Error)
        - 批量对比面板
        - 频率注意力可视化
        - 基础层可视化
        - GIF 时间序列
    """

    def __init__(self, element_centers: np.ndarray,
                 mesh_nodes: Optional[np.ndarray] = None,
                 mesh_elements: Optional[np.ndarray] = None,
                 domain_radius: float = 0.1):
        self.centers = element_centers
        self.nodes = mesh_nodes
        self.elements = mesh_elements
        self.domain_radius = domain_radius

    def plot_comparison(self, sigma_gt: np.ndarray, sigma_pred: np.ndarray,
                        title: str = "", save_path: Optional[str] = None,
                        vmin: float = 0.005, vmax: float = 0.055):
        """GT vs Pred vs Error 对比"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        titles = ['Ground Truth', 'Prediction', '|Error|']
        data = [sigma_gt, sigma_pred, np.abs(sigma_pred - sigma_gt)]

        for ax, d, t in zip(axes, data, titles):
            if t == '|Error|':
                sc = ax.scatter(self.centers[:, 0], self.centers[:, 1],
                                c=d, s=8, cmap='hot',
                                norm=Normalize(0, 0.008))
            else:
                sc = ax.scatter(self.centers[:, 0], self.centers[:, 1],
                                c=d, s=8, cmap='viridis',
                                vmin=vmin, vmax=vmax)
            ax.set_title(t)
            ax.set_aspect('equal')
            plt.colorbar(sc, ax=ax, fraction=0.046)

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_batch(self, sigma_gts: np.ndarray, sigma_preds: np.ndarray,
                   n_samples: int = 4, save_path: Optional[str] = None):
        """批量对比"""
        n = min(n_samples, len(sigma_gts))
        fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))

        for i in range(n):
            for j, (data, title) in enumerate([
                (sigma_gts[i], 'GT'),
                (sigma_preds[i], 'Pred'),
                (np.abs(sigma_preds[i] - sigma_gts[i]), 'Error')
            ]):
                ax = axes[i, j]
                vmin, vmax = (0, 0.008) if j == 2 else (0.005, 0.055)
                cmap = 'hot' if j == 2 else 'viridis'
                ax.scatter(self.centers[:, 0], self.centers[:, 1],
                           c=data, s=5, cmap=cmap,
                           norm=Normalize(vmin, vmax))
                ax.set_title(f"{title} [{i}]")
                ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_frequency_weights(self, weights: np.ndarray,
                                frequencies: List[float],
                                save_path: Optional[str] = None):
        """频率注意力权重可视化"""
        fig, ax = plt.subplots(figsize=(6, 4))

        freqs_khz = [f / 1000 for f in frequencies]
        ax.bar(range(len(freqs_khz)), weights.mean(axis=0))
        ax.set_xticks(range(len(freqs_khz)))
        ax.set_xticklabels([f"{f:.0f}" for f in freqs_khz])
        ax.set_xlabel("频率 (kHz)")
        ax.set_ylabel("平均注意力权重")
        ax.set_title("频率注意力分布")

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_blc_gates(self, gates: np.ndarray,
                        frequencies: List[float],
                        save_path: Optional[str] = None):
        """BLC 校正门控值可视化"""
        fig, ax = plt.subplots(figsize=(6, 4))

        freqs_khz = [f / 1000 for f in frequencies]
        ax.plot(freqs_khz, gates.mean(axis=0), 'o-', linewidth=2)
        ax.set_xlabel("频率 (kHz)")
        ax.set_ylabel("平均门控值")
        ax.set_title("BLC 校正门控")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
