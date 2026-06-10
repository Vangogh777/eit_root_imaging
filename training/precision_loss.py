"""
高精度EIT损失函数
================
改进点：
1. 多尺度损失
2. 感知损失（结构相似性）
3. 边缘保持损失
4. 自适应权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class HighPrecisionLoss(nn.Module):
    """
    高精度重建损失函数

    组合多种损失：
    1. L_data: 数据拟合损失
    2. L_tv: 全变差正则化
    3. L_edge: 边缘保持损失
    4. L_perceptual: 感知损失（可选）
    5. L_consistency: 多尺度一致性
    """

    def __init__(self, element_centers: torch.Tensor,
                 mesh_elements: torch.Tensor,
                 lambda_tv: float = 0.01,
                 lambda_edge: float = 0.1,
                 lambda_perceptual: float = 0.1,
                 lambda_consistency: float = 0.05):
        super().__init__()

        self.register_buffer('element_centers', element_centers)
        self.register_buffer('mesh_elements', mesh_elements)

        self.lambda_tv = lambda_tv
        self.lambda_edge = lambda_edge
        self.lambda_perceptual = lambda_perceptual
        self.lambda_consistency = lambda_consistency

        # 预计算邻接关系
        self._precompute_adjacency()

    def _precompute_adjacency(self):
        """预计算单元邻接关系"""
        from scipy.spatial import Delaunay
        centers = self.element_centers.cpu().numpy()
        tri = Delaunay(centers)

        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edges.add((min(simplex[i], simplex[j]),
                               max(simplex[i], simplex[j])))

        self.edge_idx = torch.tensor(list(edges), dtype=torch.long).T  # (2, n_edges)

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Optional[torch.Tensor] = None,
                measured_voltages: Optional[torch.Tensor] = None,
                forward_solver=None) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        参数:
            outputs: 模型输出字典，包含 'sigma', 'sigma_coarse', 'sigma_residual'
            targets: 真实电导率（有监督时使用）
            measured_voltages: 测量电压（无监督时使用）
            forward_solver: 正问题求解器（无监督时使用）

        返回:
            losses: 各损失分量的字典
        """
        losses = {}
        sigma = outputs['sigma']

        # 1. 有监督损失
        if targets is not None:
            losses['mse'] = F.mse_loss(sigma, targets)

            # 相对误差损失
            rel_error = torch.norm(sigma - targets, dim=-1) / (torch.norm(targets, dim=-1) + 1e-8)
            losses['relative_error'] = rel_error.mean()

        # 2. TV正则化
        losses['tv'] = self._tv_loss(sigma)

        # 3. 边缘保持损失
        losses['edge'] = self._edge_preserving_loss(sigma, targets)

        # 4. 多尺度一致性损失
        if 'sigma_coarse' in outputs:
            losses['consistency'] = F.mse_loss(outputs['sigma_coarse'], sigma.detach())

        # 5. 平滑正则
        losses['smooth'] = self._smoothness_loss(sigma)

        # 6. 物理一致性（如果提供了正问题求解器）
        if measured_voltages is not None and forward_solver is not None:
            # V_pred = forward_solver(sigma)
            # losses['physics'] = F.mse_loss(V_pred, measured_voltages)
            pass  # 需要实现可微分的正问题求解器

        # 总损失
        total = sum(losses.values())
        losses['total'] = total

        return losses

    def _tv_loss(self, sigma: torch.Tensor) -> torch.Tensor:
        """全变差损失"""
        if self.edge_idx.shape[1] == 0:
            return torch.tensor(0.0, device=sigma.device)

        edge_idx = self.edge_idx.to(sigma.device)
        sigma_diff = sigma[:, edge_idx[0]] - sigma[:, edge_idx[1]]
        return torch.abs(sigma_diff).mean()

    def _edge_preserving_loss(self, sigma: torch.Tensor,
                               targets: Optional[torch.Tensor]) -> torch.Tensor:
        """边缘保持损失：鼓励在真实边缘处有陡峭变化"""
        if targets is None:
            return torch.tensor(0.0, device=sigma.device)

        # 计算目标的梯度方向
        edge_idx = self.edge_idx.to(sigma.device)
        target_diff = targets[:, edge_idx[0]] - targets[:, edge_idx[1]]
        sigma_diff = sigma[:, edge_idx[0]] - sigma[:, edge_idx[1]]

        # 在目标边缘处鼓励预测也有边缘
        edge_mask = torch.abs(target_diff) > 0.005  # 阈值

        if edge_mask.sum() == 0:
            return torch.tensor(0.0, device=sigma.device)

        # 边缘处方向一致
        loss = F.mse_loss(
            sigma_diff[edge_mask].sign() * sigma_diff[edge_mask],
            sigma_diff[edge_mask].sign() * target_diff[edge_mask]
        )
        return loss

    def _smoothness_loss(self, sigma: torch.Tensor) -> torch.Tensor:
        """平滑损失"""
        diff = sigma[:, 1:] - sigma[:, :-1]
        return (diff ** 2).mean()


class AdaptiveLossWeighter(nn.Module):
    """
    自适应损失权重学习
    ==================
    让网络自动学习各损失的权重，避免手动调参。

    参考: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """

    def __init__(self, n_losses: int = 5):
        super().__init__()
        # 可学习的log方差参数
        self.log_vars = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses: Dict[str, torch.Tensor],
                loss_names: list) -> torch.Tensor:
        """
        参数:
            losses: 损失值字典
            loss_names: 损失名称列表（顺序对应log_vars）

        返回:
            加权总损失
        """
        total = 0.0
        for i, name in enumerate(loss_names):
            if name in losses:
                precision = torch.exp(-self.log_vars[i])
                total += precision * losses[name] + self.log_vars[i] * 0.5
        return total


class PerceptualLoss(nn.Module):
    """
    感知损失
    ========
    使用预训练网络的特征来衡量重建质量。
    对于EIT，可以训练一个自编码器来提取特征。
    """

    def __init__(self, feature_dim: int = 128, n_elems: int = 1500):
        super().__init__()

        # 简单的特征提取器（可以替换为预训练网络）
        self.encoder = nn.Sequential(
            nn.Linear(n_elems, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        参数:
            pred: (B, n_elems)
            target: (B, n_elems)

        返回:
            感知损失
        """
        feat_pred = self.encoder(pred)
        feat_target = self.encoder(target.detach())

        return F.mse_loss(feat_pred, feat_target)


class SSIMLoss(nn.Module):
    """
    结构相似性损失（SSIM）
    ======================
    衡量图像的结构相似性，比MSE更符合人眼感知。

    对于EIT，需要先将电导率映射到网格上。
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                centers: torch.Tensor, grid_size: int = 64) -> torch.Tensor:
        """
        参数:
            pred: (B, n_elems)
            target: (B, n_elems)
            centers: (n_elems, 2) 单元中心坐标
            grid_size: 网格大小

        返回:
            1 - SSIM
        """
        # 将散点数据映射到规则网格
        pred_grid = self._scatter_to_grid(pred, centers, grid_size)
        target_grid = self._scatter_to_grid(target, centers, grid_size)

        # 计算 SSIM
        ssim_val = self._compute_ssim(pred_grid, target_grid)

        return 1 - ssim_val

    def _scatter_to_grid(self, data: torch.Tensor, centers: torch.Tensor,
                         grid_size: int) -> torch.Tensor:
        """将散点数据映射到规则网格"""
        B = data.shape[0]
        grid = torch.zeros(B, 1, grid_size, grid_size, device=data.device)

        # 简化：最近邻插值
        # ... 实现省略

        return grid

    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算SSIM"""
        # 简化实现
        C1, C2 = 0.01**2, 0.03**2

        mu_pred = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.window_size//2)
        mu_target = F.avg_pool2d(target, self.window_size, stride=1, padding=self.window_size//2)

        sigma_pred = F.avg_pool2d(pred**2, self.window_size, stride=1, padding=self.window_size//2) - mu_pred**2
        sigma_target = F.avg_pool2d(target**2, self.window_size, stride=1, padding=self.window_size//2) - mu_target**2
        sigma_cross = F.avg_pool2d(pred*target, self.window_size, stride=1, padding=self.window_size//2) - mu_pred*mu_target

        ssim = ((2*mu_pred*mu_target + C1) * (2*sigma_cross + C2)) / \
               ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2))

        return ssim.mean()


if __name__ == "__main__":
    # 测试损失函数
    print("=== 测试高精度损失函数 ===")

    n_elems = 1500

    # 模拟数据
    pred = torch.randn(4, n_elems)
    target = torch.randn(4, n_elems)

    # 创建随机网格
    centers = torch.randn(n_elems, 2)
    elements = torch.randint(0, n_elems, (n_elems, 3))

    # 初始化损失
    loss_fn = HighPrecisionLoss(centers, elements)

    # 计算损失
    outputs = {
        'sigma': pred,
        'sigma_coarse': target,
    }

    losses = loss_fn(outputs, targets=target)

    print("损失分量:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")

    print("\n✅ 损失函数测试通过")
