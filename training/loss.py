"""
损失函数设计
============
无监督训练中使用的所有损失函数。

核心损失:
  1. L_m: 测量一致性损失（物理约束核心）
  2. L_tv: 全变差正则化（抑制伪影）
  3. L_freq: 频率交叉一致性（多频约束）
  4. L_blc: BLC校正损失
  5. L_smooth: 平滑度损失

总损失:
    L_total = λ_m * L_m + λ_tv * L_tv + λ_freq * L_freq
              + λ_blc * L_blc + λ_smooth * L_smooth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Callable


class MeasurementConsistencyLoss(nn.Module):
    """
    测量一致性损失 L_m
    =====================
    无监督学习的核心物理约束:
    ||F(σ_pred) - V_measured||²

    需要前向算子 F(·) 来计算预测电压。
    有两种实现方式：
      A. 直接调用 pyEIT 正问题求解器（精确但慢）
      B. 使用预计算的雅可比矩阵做线性近似（快但近似）
    """

    def __init__(self, use_jacobian: bool = True,
                 jacobian: Optional[torch.Tensor] = None,
                 frequencies: Optional[list] = None,
                 forward_solver: Optional[Callable] = None,
                 sigma_ref_value: float = 0.01):
        """
        参数:
            use_jacobian: True=雅可比线性近似, False=调用完整正解
            jacobian: (n_freq, n_meas, n_elems) 预计算雅可比
            frequencies: 频率列表
            forward_solver: 正问题求解函数 (sigma) → voltage
            sigma_ref_value: 参考电导率（土壤背景），默认 0.01 S/m
        """
        super().__init__()
        self.use_jacobian = use_jacobian
        self.forward_solver = forward_solver
        self.sigma_ref_value = sigma_ref_value

        if jacobian is not None:
            self.register_buffer('jacobian', jacobian)
        else:
            self.jacobian = None

    def forward(self, sigma_pred: torch.Tensor,
                voltages_measured: torch.Tensor,
                sigma_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算测量一致性损失

        参数:
            sigma_pred: (B, n_elems) 预测的电导率
            voltages_measured: (B, n_freq, n_meas) 测量的边界电压
            sigma_ref: (B, n_elems) 参考电导率（可选，用于差分EIT）

        返回:
            loss: 标量损失值
        """
        B = sigma_pred.shape[0]

        if self.use_jacobian and self.jacobian is not None:
            # 方法A: 雅可比线性近似（快速训练用）
            # V_pred[f,m] = Σ_e J[f,m,e] · (σ_pred[e] - σ_ref[e])
            J = self.jacobian  # (n_freq, n_meas, n_elems)
            J = J.unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_freq, n_meas, n_elems)

            if sigma_ref is None:
                # ★ 修复: 雅可比在土壤背景 σ=0.01 处计算，参考必须用相同的背景值
                sigma_ref = torch.full_like(sigma_pred, self.sigma_ref_value)

            # ★ 修复: matmul 收缩维度是 n_elems × n_elems
            #   J: (B, n_freq, n_meas, n_elems)
            #   δσ: (B, 1, n_elems, 1)  ← 必须让 n_elems 在倒数第二维
            #   J @ δσ → (B, n_freq, n_meas, 1) → squeeze → (B, n_freq, n_meas)
            delta_sigma = sigma_pred - sigma_ref  # (B, n_elems)
            delta_sigma = delta_sigma.unsqueeze(1).unsqueeze(-1)  # (B, 1, n_elems, 1) ✅
            V_pred = (J @ delta_sigma).squeeze(-1)  # (B, n_freq, n_meas)

        elif self.forward_solver is not None:
            # 方法B: 使用完整 pyEIT 正问题求解器（验证用，精确但慢）
            # 注意: 训练时不要用这条路径——每个 step 调 pyEIT 会极慢
            V_pred_list = []
            for b in range(B):
                sigma_b = sigma_pred[b].detach().cpu().numpy()
                V_b = self.forward_solver(sigma_b)  # (n_freq, n_meas)
                V_pred_list.append(torch.from_numpy(V_b).to(sigma_pred.device))
            V_pred = torch.stack(V_pred_list, dim=0)
        else:
            raise ValueError("需要 jacobian 或 forward_solver")

        # 均方误差
        loss = F.mse_loss(V_pred, voltages_measured)
        return loss


class TVRegularizationLoss(nn.Module):
    """
    全变差正则化损失 L_tv
    =======================
    在网格上计算电导率梯度的 ℓ1 范数。
    抑制伪影，保持边缘。

    使用单元中心的坐标计算空间梯度。
    """

    def __init__(self, element_centers: torch.Tensor,
                 mesh_elements: torch.Tensor, mesh_nodes: torch.Tensor):
        """
        参数:
            element_centers: (n_elems, 2) 单元中心坐标
            mesh_elements: (n_elems, 3) 单元节点索引
            mesh_nodes: (n_nodes, 2) 节点坐标
        """
        super().__init__()
        self.element_centers = element_centers
        self.mesh_elements = mesh_elements
        self.mesh_nodes = mesh_nodes

        # 预计算相邻单元对的边权重（用于TV）
        self.edge_weights = self._compute_edge_weights()

    def _compute_edge_weights(self) -> torch.Tensor:
        """计算相邻单元的边权重（基于K近邻）"""
        centers = self.element_centers.cpu().numpy()
        n_elems = centers.shape[0]

        # 只取前2维坐标（2D网格）
        if centers.shape[1] > 2:
            centers = centers[:, :2]

        # 使用 KNN 找相邻单元（比 Delaunay 更鲁棒）
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(8, n_elems), algorithm='kd_tree')
        nn.fit(centers)
        distances, indices = nn.kneighbors(centers)  # (n_elems, k)

        # 构建边列表（每个单元与最近的 k-1 个邻居的边）
        edges = set()
        for i in range(n_elems):
            for j in indices[i, 1:]:  # 跳过自己（索引0）
                edge = (min(i, int(j)), max(i, int(j)))
                edges.add(edge)

        edges = list(edges)
        edge_idx = torch.tensor(edges, dtype=torch.long).T  # (2, n_edges)
        print(f"  TV: {n_elems} 单元, {edge_idx.shape[1]} 条边")
        return edge_idx

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        计算 TV 损失

        参数:
            sigma: (B, n_elems) 电导率分布

        返回:
            loss: 标量 TV 损失
        """
        if self.edge_weights.shape[1] == 0:
            return torch.tensor(0.0, device=sigma.device)

        edge_idx = self.edge_weights.to(sigma.device)

        # 邻接单元的电导率差
        sigma_diff = sigma[:, edge_idx[0]] - sigma[:, edge_idx[1]]  # (B, n_edges)

        # ℓ1 范数（各向异性TV）
        tv_loss = torch.abs(sigma_diff).mean()

        return tv_loss


class FrequencyCrossConsistencyLoss(nn.Module):
    """
    频率交叉一致性损失 L_freq
    ==========================
    多频率重建结果应该在空间结构上一致。
    鼓励不同频率的隐特征产生相似的结构。
    """

    def __init__(self):
        super().__init__()

    def forward(self, freq_weights: torch.Tensor,
                sigma_pred: torch.Tensor,
                base_map: torch.Tensor) -> torch.Tensor:
        """
        如果不同频率关注的不同区域，则 loss 小；
        如果频率权重分布随机，则 loss 大。

        参数:
            freq_weights: (B, n_freq) 频率注意力权重
            sigma_pred: (B, n_elems) 预测电导率
            base_map: (B, n_elems) 基础层估计
        """
        loss = 0.0

        # 1. 频率权重的熵 → 鼓励集中注意力
        if freq_weights is not None:
            entropy = - (freq_weights * torch.log(freq_weights + 1e-8)).sum(dim=-1).mean()
            loss += -0.01 * entropy  # 最大化熵（避免坍缩到单一频率）

        # 2. 基础层与预测的结构一致性
        if base_map is not None:
            # 归一化后计算余弦相似度
            s_norm = (sigma_pred - sigma_pred.mean(dim=-1, keepdim=True)) / \
                     (sigma_pred.std(dim=-1, keepdim=True) + 1e-8)
            b_norm = (base_map - base_map.mean(dim=-1, keepdim=True)) / \
                     (base_map.std(dim=-1, keepdim=True) + 1e-8)
            struct_loss = 1 - F.cosine_similarity(s_norm, b_norm, dim=-1).mean()
            loss += 0.1 * struct_loss

        return loss


class BLCCorrectionLoss(nn.Module):
    """
    BLC 校正损失 L_blc
    =====================
    约束 BLC 校正值不要过大，避免扭曲重建。
    """

    def __init__(self):
        super().__init__()

    def forward(self, blc_gates: torch.Tensor) -> torch.Tensor:
        """
        鼓励门控值适度（接近 0.5），但不过度校正

        参数:
            blc_gates: (B, n_freq) BLC 门控值
        """
        # 门控值偏离 0.5 的惩罚
        gate_penalty = (blc_gates - 0.5).pow(2).mean()
        return 0.1 * gate_penalty


class SmoothnessLoss(nn.Module):
    """
    平滑度损失 L_smooth
    =====================
    鼓励重建结果在空间上平滑（物理合理性）
    """

    def __init__(self):
        super().__init__()

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        参数:
            sigma: (B, n_elems)
        """
        # 简单版本：相邻重建值的差
        # 适用于已排序的单元
        diff = sigma[:, 1:] - sigma[:, :-1]
        smooth_loss = diff.pow(2).mean()
        return smooth_loss


class AdaptiveLossWeighter(nn.Module):
    """
    自适应损失权重
    使用可学习的权重平衡多个损失项
    参考: Multi-Task Learning Using Uncertainty to Weigh Losses
    """

    def __init__(self, n_losses: int = 5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        参数:
            losses: 包含各个损失的 dict

        返回:
            加权总损失
        """
        total = 0.0
        for i, (name, loss) in enumerate(losses.items()):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i] * 0.5
        return total
