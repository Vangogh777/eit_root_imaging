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

    提供三种模式：
      A. 'jacobian' — 雅可比线性近似（最快，但偏离 σ_ref 后不准）
      B. 'full_fem'  — 完整 FEM 正解（准确，梯度通过 Jacobian 近似回传）
      C. 'hybrid'    — 混合：每 N 步用一次 full_fem，其余用 jacobian
    """

    def __init__(self, mode: str = 'full_fem',
                 jacobian: Optional[torch.Tensor] = None,
                 forward_solver: Optional[Callable] = None,
                 sigma_ref_value: float = 0.01,
                 fem_interval: int = 5,
                 fem_subset_size: int = 0):
        """
        参数:
            mode: 'jacobian' | 'full_fem' | 'hybrid'
            jacobian: (n_freq, n_meas, n_elems) 预计算雅可比
            forward_solver: 正问题求解函数 (sigma_np) → (n_freq, n_meas) np数组
            sigma_ref_value: 参考电导率（土壤背景），默认 0.01 S/m
            fem_interval: 每 N 步执行一次完整 FEM（N=1 每步都跑，N=5 每5步跑一次）
            fem_subset_size: >0 时每次 FEM 只算前 N 个样本（其余用雅可比近似），加速训练
        """
        super().__init__()
        assert mode in ('jacobian', 'full_fem', 'hybrid'), f"未知模式: {mode}"
        self.mode = mode
        self.forward_solver = forward_solver
        self.sigma_ref_value = sigma_ref_value
        self.fem_interval = fem_interval
        self.fem_subset_size = fem_subset_size
        self._fem_step = 0
        self._cached_V_full = None
        self._last_real_loss = 0.0

        if jacobian is not None:
            self.register_buffer('jacobian', jacobian)
        else:
            self.jacobian = None

    def _jacobian_forward(self, sigma_pred: torch.Tensor,
                          sigma_ref: torch.Tensor) -> torch.Tensor:
        """雅可比线性近似前向（可微分）"""
        B = sigma_pred.shape[0]
        J = self.jacobian.unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_freq, n_meas, n_elems)
        delta = (sigma_pred - sigma_ref).unsqueeze(1).unsqueeze(-1)  # (B, 1, n_elems, 1)
        return (J @ delta).squeeze(-1)  # (B, n_freq, n_meas)

    def forward(self, sigma_pred: torch.Tensor,
                voltages_measured: torch.Tensor,
                sigma_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算测量一致性损失

        参数:
            sigma_pred: (B, n_elems) 预测的电导率
            voltages_measured: (B, n_freq, n_meas) 测量的边界电压
            sigma_ref: (B, n_elems) 参考电导率（可选）

        返回:
            loss: 标量损失值
        """
        B = sigma_pred.shape[0]
        device = sigma_pred.device

        if sigma_ref is None:
            sigma_ref = torch.full_like(sigma_pred, self.sigma_ref_value)

        use_full = (self.mode == 'full_fem' and self.forward_solver is not None)

        if use_full:
            # ════════════════════════════════════════════
            # 方法A: 完整 FEM 正解（间隔 fem_interval 步执行一次）
            # ════════════════════════════════════════════
            run_fem = (self._fem_step % self.fem_interval == 0
                       or self._cached_V_full is None)

            if run_fem:
                # 完整 FEM 求解 → 缓存 V_full
                n_fem = min(B, self.fem_subset_size) if self.fem_subset_size > 0 else B
                sigma_np = sigma_pred.detach().cpu().numpy()
                V_full_list = []
                for b in range(n_fem):
                    V_b = self.forward_solver(sigma_np[b])  # (n_freq, n_meas)
                    V_full_list.append(V_b.astype(np.float32))
                self._cached_V_full = torch.from_numpy(np.stack(V_full_list)).to(device)
                self._last_real_loss = F.mse_loss(
                    self._cached_V_full, voltages_measured[:n_fem]).item()

            V_full = self._cached_V_full

            # Jacobian 预测（每次都有梯度）+ 双目标损失
            if self.jacobian is not None:
                V_jac = self._jacobian_forward(sigma_pred, sigma_ref)  # 可微分
                loss_main = F.mse_loss(V_jac, voltages_measured)
                # 辅助：让 V_jac 匹配缓存的 V_full（只对 FEM 算过的子集做比较）
                n_fem_cached = self._cached_V_full.shape[0]
                loss_aux = F.mse_loss(V_jac[:n_fem_cached], self._cached_V_full.detach())
                total_loss = loss_main + 0.1 * loss_aux
            else:
                total_loss = F.mse_loss(V_full, voltages_measured)

            self._fem_step += 1

        elif self.mode == 'hybrid' and self.forward_solver is not None:
            # 混合模式：每步都用 Jacobian，定期用 FEM 校正
            # 简单实现：full_fem 模式已足够
            V_jac = self._jacobian_forward(sigma_pred, sigma_ref)
            v_norm = voltages_measured.norm(dim=-1, keepdim=True).detach() + 1e-8
            total_loss = F.mse_loss(V_jac / v_norm, voltages_measured / v_norm)
        else:
            # ════════════════════════════════════════════
            # 方法B: 纯 Jacobian 线性近似（快速，备用）
            # ════════════════════════════════════════════
            V_jac = self._jacobian_forward(sigma_pred, sigma_ref)
            v_norm = voltages_measured.norm(dim=-1, keepdim=True).detach() + 1e-8
            total_loss = F.mse_loss(V_jac / v_norm, voltages_measured / v_norm)

        return total_loss


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

        # 使用 KDTree 找相邻单元（scipy 已安装，不需额外依赖）
        from scipy.spatial import cKDTree
        tree = cKDTree(centers)
        k = min(8, n_elems)
        distances, indices = tree.query(centers, k=k)  # (n_elems, k)

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


class SigmaDeviationLoss(nn.Module):
    """
    电导率偏离惩罚 L_dev
    =====================
    约束 σ_pred 不远离 σ_ref=0.01（Jacobian 的线性化点）。
    因为 Jacobian 线性近似只在 σ_ref 附近有效，
    偏离太大会导致物理意义完全失真。

    数学: L_dev = ||σ_pred - σ_ref||² / n_elems
    """

    def __init__(self, sigma_ref_value: float = 0.01):
        super().__init__()
        self.sigma_ref_value = sigma_ref_value

    def forward(self, sigma_pred: torch.Tensor) -> torch.Tensor:
        """
        参数:
            sigma_pred: (B, n_elems) 预测电导率

        返回:
            loss: 标量
        """
        sigma_ref = torch.full_like(sigma_pred, self.sigma_ref_value)
        diff = sigma_pred - sigma_ref
        loss = diff.pow(2).mean()  # 平均到每个单元
        return loss


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
