"""
通用EIT重建模型 - 高精度版本
============================
改进点：
1. 移除根系特定假设，使用通用体模
2. 物理先验嵌入（Jacobian作为网络输入）
3. 多尺度重建架构
4. 更好的损失函数设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


class PhysicsInformedEIT(nn.Module):
    """
    物理信息增强的EIT重建网络

    核心思想：
    - 传统方法（如GREIT）使用Jacobian重建，精度有限
    - 深度学习方法纯数据驱动，缺乏物理约束
    - 本方法：将Jacobian作为先验输入网络，学习残差校正

    输入：
        - voltages: (B, n_freq, n_meas) 边界电压
        - jacobian: (B, n_meas, n_elems) 雅可比矩阵（可选）
    输出：
        - sigma: (B, n_elems) 电导率分布
    """

    def __init__(self, input_dim: int = 208, hidden_dim: int = 512,
                 n_frequencies: int = 6, n_elems: int = 1500,
                 use_jacobian_prior: bool = True):
        super().__init__()

        self.n_freq = n_frequencies
        self.n_elems = n_elems
        self.use_jacobian_prior = use_jacobian_prior

        # ============ 1. 电压编码器 ============
        self.voltage_encoder = nn.Sequential(
            nn.Linear(input_dim * n_frequencies, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )

        # ============ 2. 物理先验编码器（Jacobian） ============
        if use_jacobian_prior:
            # 将Jacobian投影到低维空间
            self.jacobian_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
            )
            # 融合电压特征和Jacobian特征
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        # ============ 3. 多尺度重建器 ============
        # 粗尺度：直接从特征重建
        self.coarse_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_elems),
        )

        # 细尺度：残差校正
        self.fine_reconstructor = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )
        self.residual_head = nn.Linear(hidden_dim, n_elems)

        # ============ 4. 置信度估计（可选） ============
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_elems),
            nn.Softplus(),  # 输出标准差（正数）
        )

    def forward(self, voltages: torch.Tensor,
                jacobian: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        参数:
            voltages: (B, n_freq, n_meas) 边界电压
            jacobian: (B, n_meas, n_elems) 雅可比矩阵（可选）

        返回:
            dict: {
                'sigma': (B, n_elems) 重建电导率,
                'sigma_coarse': (B, n_elems) 粗尺度重建,
                'sigma_residual': (B, n_elems) 残差校正,
                'uncertainty': (B, n_elems) 不确定性估计,
            }
        """
        B = voltages.shape[0]

        # 1. 编码电压
        v_flat = voltages.view(B, -1)  # (B, n_freq * n_meas)
        h = self.voltage_encoder(v_flat)  # (B, hidden_dim)

        # 2. 融合物理先验
        if self.use_jacobian_prior and jacobian is not None:
            # 对Jacobian做池化得到全局特征
            # jacobian: (B, n_meas, n_elems) -> (B, n_meas) -> (B, hidden_dim//2)
            J_pooled = jacobian.mean(dim=-1)  # (B, n_meas)
            j_feat = self.jacobian_encoder(J_pooled)  # (B, hidden_dim//2)
            h = self.fusion(torch.cat([h, j_feat], dim=-1))  # (B, hidden_dim)

        # 3. 粗尺度重建
        sigma_coarse = self.coarse_reconstructor(h)  # (B, n_elems)

        # 4. 细尺度残差校正
        h_fine = self.fine_reconstructor(h)
        sigma_residual = self.residual_head(h_fine)  # (B, n_elems)

        # 5. 最终输出 = 粗尺度 + 细尺度残差
        sigma = sigma_coarse + sigma_residual

        # 6. 不确定性估计
        uncertainty = self.uncertainty_head(h)  # (B, n_elems)

        return {
            'sigma': sigma,
            'sigma_coarse': sigma_coarse,
            'sigma_residual': sigma_residual,
            'uncertainty': uncertainty,
        }


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DeepImagePriorEIT(nn.Module):
    """
    Deep Image Prior for EIT
    ========================
    不需要训练数据！在推理时直接优化网络权重。

    原理：
    - 网络结构本身就蕴含了图像先验
    - 给定随机噪声输入，通过优化使网络输出满足测量约束
    - 适合单样本重建，精度高但速度慢
    """

    def __init__(self, n_elems: int = 1500, hidden_dim: int = 256,
                 n_scales: int = 3):
        super().__init__()

        self.n_scales = n_scales

        # 多尺度解码器
        self.decoders = nn.ModuleList()
        for s in range(n_scales):
            scale_dim = hidden_dim // (2 ** s)
            self.decoders.append(nn.Sequential(
                nn.Linear(hidden_dim, scale_dim),
                nn.LayerNorm(scale_dim),
                nn.GELU(),
                nn.Linear(scale_dim, n_elems),
            ))

        # 上采样和下采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.AvgPool1d(2)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        参数:
            noise: (B, hidden_dim) 随机噪声/固定噪声

        返回:
            sigma: (B, n_elems)
        """
        outputs = []
        h = noise

        for decoder in self.decoders:
            out = decoder(h)  # (B, n_elems)
            outputs.append(out)

        # 多尺度融合
        sigma = sum(outputs) / len(outputs)
        return sigma


class IterativeRefinementEIT(nn.Module):
    """
    迭代细化EIT重建
    ================
    多步迭代，每步利用物理约束更新重建结果。

    类似于传统迭代算法（如Gauss-Newton），但用神经网络学习更新步。
    """

    def __init__(self, input_dim: int = 208, hidden_dim: int = 256,
                 n_frequencies: int = 6, n_elems: int = 1500,
                 n_iterations: int = 5):
        super().__init__()

        self.n_iterations = n_iterations

        # 初始重建网络
        self.initial_net = nn.Sequential(
            nn.Linear(input_dim * n_frequencies, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_elems),
        )

        # 迭代更新网络
        self.update_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * n_frequencies + n_elems + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_elems),
            ) for _ in range(n_iterations)
        ])

    def forward(self, voltages: torch.Tensor,
                forward_solver=None) -> torch.Tensor:
        """
        参数:
            voltages: (B, n_freq, n_meas)
            forward_solver: 可调用的正问题求解器（用于计算残差）

        返回:
            sigma: (B, n_elems)
        """
        B = voltages.shape[0]
        v_flat = voltages.view(B, -1)

        # 初始重建
        sigma = self.initial_net(v_flat)

        # 迭代更新
        for i, update_net in enumerate(self.update_nets):
            # 计算当前残差（如果提供了正问题求解器）
            if forward_solver is not None:
                with torch.no_grad():
                    v_pred = forward_solver(sigma)
                    residual = (voltages - v_pred).view(B, -1).mean(dim=-1, keepdim=True)
            else:
                residual = torch.zeros(B, 1, device=voltages.device)

            # 拼接输入：电压 + 当前重建 + 残差
            update_input = torch.cat([v_flat, sigma, residual], dim=-1)
            delta = update_net(update_input)

            # 更新重建
            sigma = sigma + 0.1 * delta  # 小步长更新

        return sigma


# ============ 通用体模生成器 ============

class UniversalPhantomGenerator:
    """
    通用仿真体模生成器
    ==================
    生成各种形状的电导率分布，用于训练通用重建模型。
    """

    def __init__(self, mesh_nodes: np.ndarray, mesh_elements: np.ndarray,
                 domain_radius: float = 0.1,
                 sigma_background: float = 0.01,
                 sigma_inclusion: float = 0.05):
        self.nodes = mesh_nodes
        self.elements = mesh_elements
        self.centers = np.mean(mesh_nodes[mesh_elements], axis=1)
        self.n_elems = len(mesh_elements)
        self.radius = domain_radius
        self.sigma_bg = sigma_background
        self.sigma_inc = sigma_inclusion

    def generate_random_circular_inclusions(self, n_inclusions: int = None,
                                             seed: int = None) -> np.ndarray:
        """随机圆形包含物"""
        if seed is not None:
            np.random.seed(seed)

        if n_inclusions is None:
            n_inclusions = np.random.randint(1, 5)

        sigma = np.full(self.n_elems, self.sigma_bg)

        for _ in range(n_inclusions):
            # 随机圆心和半径
            cx = np.random.uniform(-self.radius * 0.6, self.radius * 0.6)
            cy = np.random.uniform(-self.radius * 0.6, self.radius * 0.6)
            r = np.random.uniform(0.01, 0.03)

            # 设置包含物电导率
            dist = np.sqrt((self.centers[:, 0] - cx)**2 + (self.centers[:, 1] - cy)**2)
            mask = dist < r
            sigma_val = np.random.uniform(self.sigma_bg * 2, self.sigma_inc)
            sigma[mask] = sigma_val

        return sigma

    def generate_random_elliptical_inclusions(self, n_inclusions: int = None,
                                               seed: int = None) -> np.ndarray:
        """随机椭圆包含物"""
        if seed is not None:
            np.random.seed(seed)

        if n_inclusions is None:
            n_inclusions = np.random.randint(1, 4)

        sigma = np.full(self.n_elems, self.sigma_bg)

        for _ in range(n_inclusions):
            cx = np.random.uniform(-self.radius * 0.5, self.radius * 0.5)
            cy = np.random.uniform(-self.radius * 0.5, self.radius * 0.5)
            a = np.random.uniform(0.01, 0.04)  # 长轴
            b = np.random.uniform(0.01, 0.03)  # 短轴
            theta = np.random.uniform(0, np.pi)  # 旋转角

            # 旋转坐标
            x_rot = (self.centers[:, 0] - cx) * np.cos(theta) + \
                    (self.centers[:, 1] - cy) * np.sin(theta)
            y_rot = -(self.centers[:, 0] - cx) * np.sin(theta) + \
                    (self.centers[:, 1] - cy) * np.cos(theta)

            mask = (x_rot / a)**2 + (y_rot / b)**2 < 1
            sigma_val = np.random.uniform(self.sigma_bg * 2, self.sigma_inc)
            sigma[mask] = sigma_val

        return sigma

    def generate_gradient_distribution(self, seed: int = None) -> np.ndarray:
        """梯度分布（测试平滑重建）"""
        if seed is not None:
            np.random.seed(seed)

        # 随机方向
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])

        # 沿方向的梯度
        projection = self.centers @ direction
        sigma = self.sigma_bg + (self.sigma_inc - self.sigma_bg) * \
                (projection - projection.min()) / (projection.max() - projection.min())

        return sigma

    def generate_complex_scene(self, seed: int = None) -> np.ndarray:
        """复杂场景：多个不同形状"""
        if seed is not None:
            np.random.seed(seed)

        sigma = np.full(self.n_elems, self.sigma_bg)

        # 添加圆形
        n_circles = np.random.randint(1, 3)
        for _ in range(n_circles):
            cx = np.random.uniform(-self.radius * 0.5, self.radius * 0.5)
            cy = np.random.uniform(-self.radius * 0.5, self.radius * 0.5)
            r = np.random.uniform(0.01, 0.03)
            dist = np.sqrt((self.centers[:, 0] - cx)**2 + (self.centers[:, 1] - cy)**2)
            sigma[dist < r] = np.random.uniform(self.sigma_bg * 2, self.sigma_inc)

        # 添加环形
        if np.random.rand() > 0.5:
            cx, cy = 0, 0
            r_inner = np.random.uniform(0.02, 0.04)
            r_outer = np.random.uniform(0.04, 0.06)
            dist = np.sqrt(self.centers[:, 0]**2 + self.centers[:, 1]**2)
            mask = (dist > r_inner) & (dist < r_outer)
            sigma[mask] = np.random.uniform(self.sigma_bg * 2, self.sigma_inc)

        return sigma

    def generate_random(self, seed: int = None) -> np.ndarray:
        """随机选择一种模式"""
        if seed is not None:
            np.random.seed(seed)

        method = np.random.choice([
            self.generate_random_circular_inclusions,
            self.generate_random_elliptical_inclusions,
            self.generate_gradient_distribution,
            self.generate_complex_scene,
        ])

        return method(seed=seed)


if __name__ == "__main__":
    # 测试模型
    print("=== 测试物理信息增强模型 ===")
    model = PhysicsInformedEIT(input_dim=208, hidden_dim=512, n_frequencies=6, n_elems=1500)

    v = torch.randn(4, 6, 208)
    J = torch.randn(4, 208, 1500)

    out = model(v, J)
    print(f"输入电压: {v.shape}")
    print(f"输入Jacobian: {J.shape}")
    print(f"输出sigma: {out['sigma'].shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n=== 测试迭代细化模型 ===")
    model2 = IterativeRefinementEIT(input_dim=208, hidden_dim=256, n_frequencies=6, n_elems=1500)
    sigma = model2(v)
    print(f"输出sigma: {sigma.shape}")
    print(f"参数量: {sum(p.numel() for p in model2.parameters()):,}")

    print("\n✅ 模型测试通过")
