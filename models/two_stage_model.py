"""
两阶段 EIT 重建模型
==================
第一阶段: 传统反演 (Gauss-Newton / BP)
第二阶段: 神经网络精调

流程:
    电压 → 传统反演 → 粗略电导率 → UNet → 精调电导率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple

from pyeit.eit import bp, jac


class TraditionalReconstructor:
    """
    传统 EIT 反演器
    封装 pyEIT 的 BP/JAC 算法
    """

    def __init__(self, solver, method: str = 'jac'):
        """
        参数:
            solver: EITForwardSolver 实例
            method: 'bp' (反投影) 或 'jac' (Gauss-Newton)
        """
        self.solver = solver
        self.mesh = solver.mesh
        self.protocol = solver.protocol
        self.method = method

        # 创建反演器
        if method == 'bp':
            self.eit = bp.BP(
                mesh=self.mesh,
                protocol=self.protocol,
                solver='lsqr'  # 最小二乘求解器
            )
        elif method == 'jac':
            self.eit = jac.JAC(
                mesh=self.mesh,
                protocol=self.protocol,
                solver='lsqr',
                p=0.5,          # Tikhonov 正则化参数
                lam=0.01,       # 正则化权重
                method='kotre'  # Kotre 方法
            )
        else:
            raise ValueError(f"不支持的方法: {method}")

        # 参考电压（均匀场）
        self.v0 = solver.V_uniform
        self.sigma0 = solver.sigma_uniform

    def reconstruct(self, voltage: np.ndarray) -> np.ndarray:
        """
        反演电导率分布

        参数:
            voltage: (n_meas,) 差分电压

        返回:
            sigma: (n_elems,) 电导率分布
        """
        # pyEIT 反演
        # 返回的是相对电导率变化 ds
        ds = self.eit.solve(voltage, self.v0)

        # 转换为绝对电导率
        sigma = self.sigma0 * (1 + ds)

        # 限制范围
        sigma = np.clip(sigma, 0.001, 0.2)

        return sigma.astype(np.float32)

    def batch_reconstruct(self, voltages: np.ndarray) -> np.ndarray:
        """
        批量反演

        参数:
            voltages: (B, n_meas) 差分电压

        返回:
            sigmas: (B, n_elems) 电导率分布
        """
        B = voltages.shape[0]
        n_elems = self.solver.n_elems

        sigmas = np.zeros((B, n_elems), dtype=np.float32)

        for i in range(B):
            sigmas[i] = self.reconstruct(voltages[i])

        return sigmas


class UNetRefineBlock(nn.Module):
    """
    UNet 风格的精调块
    用于处理网格数据（非规则图像）
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64):
        super().__init__()

        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
        )

        # 解码器
        self.dec3 = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim, 3, padding=1),  # 4x = 2x + 2x skip
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # 输出
        self.out = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, n_elems) 粗略电导率

        返回:
            (B, n_elems) 残差修正
        """
        # 添加通道维度
        x = x.unsqueeze(1)  # (B, 1, n_elems)

        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # 解码 + skip connections
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        # 输出残差
        residual = self.out(d1).squeeze(1)  # (B, n_elems)

        return residual


class GraphUNet(nn.Module):
    """
    图 UNet - 适用于非规则网格
    使用图卷积替代标准卷积
    """

    def __init__(self,
                 n_elems: int,
                 hidden_dim: int = 64,
                 n_layers: int = 4):
        super().__init__()

        self.n_elems = n_elems
        self.hidden_dim = hidden_dim

        # 输入嵌入
        self.input_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 图卷积层
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(GraphConvBlock(hidden_dim, hidden_dim))

        # 输出
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, n_elems) 粗略电导率
            adj: (n_elems, n_elems) 邻接矩阵

        返回:
            (B, n_elems) 精调电导率
        """
        B = x.shape[0]

        # 嵌入
        h = self.input_embed(x.unsqueeze(-1))  # (B, n_elems, hidden_dim)

        # 图卷积
        for layer in self.layers:
            h = layer(h, adj)

        # 输出残差
        residual = self.output(h).squeeze(-1)  # (B, n_elems)

        # 残差连接
        out = x + residual

        return out


class GraphConvBlock(nn.Module):
    """图卷积块"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, N, D)
            adj: (N, N)

        返回:
            (B, N, D)
        """
        # 归一化邻接矩阵
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)  # (1, N, N)

        # 度归一化
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / degree

        # 图卷积: AXW
        h = torch.matmul(adj_norm, x)  # (B, N, D)
        h = self.linear(h)
        h = self.norm(h)
        h = F.relu(h)

        # 残差
        if x.shape[-1] == h.shape[-1]:
            h = h + x

        return h


class TwoStageEITModel(nn.Module):
    """
    两阶段 EIT 重建模型

    Stage 1: 传统反演 (Gauss-Newton)
    Stage 2: 神经网络精调
    """

    def __init__(self,
                 n_elems: int,
                 refine_type: str = 'unet',  # 'unet' 或 'graph'
                 hidden_dim: int = 128,
                 sigma_min: float = 0.005,
                 sigma_max: float = 0.1):
        super().__init__()

        self.n_elems = n_elems
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # 第二阶段: 精调网络
        if refine_type == 'unet':
            self.refine_net = UNetRefineBlock(in_channels=1, hidden_dim=hidden_dim)
        elif refine_type == 'graph':
            self.refine_net = GraphUNet(n_elems, hidden_dim)
        else:
            raise ValueError(f"不支持的精调类型: {refine_type}")

        self.refine_type = refine_type

        # 邻接矩阵缓存
        self._adj_cache = None

    def setup_mesh(self, centers: np.ndarray, elements: np.ndarray):
        """设置网格结构（用于 GraphUNet）"""
        if self.refine_type == 'graph':
            from models.physics_gnn import build_element_adjacency
            adj = build_element_adjacency(elements, len(elements))
            self._adj_cache = torch.from_numpy(adj).float()

    def forward(self,
                coarse_sigma: torch.Tensor,
                target_sigma: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            coarse_sigma: (B, n_elems) 传统反演的粗略电导率
            target_sigma: (B, n_elems) 目标电导率（用于计算损失）

        返回:
            dict: {'sigma': 精调电导率, 'coarse': 粗略电导率}
        """
        B = coarse_sigma.shape[0]
        device = coarse_sigma.device

        # 第二阶段: 精调
        if self.refine_type == 'graph' and self._adj_cache is not None:
            adj = self._adj_cache.to(device)
            residual = self.refine_net(coarse_sigma, adj)
        else:
            residual = self.refine_net(coarse_sigma)

        # 残差精调
        refined = coarse_sigma + residual

        # 限制范围
        refined = torch.clamp(refined, self.sigma_min, self.sigma_max)

        return {
            'sigma': refined,
            'coarse': coarse_sigma,
            'residual': residual,
        }


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 60)
    print("测试两阶段 EIT 模型")
    print("=" * 60)

    # 模拟输入
    n_elems = 2824
    B = 4

    coarse = torch.rand(B, n_elems) * 0.05 + 0.01  # 粗略电导率

    # 测试 UNet 版本
    print("\n测试 UNet 精调...")
    model_unet = TwoStageEITModel(n_elems, refine_type='unet', hidden_dim=64)
    out = model_unet(coarse)
    print(f"  输入: {coarse.shape}")
    print(f"  输出: {out['sigma'].shape}")
    print(f"  参数量: {sum(p.numel() for p in model_unet.parameters()):,}")

    # 测试 GraphUNet 版本
    print("\n测试 GraphUNet 精调...")
    model_graph = TwoStageEITModel(n_elems, refine_type='graph', hidden_dim=64)

    # 模拟网格
    centers = np.random.randn(n_elems, 2).astype(np.float32) * 0.1
    elements = np.random.randint(0, 500, (n_elems, 3)).astype(np.int64)
    model_graph.setup_mesh(centers, elements)

    out = model_graph(coarse)
    print(f"  输出: {out['sigma'].shape}")
    print(f"  参数量: {sum(p.numel() for p in model_graph.parameters()):,}")

    print("\n✅ 两阶段模型测试通过")
