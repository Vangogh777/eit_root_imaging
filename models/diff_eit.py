"""
DiffEIT v4: Diffusion-based EIT Reconstruction Model
========================================================
组合 MeshUNet denoiser + DDPM diffusion + sensitivity conditioning.
Standardized DDPM: [0,1] normalization + N(0,1) standardization.

v4 改进:
  - 标准化扩散空间 (N(0,1) 替代原始 σ 空间)
  - 两阶段归一化: [0,1] → N(0,1)
  - 移除 warm-start (架构简化)
  - MeshUNet out_head Sigmoid → Linear (支持负值)
  - T 默认 500→200 (推理加速)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from models.mesh_unet import MeshUNet
from models.diffusion_utils import DiffusionProcess, TimeEmbedding


class VoltageEncoder(nn.Module):
    """电压编码器: 208-dim → hidden, 支持多频 Cross-Attention"""
    def __init__(self, n_meas: int = 208, hidden: int = 512, n_freq: int = 6):
        super().__init__()
        self.n_freq = n_freq
        # 先对各频率独立编码, 再做频率间 attention
        self.freq_proj = nn.Sequential(
            nn.Linear(n_meas, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
        )
        # 跨频率 attention (轻量)
        self.freq_attn = nn.MultiheadAttention(hidden // 2, num_heads=4,
                                                dropout=0.1, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden // 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

    def forward(self, V):
        """
        V: (B, 208) 或 (B, 6, 208) — 多频用 attention 融合
        """
        if V.dim() == 2:
            # 单频: 直接编码
            h = self.freq_proj(V)
            return self.out_proj(h)
        else:
            # 多频: 各频率独立投影 → cross-attention → 平均 → 最终投影
            B, F, M = V.shape
            h_freq = self.freq_proj(V.view(B * F, M)).view(B, F, -1)  # (B, F, h/2)
            h_attn, _ = self.freq_attn(h_freq, h_freq, h_freq)          # (B, F, h/2)
            h_pooled = h_attn.mean(dim=1)                                 # (B, h/2)
            return self.out_proj(h_pooled)


class DiffEIT(nn.Module):
    """
    DiffEIT v4: Diffusion-based EIT 重建 (标准化 DDPM)

    用法:
        model = DiffEIT(n_elems=4424)
        model.setup_mesh(centers, elements, jacobian, hierarchy)
        model.configure_sigma_stats(sigma_min, sigma_max, sigma_mean, sigma_std)
        model.to(device)

        # 训练 (x₀-prediction in N(0,1) space)
        sigma_0_true = batch['sigmas']
        V = batch['voltages']
        t = torch.randint(0, 200, (B,), device=device)
        sigma_0_pred, sigma_0_std = model(sigma_0_true, t, V)
        loss = MSE(sigma_0_pred, sigma_0_std)

        # 推理 (DDIM)
        sigma = model.sample(V, n_steps=50)
    """

    def __init__(self,
                 n_elems: int = 4424,
                 n_meas: int = 208,
                 hidden_dim: int = 384,
                 time_dim: int = 256,
                 voltage_dim: int = 512,
                 pos_dim: int = 35,
                 T: int = 200,
                 n_levels: int = 3,
                 dropout: float = 0.1,
                 schedule: str = 'cosine'):
        super().__init__()

        self.n_elems = n_elems
        self.voltage_dim = voltage_dim

        # 归一化参数 (通过 configure_sigma_stats 或 setup_mesh 设置)
        self.sigma_min = None
        self.sigma_max = None
        self.sigma_range = None
        self.sigma_mean = None
        self.sigma_std = None

        # 扩散过程 (余弦调度)
        self.diffusion = DiffusionProcess(T=T, schedule=schedule)

        # 时间嵌入
        self.time_embed = TimeEmbedding(dim=time_dim)

        # 电压编码器 (多频 attention)
        self.voltage_encoder = VoltageEncoder(n_meas=n_meas, hidden=voltage_dim)

        # 去噪网络 (extra_dim=2: J^T·V + J_energy)
        self.denoiser = MeshUNet(
            n_elems=n_elems,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            voltage_dim=voltage_dim,
            pos_dim=pos_dim,
            extra_dim=2,  # J^T·V, J_energy
            dropout=dropout,
        )

        # Jacobian 缓冲 (在 setup_mesh 时注册)
        self.register_buffer('J_T', torch.empty(0), persistent=False)
        self.register_buffer('J_energy', torch.empty(0), persistent=False)
        self.register_buffer('pos_encoding', torch.empty(0), persistent=False)

    def configure_sigma_stats(self, sigma_min: float, sigma_max: float,
                              sigma_mean: float, sigma_std: float):
        """设置从训练数据推断的归一化参数."""
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_range = sigma_max - sigma_min
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std

    def setup_mesh(self, centers: np.ndarray, elements: np.ndarray,
                   jacobian: np.ndarray, hierarchy: list,
                   sigma_ref: float = 0.01):
        """
        设置网格、层次图结构、Jacobian 相关缓冲.

        参数:
            centers: (n_elems, 2) 元素中心
            elements: (n_elems, 3) 三角连接
            jacobian: (208, n_elems) 灵敏度矩阵
            hierarchy: 多尺度图结构
            sigma_ref: (保留, 不再使用)
        """
        # ---- 位置编码 ----
        c = centers[:, :2].astype(np.float32)
        r_max = np.abs(c).max() + 1e-8
        pos = torch.from_numpy(c / r_max).float()

        def fourier(x, n_freq=8, scale=2.0):
            x = torch.from_numpy(x).float()
            freqs = (scale ** torch.arange(n_freq).float()) * np.pi
            args = x[:, :, None] * freqs
            return torch.cat([torch.sin(args), torch.cos(args)], -1).reshape(len(x), -1)

        radius = torch.norm(pos, dim=1, keepdim=True)
        pe = torch.cat([pos, fourier(c / r_max), radius], dim=-1)
        self.register_buffer('pos_encoding', pe)
        pos_dim = pe.shape[1]

        # ---- Jacobian 缓冲 ----
        J = torch.from_numpy(jacobian).float()
        if J.dim() == 3:
            J = J[0]  # 取第一个频率 (pyEIT 多频返回相同 J)
        self.register_buffer('J_T', J.T.contiguous())  # (n_elems, 208)
        self.register_buffer('J_energy', (J ** 2).sum(dim=0))  # (n_elems,)

        # ---- 默认归一化参数 (若未通过 configure_sigma_stats 设置) ----
        if self.sigma_min is None:
            self.sigma_min = 0.005
            self.sigma_max = 0.1
            self.sigma_range = self.sigma_max - self.sigma_min
        if self.sigma_mean is None:
            self.sigma_mean = 0.5
            self.sigma_std = 0.25

        # ---- 设置 denoiser 层次结构 ----
        self.denoiser.setup_mesh(hierarchy, pe)

        # 如果 pos_dim 不匹配, 重建 node_proj
        in_dim = 1 + self.voltage_dim + pos_dim + 1 + 2  # σ_t + V + PE + radius + 2 extra
        if hasattr(self.denoiser, 'node_proj') and self.denoiser.node_proj.in_features != in_dim:
            self.denoiser.node_proj = nn.Linear(in_dim, self.denoiser.hidden_dim).to(pe.device)

    def _compute_sensitivity_features(self, V: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        计算灵敏度特征: J^T·V 和 J_energy, (B, n_elems, 2)
        如果 V 为 None (无条件模式), 返回 None.
        V 可以是 (B, 208) 或 (B, 6, 208) — 多频取第一频率.
        """
        if V is None or self.J_T.numel() == 0:
            return None
        # 取第一个频率 (或单频)
        if V.dim() == 3:
            V_0 = V[:, 0, :]  # (B, 208)
        elif V.dim() == 2 and V.shape[-1] == 208:
            V_0 = V  # (B, 208)
        else:
            # 单个样本 (208,)
            V_0 = V.unsqueeze(0)  # (1, 208)
        B = V_0.shape[0]

        J_T_V = V_0 @ self.J_T.T  # (B, 208) @ (208, n_elems) = (B, n_elems)
        J_energy = self.J_energy.unsqueeze(0).expand(B, -1)  # (B, n_elems)
        return torch.stack([J_T_V, J_energy], dim=-1)  # (B, n_elems, 2)

    def forward(self, sigma_0: torch.Tensor, t: torch.Tensor,
                V: Optional[torch.Tensor] = None):
        """
        训练前向 (x₀-prediction): 两阶段归一化 → 加噪 → 去噪

        参数:
            sigma_0: (B, n_elems) 干净电导率真值
            t: (B,) 时间步
            V: (B, 6, 208) 或 (B, 208) 边界电压

        返回:
            sigma_0_pred_std: (B, n_elems, 1) — 预测的干净 σ (N(0,1) 空间)
            sigma_0_std:      (B, n_elems, 1) — 真实的干净 σ (N(0,1) 空间)
        """
        B = sigma_0.shape[0]
        device = sigma_0.device

        # Step 1: 归一化到 [0, 1]
        sigma_0_norm = ((sigma_0 - self.sigma_min) / self.sigma_range).clamp(0, 1)

        # Step 2: 标准化到 N(0,1)
        sigma_0_std = (sigma_0_norm - self.sigma_mean) / self.sigma_std

        # Step 3: 标准 DDPM 前向加噪 (no warm-start)
        sigma_t_std, noise = self.diffusion.forward_diffuse(sigma_0_std, t)

        # Step 4: 条件编码
        t_emb = self.time_embed(t)
        v_emb = self.voltage_encoder(V) if V is not None else torch.zeros(
            B, self.voltage_dim, device=device)

        # 灵敏度特征
        extra_feat = self._compute_sensitivity_features(V)

        # Step 5: 在标准化空间去噪
        sigma_0_pred_std = self.denoiser(
            sigma_t_std.unsqueeze(-1),   # (B, n_elems, 1)
            t_emb,
            v_emb,
            extra_feat=extra_feat,
        )  # (B, n_elems, 1) — 在 N(0,1) 空间 (Linear 输出)

        return sigma_0_pred_std, sigma_0_std.unsqueeze(-1)

    @torch.no_grad()
    def sample(self, V, n_steps=50, n_samples=1):
        """
        推理: N(0,1) 噪声 → 标准化 DDIM → 反变换到物理空间

        参数:
            V: (208,) 或 (6, 208) 边界电压
            n_steps: DDIM 步数
            n_samples: 采样次数 (用于不确定性估计)

        返回:
            sigma: (n_elems,) 或 (n_samples, n_elems)
        """
        if V.dim() == 2:
            V_enc = V.unsqueeze(0)  # (1, n_freq, n_meas)
        elif V.dim() == 1:
            V_enc = V.unsqueeze(0).unsqueeze(0)  # (1, 1, n_meas)
        else:
            V_enc = V

        v_emb = self.voltage_encoder(V_enc)

        V_sens = V_enc[:, 0, :] if V_enc.dim() == 3 else V_enc
        extra_feat = self._compute_sensitivity_features(V_sens)

        def denoise_fn(sigma_t_std, t_tensor, v_emb_in):
            """Standardized space → standardized space"""
            t_emb = self.time_embed(t_tensor)
            if sigma_t_std.dim() == 1:
                sigma_t_std = sigma_t_std.unsqueeze(0).unsqueeze(-1)
            elif sigma_t_std.dim() == 2:
                sigma_t_std = sigma_t_std.unsqueeze(-1)
            if v_emb_in.dim() == 1:
                v_emb_in = v_emb_in.unsqueeze(0)
            B_in = sigma_t_std.shape[0]
            ef = extra_feat.expand(B_in, -1, -1) if extra_feat is not None and extra_feat.shape[0] != B_in else extra_feat
            return self.denoiser(sigma_t_std, t_emb, v_emb_in, extra_feat=ef).squeeze(0).squeeze(-1)

        samples = []
        for _ in range(n_samples):
            sigma_std = self.diffusion.ddim_sample(
                denoise_fn, v_emb.squeeze(0) if v_emb.dim() > 1 else v_emb,
                self.n_elems, n_steps=n_steps,
            )
            # 反变换: N(0,1) → [0,1] → 物理空间
            sigma_norm = sigma_std * self.sigma_std + self.sigma_mean
            sigma_norm = sigma_norm.clamp(0, 1)
            sigma_phys = sigma_norm * self.sigma_range + self.sigma_min
            sigma_phys = sigma_phys.clamp(self.sigma_min, self.sigma_max)
            samples.append(sigma_phys)

        if n_samples == 1:
            return samples[0]
        else:
            return torch.stack(samples)

    def to(self, device):
        super().to(device)
        self.diffusion.to(device)
        return self
