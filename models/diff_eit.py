"""
DiffEIT: Diffusion-based EIT Reconstruction Model
====================================================
组合 MeshUNet denoiser + DDPM diffusion + optional warm-start.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from models.mesh_unet import MeshUNet
from models.diffusion_utils import DiffusionProcess, TimeEmbedding


class VoltageEncoder(nn.Module):
    """电压编码器: 208-dim → hidden"""
    def __init__(self, n_meas: int = 208, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_meas, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

    def forward(self, V):
        """
        V: (B, 208) 或 (B, 6, 208) — 多频取平均
        """
        if V.dim() == 3:
            V = V.mean(dim=1)  # 6频 → 均值
        return self.net(V)


class DiffEIT(nn.Module):
    """
    DiffEIT: Diffusion-based EIT 重建

    用法:
        model = DiffEIT(n_elems=4424)
        model.setup_mesh(centers, elements, jacobian, hierarchy)
        model.to(device)

        # 训练
        sigma_0 = batch['sigmas']  # (B, n_elems)
        V = batch['voltages']     # (B, 6, 208)
        t = torch.randint(0, 500, (B,), device=device)
        epsilon_pred, epsilon_true = model(sigma_0, t, V)

        # 推理
        sigma = model.sample(V, n_steps=50)
    """

    def __init__(self,
                 n_elems: int = 4424,
                 n_meas: int = 208,
                 hidden_dim: int = 384,
                 time_dim: int = 256,
                 voltage_dim: int = 512,
                 pos_dim: int = 35,
                 T: int = 500,
                 n_levels: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        self.n_elems = n_elems
        self.voltage_dim = voltage_dim

        # 扩散过程
        self.diffusion = DiffusionProcess(T=T)

        # 时间嵌入
        self.time_embed = TimeEmbedding(dim=time_dim)

        # 电压编码器
        self.voltage_encoder = VoltageEncoder(n_meas=n_meas, hidden=voltage_dim)

        # 去噪网络
        self.denoiser = MeshUNet(
            n_elems=n_elems,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            voltage_dim=voltage_dim,
            pos_dim=pos_dim,
            dropout=dropout,
        )

        # 位置编码 (在 setup_mesh 时计算)
        self.register_buffer('pos_encoding', torch.empty(0), persistent=False)

    def setup_mesh(self, centers: np.ndarray, elements: np.ndarray,
                   hierarchy: list):
        """设置网格和层次图结构"""
        # 位置编码
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

        # 设置 denoiser 的层次结构
        self.denoiser.setup_mesh(hierarchy, pe)

        # 如果 pos_dim 不匹配，重建 denoiser 的 node_proj
        if hasattr(self.denoiser, 'node_proj') and self.denoiser.node_proj.in_features != 1 + self.voltage_dim + pos_dim + 1:
            self.denoiser.node_proj = nn.Linear(1 + self.voltage_dim + pos_dim + 1, self.denoiser.hidden_dim).to(pe.device)

    def forward(self, sigma_0: torch.Tensor, t: torch.Tensor,
                V: Optional[torch.Tensor] = None):
        """
        训练前向: 加噪 → 去噪 → 返回预测噪声和真实噪声

        参数:
            sigma_0: (B, n_elems) 干净电导率
            t: (B,) 时间步
            V: (B, 6, 208) 或 (B, 208) 边界电压 (可选, 无条件模式)

        返回:
            epsilon_pred: (B, n_elems, 1) 预测噪声
            epsilon_true: (B, n_elems) 真实噪声 (用于 loss)
        """
        # 加噪
        sigma_t, noise = self.diffusion.forward_diffuse(sigma_0, t)

        # 条件编码
        t_emb = self.time_embed(t)
        v_emb = self.voltage_encoder(V) if V is not None else torch.zeros(
            sigma_0.shape[0], self.voltage_dim, device=sigma_0.device)

        # 去噪
        epsilon_pred = self.denoiser(
            sigma_t.unsqueeze(-1),  # (B, n_elems, 1)
            t_emb,
            v_emb,
        ).squeeze(-1)  # (B, n_elems, 1) → (B, n_elems)

        return epsilon_pred, noise

    @torch.no_grad()
    def sample(self, V: torch.Tensor, n_steps: int = 50, n_samples: int = 1,
               w_phys: float = 0.1):
        """
        推理: 从噪声采样电导率

        参数:
            V: (208,) 或 (6, 208) 边界电压
            n_steps: DDIM 步数
            n_samples: 采样次数 (用于不确定性估计)
            w_phys: 物理引导强度 (预留)

        返回:
            sigma: (n_elems,)  或  (n_samples, n_elems) 如果 n_samples > 1
        """
        if V.dim() == 2:
            V = V.mean(dim=0)  # (6, 208) → (208,)
        V = V.unsqueeze(0)  # (1, 208)
        v_emb = self.voltage_encoder(V)

        def denoise_fn(sigma_t_batch, t_tensor, v_emb_in):
            t_emb = self.time_embed(t_tensor)
            # Ensure (B, N, 1) shape
            if sigma_t_batch.dim() == 1:
                sigma_t_batch = sigma_t_batch.unsqueeze(0).unsqueeze(-1)  # (N,) → (1, N, 1)
            elif sigma_t_batch.dim() == 2:
                sigma_t_batch = sigma_t_batch.unsqueeze(-1)  # (1, N) → (1, N, 1)
            if v_emb_in.dim() == 1:
                v_emb_in = v_emb_in.unsqueeze(0)  # (Vd,) → (1, Vd)
            eps = self.denoiser(sigma_t_batch, t_emb, v_emb_in)
            eps = eps.squeeze(0).squeeze(-1) if eps.dim() >= 2 else eps
            sigma_t_1d = sigma_t_batch.squeeze(0).squeeze(-1) if sigma_t_batch.dim() >= 2 else sigma_t_batch
            sqrt_alpha = self.diffusion.sqrt_alphas_cumprod[t_tensor[0]]
            sqrt_one_minus = self.diffusion.sqrt_one_minus_alphas_cumprod[t_tensor[0]]
            return (sigma_t_1d - sqrt_one_minus * eps) / (sqrt_alpha + 1e-8)

        samples = []
        for _ in range(n_samples):
            sigma = self.diffusion.ddim_sample(
                denoise_fn, v_emb, self.n_elems, n_steps=n_steps)
            samples.append(sigma)

        if n_samples == 1:
            return samples[0].clamp(0.005, 0.1)
        else:
            return torch.stack(samples)  # (n_samples, n_elems)

    def to(self, device):
        super().to(device)
        self.diffusion.to(device)
        return self
