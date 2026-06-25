"""
DiffEIT: Diffusion utilities — DDPM / DDIM noise schedule & sampling.
========================================================================
独立模块，不依赖项目其他代码。
"""
import torch
import torch.nn as nn
import numpy as np


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Linear 噪声调度，返回 β_t, α_t, ᾱ_t"""
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


class DiffusionProcess:
    """
    DDPM 扩散过程

    用法:
        diff = DiffusionProcess(T=500)
        sigma_t, noise = diff.forward_diffuse(sigma_0, t)   # 训练用
        sigma = diff.ddim_sample(model, V, n_steps=50)       # 推理用
    """

    def __init__(self, T: int = 500, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        betas, alphas, alphas_cumprod = linear_beta_schedule(T, beta_start, beta_end)
        self.register('betas', betas)
        self.register('alphas', alphas)
        self.register('alphas_cumprod', alphas_cumprod)
        self.register('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt())

    def register(self, name, tensor):
        """Helper to register buffer-like attributes without nn.Module"""
        setattr(self, name, tensor)

    @property
    def device(self):
        return self.betas.device

    def forward_diffuse(self, sigma_0: torch.Tensor, t: torch.Tensor):
        """
        前向加噪: σ_t = √ᾱ_t σ_0 + √(1-ᾱ_t) ε

        参数:
            sigma_0: (B, n_elems) 干净电导率
            t: (B,) 时间步 [0, T-1]

        返回:
            sigma_t: (B, n_elems) 加噪后
            noise:   (B, n_elems) 实际添加的噪声（训练目标）
        """
        batch_size = sigma_0.shape[0]
        t = t % self.T
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(batch_size, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1)
        noise = torch.randn_like(sigma_0)
        sigma_t = sqrt_alpha_bar * sigma_0 + sqrt_one_minus * noise
        return sigma_t, noise

    @torch.no_grad()
    def ddim_sample(self, denoiser, V_cond, n_elems: int, n_steps: int = 50,
                    eta: float = 0.0, w_phys: float = 0.1):
        """
        DDIM 加速采样

        参数:
            denoiser: callable (sigma_t, t) → sigma_0_hat
            V_cond: (512,) or (B, 512) 电压条件
            n_elems: 网格元素数
            n_steps: DDIM 步数 (越少越快)
            eta: 0=确定性, 1=随机
            w_phys: 物理引导强度 (暂未使用, 预留)

        返回:
            sigma_0: (n_elems,) 重建电导率
        """
        device = V_cond.device if isinstance(V_cond, torch.Tensor) else self.device

        # DDIM timestep 选择
        stride = self.T // n_steps
        timesteps = list(range(0, self.T, stride))[:n_steps]
        timesteps_next = timesteps[1:] + [self.T]

        sigma_t = torch.randn(n_elems, device=device)

        for t, t_next in zip(reversed(timesteps), reversed(timesteps_next)):
            t_tensor = torch.tensor([t], device=device)
            t_next_tensor = torch.tensor([t_next], device=device)

            # 预测 σ_0
            sigma_0_hat = denoiser(sigma_t.unsqueeze(0), t_tensor, V_cond).squeeze(0)

            # DDIM 递推
            alpha_bar_t = self.alphas_cumprod[t]
            if t_next >= self.T:
                alpha_bar_next = 1.0
                sqrt_alpha_next = 1.0
            else:
                alpha_bar_next = self.alphas_cumprod[t_next]
                sqrt_alpha_next = self.sqrt_alphas_cumprod[t_next]

            # 预测的噪声方向
            eps_pred = (sigma_t - self.sqrt_alphas_cumprod[t] * sigma_0_hat) / \
                       (self.sqrt_one_minus_alphas_cumprod[t] + 1e-8)

            # DDIM step
            sigma_t = sqrt_alpha_next * sigma_0_hat + \
                      (max(1.0 - alpha_bar_next, 0.0)) ** 0.5 * eps_pred

        return sigma_0_hat.clamp(0.005, 0.1)

    def to(self, device):
        for name in ['betas', 'alphas', 'alphas_cumprod', 'sqrt_alphas_cumprod',
                     'sqrt_one_minus_alphas_cumprod']:
            if hasattr(self, name):
                setattr(self, name, getattr(self, name).to(device))
        return self


class TimeEmbedding(nn.Module):
    """Sinusoidal 时间步嵌入, 标准扩散模型做法"""
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        参数:
            t: (B,) 时间步

        返回:
            emb: (B, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(-torch.arange(half_dim, dtype=torch.float32, device=t.device) *
                          (np.log(10000.0) / (half_dim - 1)))
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)
