"""
DiffEIT: Diffusion utilities — DDPM / DDIM noise schedule & sampling.
========================================================================
v2: 余弦噪声调度 + x₀-prediction 支持
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


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    Cosine 噪声调度 (Nichol & Dhariwal 2021).
    对窄值域问题 (如 EIT σ ∈ [0.005, 0.1]) 更友好 — 早期保留更多信号.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, max=0.999)
    alphas = 1.0 - betas
    return betas, alphas, alphas_cumprod[1:]


class DiffusionProcess:
    """
    DDPM 扩散过程 (v2: 支持余弦调度 + x₀-prediction)

    用法:
        diff = DiffusionProcess(T=500, schedule='cosine')
        sigma_t, noise = diff.forward_diffuse(sigma_0, t)   # 训练用
        sigma = diff.ddim_sample(denoiser, V, n_steps=50)    # 推理用
    """

    def __init__(self, T: int = 500, schedule: str = 'cosine',
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        if schedule == 'cosine':
            betas, alphas, alphas_cumprod = cosine_beta_schedule(T)
        else:
            betas, alphas, alphas_cumprod = linear_beta_schedule(T, beta_start, beta_end)

        self.register('betas', betas)
        self.register('alphas', alphas)
        self.register('alphas_cumprod', alphas_cumprod)
        self.register('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt())

    def register(self, name, tensor):
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
            noise:   (B, n_elems) 实际添加的噪声（ε-pred 训练目标）
        """
        batch_size = sigma_0.shape[0]
        t = t % self.T
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(batch_size, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1)
        noise = torch.randn_like(sigma_0)
        sigma_t = sqrt_alpha_bar * sigma_0 + sqrt_one_minus * noise
        return sigma_t, noise

    def forward_diffuse_residual(self, residual: torch.Tensor, t: torch.Tensor,
                                  noise_scale: float = 0.5):
        """
        残差加噪 (warm-start 模式): r_t = √ᾱ_t r_0 + noise_scale * √(1-ᾱ_t) ε

        对残差使用缩放的噪声, 因为残差方差比全 σ 方差小得多.
        """
        batch_size = residual.shape[0]
        t = t % self.T
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(batch_size, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1)
        noise = torch.randn_like(residual)
        residual_t = sqrt_alpha_bar * residual + noise_scale * sqrt_one_minus * noise
        return residual_t, noise

    @torch.no_grad()
    def ddim_sample(self, denoiser, V_cond, n_elems: int, n_steps: int = 50,
                    eta: float = 0.0, w_phys: float = 0.1,
                    sigma_warm: torch.Tensor = None,
                    sigma_min: float = 0.005, sigma_max: float = 0.1):
        """
        DDIM 加速采样 (v2: 支持 warm-start + x₀-prediction)

        参数:
            denoiser: callable (sigma_t, t, v_emb) → sigma_0_hat (x₀-pred) 或 epsilon (ε-pred)
            V_cond: (512,) or (B, 512) 电压条件
            n_elems: 网格元素数
            n_steps: DDIM 步数
            eta: 0=确定性, 1=随机
            w_phys: 物理引导强度 (预留)
            sigma_warm: (n_elems,) warm-start 粗估计, None = 从噪声起步
            sigma_min/max: 输出钳位范围

        返回:
            sigma_0: (n_elems,) 重建电导率
        """
        device = V_cond.device if isinstance(V_cond, torch.Tensor) else self.device

        # DDIM timestep
        stride = self.T // n_steps
        timesteps = list(range(0, self.T, stride))[:n_steps]
        timesteps_next = timesteps[1:] + [self.T]

        # 初始化: warm-start 模式用缩小噪声, 否则标准噪声
        if sigma_warm is not None:
            # 残差扩散: 初始噪声尺度与训练一致 (noise_scale=0.5)
            sigma_t = sigma_warm + 0.5 * torch.randn(n_elems, device=device)
        else:
            sigma_t = torch.randn(n_elems, device=device)

        for t, t_next in zip(reversed(timesteps), reversed(timesteps_next)):
            t_tensor = torch.tensor([t], device=device)
            t_next_tensor = torch.tensor([t_next], device=device)

            # 预测 σ_0 (denoiser 返回的是干净 σ 的估计, x₀-pred 模式)
            sigma_0_hat = denoiser(sigma_t.unsqueeze(0), t_tensor, V_cond)
            if sigma_0_hat.dim() >= 2:
                sigma_0_hat = sigma_0_hat.squeeze(0)
            if sigma_0_hat.dim() > 1:
                sigma_0_hat = sigma_0_hat.squeeze(-1)

            # 从 σ_0_hat 反推 ε
            alpha_bar_t = self.alphas_cumprod[t]
            eps_pred = (sigma_t - self.sqrt_alphas_cumprod[t] * sigma_0_hat) / \
                       (self.sqrt_one_minus_alphas_cumprod[t] + 1e-8)

            # DDIM step
            if t_next >= self.T:
                sigma_t = sigma_0_hat  # 最后一步直接输出
            else:
                alpha_bar_next = self.alphas_cumprod[t_next]
                sqrt_alpha_next = self.sqrt_alphas_cumprod[t_next]
                # 确定性 (eta=0) 或 随机 (eta>0)
                sigma_noise = 0.0
                if eta > 0:
                    sigma_noise = eta * torch.sqrt(
                        (1 - alpha_bar_next) / (1 - alpha_bar_t + 1e-8) *
                        (1 - alpha_bar_t / (alpha_bar_next + 1e-8))
                    )
                    sigma_t = sqrt_alpha_next * sigma_0_hat + \
                              torch.sqrt(1 - alpha_bar_next - sigma_noise**2 + 1e-8) * eps_pred + \
                              sigma_noise * torch.randn_like(sigma_t)
                else:
                    sigma_t = sqrt_alpha_next * sigma_0_hat + \
                              torch.sqrt(1 - alpha_bar_next + 1e-8) * eps_pred

        return sigma_0_hat.clamp(sigma_min, sigma_max)

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
