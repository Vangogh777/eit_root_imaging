"""
DiffEIT v3: Diffusion-based EIT Reconstruction Model
========================================================
组合 MeshUNet denoiser + DDPM diffusion + sensitivity conditioning + warm-start.

v3 改进:
  - 灵敏度特征注入 (J^T·V, J_energy) 作为逐节点条件
  - x₀-prediction 替代 ε-prediction (数值更稳定, 物理范围约束)
  - 线性 warm-start 支持 (残差扩散)
  - 余弦噪声调度
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
    DiffEIT v3: Diffusion-based EIT 重建

    用法:
        model = DiffEIT(n_elems=4424)
        model.setup_mesh(centers, elements, jacobian, hierarchy)
        model.to(device)

        # 训练 (x₀-prediction)
        sigma_0_true = batch['sigmas']
        V = batch['voltages']
        t = torch.randint(0, 500, (B,), device=device)
        sigma_0_pred = model(sigma_0_true, t, V)  # 直接预测干净 σ
        loss = MSE(sigma_0_pred, sigma_0_true)

        # 推理 (warm-start)
        sigma = model.sample(V, n_steps=50, warm_start=True)
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
                 dropout: float = 0.1,
                 sigma_min: float = 0.005,
                 sigma_max: float = 0.1,
                 schedule: str = 'cosine'):
        super().__init__()

        self.n_elems = n_elems
        self.voltage_dim = voltage_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_range = sigma_max - sigma_min

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

        # Warm-start: 线性最小二乘伪逆 (在 setup_mesh 时预计算)
        self.register_buffer('J_pinv', torch.empty(0), persistent=False)
        self._sigma_ref = None

    def setup_mesh(self, centers: np.ndarray, elements: np.ndarray,
                   jacobian: np.ndarray, hierarchy: list,
                   sigma_ref: float = 0.01):
        """
        设置网格、层次图结构、Jacobian 相关缓冲。

        参数:
            centers: (n_elems, 2) 元素中心
            elements: (n_elems, 3) 三角连接
            jacobian: (208, n_elems) 灵敏度矩阵
            hierarchy: 多尺度图结构
            sigma_ref: 参考电导率 (用于 warm-start 和 J 线性化)
        """
        self._sigma_ref = sigma_ref

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

        # ---- 线性最小二乘伪逆 (用于 warm-start) ----
        # J_pinv = J^T (J J^T + λI)^{-1}, λ=0.01
        J_np = J.numpy() if isinstance(J, torch.Tensor) else J
        lam = 0.01 * np.eye(J_np.shape[0])
        JJT_inv = np.linalg.inv(J_np @ J_np.T + lam)
        J_pinv_np = J_np.T @ JJT_inv
        self.register_buffer('J_pinv', torch.from_numpy(J_pinv_np).float())

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

    def compute_warm_start(self, V: torch.Tensor) -> torch.Tensor:
        """
        线性最小二乘 warm-start: σ_warm = σ_ref + J_pinv @ (V - J @ σ_ref)

        参数:
            V: (B, 208) 或 (1, 208)
        返回:
            sigma_warm: (B, n_elems) 粗估计
        """
        if self.J_pinv.numel() == 0:
            raise RuntimeError("Call setup_mesh() first.")

        if V.dim() == 3:
            V = V[:, 0, :]  # 取第一频率
        if V.dim() == 1:
            V = V.unsqueeze(0)

        B = V.shape[0]
        sigma_ref = torch.full((B, self.n_elems), self._sigma_ref,
                                device=V.device, dtype=V.dtype)
        # ΔV = V - J @ σ_ref
        # J: (208, n_elems), sigma_ref: (B, n_elems) → J @ sigma_ref: (B, 208)
        J = self.J_T.T  # (208, n_elems)
        J_sigma_ref = (J @ sigma_ref.T).T  # (B, 208)
        delta_V = V - J_sigma_ref
        delta_sigma = delta_V @ self.J_pinv.T  # (B, n_elems)
        sigma_warm = sigma_ref + delta_sigma
        return sigma_warm

    def forward(self, sigma_0: torch.Tensor, t: torch.Tensor,
                V: Optional[torch.Tensor] = None,
                sigma_warm: Optional[torch.Tensor] = None):
        """
        训练前向 (x₀-prediction): 加噪 → 去噪 → 预测干净 σ

        参数:
            sigma_0: (B, n_elems) 干净电导率真值
            t: (B,) 时间步
            V: (B, 6, 208) 或 (B, 208) 边界电压
            sigma_warm: (B, n_elems) warm-start 粗估计 (可选)

        返回:
            sigma_0_pred: (B, n_elems, 1) — 预测的干净 σ (在 [0, 1] 归一化空间)
            sigma_0_true:  (B, n_elems, 1) — 真实的干净 σ (同样归一化)
        """
        B = sigma_0.shape[0]
        device = sigma_0.device

        # --- Warm-start: 训练目标为残差 ---
        if sigma_warm is not None:
            residual = sigma_0 - sigma_warm  # 残差 (仍可能 > 1 或 < 0)
            # 加噪 (残差模式, 缩小噪声尺度)
            residual_t, noise = self.diffusion.forward_diffuse_residual(
                residual, t, noise_scale=0.5)
            # 输入到 denoiser 的是 noisy 残差 + warm_start
            sigma_t = sigma_warm + residual_t
        else:
            # 标准模式: 直接加噪全 σ
            sigma_t, noise = self.diffusion.forward_diffuse(sigma_0, t)

        # --- 条件编码 ---
        t_emb = self.time_embed(t)
        v_emb = self.voltage_encoder(V) if V is not None else torch.zeros(
            B, self.voltage_dim, device=device)

        # --- 灵敏度特征 ---
        extra_feat = self._compute_sensitivity_features(V)  # (B, n_elems, 2)

        # --- 去噪: 预测干净 σ ---
        sigma_0_pred = self.denoiser(
            sigma_t.unsqueeze(-1),   # (B, n_elems, 1)
            t_emb,
            v_emb,
            extra_feat=extra_feat,
        )  # (B, n_elems, 1) — 在 [0, 1] 范围 (Sigmoid 输出)

        # 归一化真值到 [0, 1]
        sigma_0_norm = ((sigma_0 - self.sigma_min) / self.sigma_range).clamp(0, 1)
        sigma_0_true = sigma_0_norm.unsqueeze(-1)

        return sigma_0_pred, sigma_0_true

    def forward_epsilon(self, sigma_0: torch.Tensor, t: torch.Tensor,
                        V: Optional[torch.Tensor] = None,
                        sigma_warm: Optional[torch.Tensor] = None):
        """
        训练前向 (ε-prediction 兼容模式): 返回预测和真实噪声

        用于与旧版 checkpoint / 对比实验兼容.
        """
        B = sigma_0.shape[0]
        device = sigma_0.device

        if sigma_warm is not None:
            residual = sigma_0 - sigma_warm
            sigma_t, noise = self.diffusion.forward_diffuse_residual(residual, t, noise_scale=0.5)
            sigma_t = sigma_warm + residual if sigma_warm is not None else sigma_t
        else:
            sigma_t, noise = self.diffusion.forward_diffuse(sigma_0, t)

        t_emb = self.time_embed(t)
        v_emb = self.voltage_encoder(V) if V is not None else torch.zeros(
            B, self.voltage_dim, device=device)
        extra_feat = self._compute_sensitivity_features(V)

        sigma_0_pred_norm = self.denoiser(
            sigma_t.unsqueeze(-1), t_emb, v_emb, extra_feat=extra_feat,
        ).squeeze(-1)  # (B, n_elems)

        # 从 x₀_hat 反推 ε
        t_idx = t % self.diffusion.T
        sqrt_alpha_bar = self.diffusion.sqrt_alphas_cumprod[t_idx].view(B, 1)
        sqrt_one_minus = self.diffusion.sqrt_one_minus_alphas_cumprod[t_idx].view(B, 1)

        # sigma_0_pred_norm 在 [0,1], 需要 scale 回物理空间
        sigma_0_pred_phys = sigma_0_pred_norm * self.sigma_range + self.sigma_min
        if sigma_warm is not None:
            sigma_0_pred_phys = sigma_0_pred_phys + sigma_warm

        epsilon_pred = (sigma_t - sqrt_alpha_bar * sigma_0_pred_phys) / (sqrt_one_minus + 1e-8)
        return epsilon_pred, noise

    @torch.no_grad()
    def sample(self, V: torch.Tensor, n_steps: int = 50, n_samples: int = 1,
               warm_start: bool = True, w_phys: float = 0.0):
        """
        推理: 从噪声 / warm-start 采样电导率

        参数:
            V: (208,) 或 (6, 208) 边界电压
            n_steps: DDIM 步数
            n_samples: 采样次数 (用于不确定性估计)
            warm_start: 是否使用线性 warm-start
            w_phys: 物理引导强度 (预留, 当前未实现)

        返回:
            sigma: (n_elems,) 或 (n_samples, n_elems)
        """
        if V.dim() == 2:
            # V: (n_freq, n_meas) → 单样本多频, reshape 为 (1, n_freq, n_meas)
            V_mean = V.mean(dim=0)
            V_enc_input = V.unsqueeze(0)  # (1, n_freq, n_meas)
        elif V.dim() == 1:
            V_mean = V
            V_enc_input = V.unsqueeze(0).unsqueeze(0)  # (1, 1, n_meas)
        else:
            V_mean = V.squeeze(0) if V.dim() == 3 else V
            V_enc_input = V  # already (B, n_freq, n_meas)

        v_emb = self.voltage_encoder(V_enc_input)  # (1 or B, voltage_dim)

        # 计算 warm-start
        sigma_warm = None
        if warm_start:
            sigma_warm = self.compute_warm_start(V_enc_input).squeeze(0)  # (n_elems,)

        # 预计算灵敏度特征 (取第一频率, batch=1)
        V_for_sens = V_enc_input[:, 0, :] if V_enc_input.dim() == 3 else V_enc_input
        extra_feat = self._compute_sensitivity_features(V_for_sens)  # (1, n_elems, 2) or None

        def denoise_fn(sigma_t_batch, t_tensor, v_emb_in):
            """denoiser wrapper: returns σ_0_hat in physical space"""
            t_emb = self.time_embed(t_tensor)
            if sigma_t_batch.dim() == 1:
                sigma_t_batch = sigma_t_batch.unsqueeze(0).unsqueeze(-1)
            elif sigma_t_batch.dim() == 2:
                sigma_t_batch = sigma_t_batch.unsqueeze(-1)
            if v_emb_in.dim() == 1:
                v_emb_in = v_emb_in.unsqueeze(0)

            # 匹配 batch size
            B_in = sigma_t_batch.shape[0]
            ef = None
            if extra_feat is not None:
                if extra_feat.shape[0] != B_in:
                    ef = extra_feat.expand(B_in, -1, -1)
                else:
                    ef = extra_feat

            sigma_0_norm = self.denoiser(sigma_t_batch, t_emb, v_emb_in, extra_feat=ef)
            sigma_0_phys = sigma_0_norm.squeeze(0).squeeze(-1) * self.sigma_range + self.sigma_min
            return sigma_0_phys

        samples = []
        for _ in range(n_samples):
            sigma = self.diffusion.ddim_sample(
                denoise_fn, v_emb.squeeze(0) if v_emb.dim() > 1 else v_emb,
                self.n_elems, n_steps=n_steps,
                w_phys=w_phys, sigma_warm=sigma_warm,
                sigma_min=self.sigma_min, sigma_max=self.sigma_max,
            )
            samples.append(sigma)

        if n_samples == 1:
            return samples[0].clamp(self.sigma_min, self.sigma_max)
        else:
            return torch.stack(samples)

    def to(self, device):
        super().to(device)
        self.diffusion.to(device)
        return self
