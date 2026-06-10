"""
M1 Mac 快速训练脚本
===================
针对 M1 Pro 优化的轻量级训练配置

用法:
    python train_m1.py              # 使用简化模型训练
    python train_m1.py --quick      # 快速测试 (100样本, 10轮)
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.eit_forward import EITForwardSolver
from models.simple_model import SimpleSFSBLC
from models.universal_eit import UniversalPhantomGenerator

def main():
    parser = argparse.ArgumentParser(description="M1 Mac 训练脚本")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 M1 Mac 训练")
    print("=" * 60)

    # 检查设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n设备: {device}")
    if device.type == 'mps':
        print("✅ 使用 Apple Metal 加速")

    # ============ 配置 ============
    if args.quick:
        N_TRAIN = 100
        N_VAL = 20
        N_EPOCHS = 10
    else:
        N_TRAIN = 500
        N_VAL = 100
        N_EPOCHS = 30

    BATCH_SIZE = 16
    LR = 1e-3

    print(f"\n配置: 训练={N_TRAIN}, 验证={N_VAL}, 轮数={N_EPOCHS}")

    # ============ 1. 初始化求解器 ============
    print("\n[1/5] 初始化 EIT 求解器...")
    solver = EITForwardSolver("config/mesh_config.yaml")
    n_elems = solver.n_elems
    n_freq = len(solver.frequencies)
    n_meas = solver.n_measurements
    print(f"  网格: {n_elems} 单元")

    # ============ 2. 生成数据 ============
    print("\n[2/5] 生成数据集...")

    phantom_gen = UniversalPhantomGenerator(
        solver.mesh.node,
        solver.mesh.element,
        domain_radius=solver.cfg['mesh']['radius'],
        sigma_background=0.01,
        sigma_inclusion=0.05
    )

    def generate_sample(seed):
        sigma = phantom_gen.generate_random(seed=seed)
        V = solver.solve_multi_frequency(sigma)
        if np.isnan(V).any():
            V = np.random.randn(n_freq, n_meas).astype(np.float32) * 1e-6
        V_noisy = solver.add_noise(V, noise_db=np.random.uniform(-40, -20))
        return sigma.astype(np.float32), V_noisy.astype(np.float32)

    # 生成训练数据
    print(f"  生成 {N_TRAIN} 训练样本...")
    train_sigmas, train_voltages = [], []
    for i in range(N_TRAIN):
        sigma, V = generate_sample(seed=i)
        train_sigmas.append(sigma)
        train_voltages.append(V)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{N_TRAIN}")

    train_sigmas = np.stack(train_sigmas)
    train_voltages = np.stack(train_voltages)

    # 生成验证数据
    print(f"  生成 {N_VAL} 验证样本...")
    val_sigmas, val_voltages = [], []
    for i in range(N_VAL):
        sigma, V = generate_sample(seed=N_TRAIN + i)
        val_sigmas.append(sigma)
        val_voltages.append(V)
    val_sigmas = np.stack(val_sigmas)
    val_voltages = np.stack(val_voltages)

    # ============ 3. 构建模型 ============
    print("\n[3/5] 构建模型...")
    model = SimpleSFSBLC(
        input_dim=n_meas,
        hidden_dim=256,
        n_frequencies=n_freq,
        n_elems=n_elems,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    # ============ 4. 训练 ============
    print("\n[4/5] 开始训练...")

    train_V = torch.from_numpy(train_voltages).float().to(device)
    train_S = torch.from_numpy(train_sigmas).float().to(device)
    val_V = torch.from_numpy(val_voltages).float().to(device)
    val_S = torch.from_numpy(val_sigmas).float().to(device)

    best_val_loss = float('inf')

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = torch.randperm(N_TRAIN)

        epoch_loss = 0.0
        n_batches = (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE

        for b in range(n_batches):
            idx = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            V_batch = train_V[idx]
            S_batch = train_S[idx]

            optimizer.zero_grad()
            out = model(V_batch)
            loss = torch.nn.functional.mse_loss(out['sigma'], S_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= n_batches
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_out = model(val_V)
            val_loss = torch.nn.functional.mse_loss(val_out['sigma'], val_S).item()
            re = torch.norm(val_out['sigma'] - val_S, dim=-1) / (torch.norm(val_S, dim=-1) + 1e-8)
            val_re = re.mean().item()

        print(f"  Epoch {epoch:2d}/{N_EPOCHS} | Loss: {epoch_loss:.6f} | Val: {val_loss:.6f} | RE: {val_re:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)

    # ============ 5. 测试 ============
    print("\n[5/5] 测试...")
    model.eval()
    with torch.no_grad():
        out = model(val_V[:4])
        for i in range(4):
            re_i = torch.norm(out['sigma'][i] - val_S[i]) / (torch.norm(val_S[i]) + 1e-8)
            print(f"  样本 {i}: RE = {re_i.item():.4f}")

    # ============ 完成 ============
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print(f"📊 最终验证相对误差: {val_re:.4f}")
    print("=" * 60)

    # 保存模型
    save_path = "checkpoints/m1_model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'n_elems': n_elems, 'n_freq': n_freq, 'n_meas': n_meas}
    }, save_path)
    print(f"💾 模型已保存: {save_path}")

if __name__ == "__main__":
    main()
