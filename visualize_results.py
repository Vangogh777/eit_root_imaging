"""
可视化重建结果
==============
查看训练好的模型的重建效果

用法:
    python visualize_results.py                    # 使用默认模型
    python visualize_results.py --model xxx.pt     # 指定模型
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.eit_forward import EITForwardSolver
from models.simple_model import SimpleSFSBLC
from models.universal_eit import UniversalPhantomGenerator

def main():
    parser = argparse.ArgumentParser(description="可视化重建结果")
    parser.add_argument("--model", type=str, default="checkpoints/m1_model.pt", help="模型路径")
    parser.add_argument("--n_samples", type=int, default=6, help="展示样本数")
    args = parser.parse_args()

    print("=" * 60)
    print("📊 可视化重建结果")
    print("=" * 60)

    # ============ 1. 加载模型 ============
    print(f"\n[1/3] 加载模型: {args.model}")

    checkpoint = torch.load(args.model, map_location='cpu')
    config = checkpoint['config']

    # 从模型权重推断 hidden_dim
    state_dict = checkpoint['model_state_dict']
    if 'encoder.0.weight' in state_dict:
        hidden_dim = state_dict['encoder.0.weight'].shape[0]
    else:
        hidden_dim = config.get('hidden_dim', 256)

    print(f"  配置: n_elems={config['n_elems']}, n_freq={config['n_freq']}, n_meas={config['n_meas']}, hidden_dim={hidden_dim}")

    model = SimpleSFSBLC(
        input_dim=config['n_meas'],
        hidden_dim=hidden_dim,
        n_frequencies=config['n_freq'],
        n_elems=config['n_elems'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("  ✅ 模型加载成功")

    # ============ 2. 生成测试数据 ============
    print("\n[2/3] 生成测试数据...")

    solver = EITForwardSolver("config/mesh_config.yaml")

    phantom_gen = UniversalPhantomGenerator(
        solver.mesh.node,
        solver.mesh.element,
        domain_radius=solver.cfg['mesh']['radius'],
        sigma_background=0.01,
        sigma_inclusion=0.05
    )

    # 生成测试样本
    test_sigmas = []
    test_voltages = []
    for i in range(args.n_samples):
        sigma = phantom_gen.generate_random(seed=1000 + i)
        V = solver.solve_multi_frequency(sigma)
        if np.isnan(V).any():
            V = np.random.randn(config['n_freq'], config['n_meas']).astype(np.float32) * 1e-6
        V_noisy = solver.add_noise(V, noise_db=-30)
        test_sigmas.append(sigma)
        test_voltages.append(V_noisy)

    test_sigmas = np.stack(test_sigmas)
    test_voltages = np.stack(test_voltages)

    # ============ 3. 重建并可视化 ============
    print("\n[3/3] 重建并可视化...")

    with torch.no_grad():
        V_tensor = torch.from_numpy(test_voltages).float()
        out = model(V_tensor)
        pred_sigmas = out['sigma'].numpy()

    # 计算误差
    results = []
    for i in range(args.n_samples):
        gt = test_sigmas[i]
        pred = pred_sigmas[i]
        re = np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-8)
        mse = np.mean((pred - gt) ** 2)
        results.append({'re': re, 'mse': mse})
        print(f"  样本 {i}: RE = {re:.4f}, MSE = {mse:.6f}")

    # ============ 可视化 ============
    # 获取网格中心坐标
    centers = np.mean(solver.mesh.node[solver.mesh.element], axis=1)

    # 创建图形
    fig, axes = plt.subplots(args.n_samples, 3, figsize=(12, 4*args.n_samples))
    if args.n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(args.n_samples):
        gt = test_sigmas[i]
        pred = pred_sigmas[i]
        err = np.abs(pred - gt)

        # Ground Truth
        ax = axes[i, 0]
        sc = ax.scatter(centers[:, 0], centers[:, 1], c=gt, s=30, cmap='viridis',
                       vmin=0.005, vmax=0.1)
        ax.set_title(f"Ground Truth [{i}]")
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='σ (S/m)')

        # Prediction
        ax = axes[i, 1]
        sc = ax.scatter(centers[:, 0], centers[:, 1], c=pred, s=30, cmap='viridis',
                       vmin=0.005, vmax=0.1)
        ax.set_title(f"Prediction [{i}] (RE={results[i]['re']:.3f})")
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='σ (S/m)')

        # Error
        ax = axes[i, 2]
        sc = ax.scatter(centers[:, 0], centers[:, 1], c=err, s=30, cmap='hot',
                       vmin=0, vmax=0.02)
        ax.set_title(f"Absolute Error [{i}]")
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='|σ_pred - σ_gt|')

    plt.tight_layout()

    # 保存图像
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "reconstruction_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 图像已保存: {save_path}")

    # 显示图像
    plt.show()

    # ============ 统计信息 ============
    print("\n" + "=" * 60)
    print("📈 统计信息")
    print("=" * 60)
    avg_re = np.mean([r['re'] for r in results])
    avg_mse = np.mean([r['mse'] for r in results])
    print(f"  平均相对误差 (RE): {avg_re:.4f}")
    print(f"  平均均方误差 (MSE): {avg_mse:.6f}")
    print(f"  σ_gt 范围: [{test_sigmas.min():.4f}, {test_sigmas.max():.4f}]")
    print(f"  σ_pred 范围: [{pred_sigmas.min():.4f}, {pred_sigmas.max():.4f}]")

if __name__ == "__main__":
    main()
