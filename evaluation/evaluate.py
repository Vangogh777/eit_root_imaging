"""
评估入口
========
在测试集上评估已训练的模型。
用法:
    python evaluate.py --checkpoint checkpoints/...pt --split test
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sf_sblc import SFSBLC
from data.datasets.eit_dataset import EITDataset
from evaluation.metrics import compute_all_metrics
from evaluation.visualize import EITVisualizer


def evaluate(checkpoint_path: str, config_path: str = "config/train_config.yaml",
             split: str = "test", batch_size: int = 64,
             output_dir: str = "results", visualize: bool = True):
    """在测试集上评估模型"""

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型
    model_cfg = cfg['model']
    model = SFSBLC(
        input_dim=model_cfg['input_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        n_frequencies=model_cfg['n_frequencies'],
        n_elems=model_cfg.get('n_elems', 1500),
        n_res_blocks=model_cfg.get('n_res_blocks', 8),
        dropout=model_cfg.get('dropout', 0.1),
        use_attention=model_cfg.get('use_attention', True),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"加载 checkpoint: {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")

    # 加载数据集
    data_cfg = cfg['data']
    h5_path = data_cfg.get(f'{split}_dataset_path') or data_cfg['dataset_path']
    dataset = EITDataset(h5_path, split=split, load_sigmas=True, load_masks=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # 准备可视化器
    centers = np.mean(dataset.mesh_nodes[dataset.mesh_elements], axis=1)
    visualizer = EITVisualizer(centers, dataset.mesh_nodes,
                                dataset.mesh_elements)

    os.makedirs(output_dir, exist_ok=True)

    # 逐批评估
    all_preds = []
    all_gt = []
    all_masks_pred = []
    all_masks_gt = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            voltages = batch['voltages'].to(device)
            sigmas_gt = batch['sigmas'].cpu().numpy()
            masks_gt = batch['masks'].cpu().numpy()

            out = model(voltages)
            sigmas_pred = out['sigma'].cpu().numpy()

            # 从电导率推断根掩码
            threshold = cfg['model'].get('root_threshold', 0.02)
            masks_pred = (sigmas_pred > threshold).astype(np.float32)

            all_preds.append(sigmas_pred)
            all_gt.append(sigmas_gt)
            all_masks_pred.append(masks_pred)
            all_masks_gt.append(masks_gt)

    all_preds = np.concatenate(all_preds, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    all_masks_pred = np.concatenate(all_masks_pred, axis=0)
    all_masks_gt = np.concatenate(all_masks_gt, axis=0)

    # 计算指标
    print("\n=== 评估结果 ===")
    metrics = compute_all_metrics(all_preds, all_gt,
                                   all_masks_pred, all_masks_gt)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # 保存指标
    import json
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # 可视化
    if visualize:
        print("\n生成可视化...")
        n_vis = min(8, len(all_preds))
        visualizer.plot_batch(
            all_gt[:n_vis], all_preds[:n_vis],
            save_path=os.path.join(output_dir, "reconstruction_comparison.png")
        )

        # 单样本详细对比
        for i in range(min(3, n_vis)):
            visualizer.plot_comparison(
                all_gt[i], all_preds[i],
                title=f"Sample {i}",
                save_path=os.path.join(output_dir, f"sample_{i}_comparison.png")
            )

        # Top-K 和 Worst-K
        errors = np.mean((all_preds - all_gt) ** 2, axis=1)
        best_idx = np.argsort(errors)[:4]
        worst_idx = np.argsort(errors)[-4:]

        visualizer.plot_batch(
            all_gt[best_idx], all_preds[best_idx],
            save_path=os.path.join(output_dir, "best_4.png")
        )
        visualizer.plot_batch(
            all_gt[worst_idx], all_preds[worst_idx],
            save_path=os.path.join(output_dir, "worst_4.png")
        )

        print(f"可视化已保存到 {output_dir}/")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--no_vis", action="store_true")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        output_dir=args.output,
        visualize=not args.no_vis,
    )
