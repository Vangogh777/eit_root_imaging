"""
EIT 植物根部无监督成像 — 训练入口
====================================
工作流:
  1. python train.py --generate    # 首次：生成数据 + 开始训练
  2. python train.py                # 之后：直接训练（数据已存在）
  3. python train.py --resume ...   # 恢复训练

数据集生成一次即可，后续训练反复使用。
"""

import os
import sys
import subprocess
import argparse
import torch

# 将项目根加入路径
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from training.unsupervised_loop import UnsupervisedTrainer


def check_data_files(config_path: str) -> dict:
    """检查数据文件是否存在，返回状态字典"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg['data']
    h5_path = data_cfg['dataset_path']
    jacobian_path = data_cfg.get('jacobian_path', 'data/generated/jacobian.npy')

    # 解析相对路径（相对于项目根）
    proj_root = os.path.dirname(os.path.abspath(__file__))
    h5_abs = os.path.join(proj_root, h5_path) if not os.path.isabs(h5_path) else h5_path
    jac_abs = os.path.join(proj_root, jacobian_path) if not os.path.isabs(jacobian_path) else jacobian_path

    return {
        'h5_exists': os.path.exists(h5_abs),
        'jacobian_exists': os.path.exists(jac_abs),
        'h5_path': h5_abs,
        'jacobian_path': jac_abs,
    }


def run_data_generation(config_path: str, n_train: int = 10000,
                         n_val: int = 500, n_test: int = 200,
                         workers: int = 0):
    """运行数据生成管线"""
    proj_root = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(proj_root, 'data', 'generate_dataset.py')

    # 数据生成使用 mesh_config.yaml
    mesh_config = os.path.join(proj_root, 'config', 'mesh_config.yaml')

    print("=" * 60)
    print("📦 生成训练数据集（仅首次需要，后续可复用）")
    print("=" * 60)

    cmd = [
        sys.executable, script,
        "--config", mesh_config,
        "--n_train", str(n_train),
        "--n_val", str(n_val),
        "--n_test", str(n_test),
    ]
    if workers > 0:
        cmd += ["--workers", str(workers)]
    result = subprocess.run(cmd, cwd=proj_root)
    if result.returncode != 0:
        print("❌ 数据集生成失败！")
        sys.exit(1)

    print("✅ 数据集生成完成！")


def run_jacobian_precomputation(config_path: str):
    """预计算雅可比矩阵"""
    proj_root = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(proj_root, 'data', 'precompute_jacobian.py')

    # 雅可比计算使用 mesh_config.yaml
    mesh_config = os.path.join(proj_root, 'config', 'mesh_config.yaml')

    print("=" * 60)
    print("⚡ 预计算雅可比矩阵（加速训练）")
    print("=" * 60)

    cmd = [sys.executable, script, "--config", mesh_config]
    result = subprocess.run(cmd, cwd=proj_root)
    if result.returncode != 0:
        print("⚠️  雅可比矩阵预计算失败（训练仍可用线性近似）")
    else:
        print("✅ 雅可比矩阵预计算完成！")


def main():
    parser = argparse.ArgumentParser(description="EIT 无监督训练")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                        help="训练配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖配置文件中的 batch_size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖配置文件中的 epochs")
    parser.add_argument("--generate", action="store_true",
                        help="强制重新生成数据集（默认仅首次缺失时生成）")
    parser.add_argument("--no_jacobian", action="store_true",
                        help="跳过雅可比矩阵预计算")
    parser.add_argument("--n_train", type=int, default=10000,
                        help="训练样本数")
    parser.add_argument("--workers", type=int, default=0,
                        help="数据生成并行进程数 (0=单进程, 建议服务器设为 CPU 核数)")
    args = parser.parse_args()

    # 加载配置
    trainer = UnsupervisedTrainer(args.config)

    # 命令行覆盖
    if args.batch_size:
        trainer.cfg['training']['batch_size'] = args.batch_size
    if args.epochs:
        trainer.cfg['training']['epochs'] = args.epochs

    # ============ 数据准备 ============
    data_status = check_data_files(args.config)

    if args.generate or not data_status['h5_exists']:
        if not args.generate and not data_status['h5_exists']:
            print("🔍 未检测到数据集，自动生成中...")
        n_val = trainer.cfg['data'].get('val_samples', 500)
        n_test = trainer.cfg['data'].get('test_samples', 200)
        run_data_generation(args.config, args.n_train, n_val, n_test,
                            workers=args.workers)
    else:
        h5_rel = os.path.relpath(data_status['h5_path'],
                                 os.path.dirname(os.path.abspath(__file__)))
        print(f"✅ 数据集已存在: {h5_rel}（跳过生成）")

    if not args.no_jacobian and (args.generate or not data_status['jacobian_exists']):
        run_jacobian_precomputation(args.config)
    elif not args.no_jacobian:
        jac_rel = os.path.relpath(data_status['jacobian_path'],
                                   os.path.dirname(os.path.abspath(__file__)))
        print(f"✅ 雅可比矩阵已存在: {jac_rel}（跳过预计算）")

    # ============ 训练 ============
    print("\n" + "=" * 60)
    print("🚀 开始无监督训练")
    print("=" * 60)

    trainer.setup()

    # 恢复 checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint.get('epoch', 0)
        print(f"📂 恢复训练 from epoch {trainer.current_epoch}")

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
