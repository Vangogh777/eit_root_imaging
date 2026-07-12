#!/usr/bin/env python3
"""
训练效果实时监控脚本
对比当前训练与历史最佳模型的指标

使用方法:
  python monitor_training.py                    # 查看当前状态
  python monitor_training.py --watch            # 持续监控（每5分钟更新）
"""

import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path

# 历史基准数据
BASELINES = {
    "监督训练最佳 (v2)": {"RE": 0.108, "CC": 0.976, "SSIM": 0.994},
    "雅可比模式 (epoch20)": {"RE": 0.648, "CC": -0.01, "SSIM": None},
    "训练前基线": {"RE": 0.193, "CC": 0.955, "SSIM": None},
}

def get_latest_checkpoint_dir():
    """找到最新的训练目录"""
    checkpoint_dirs = glob.glob("checkpoints/2026*")
    if not checkpoint_dirs:
        return None
    return max(checkpoint_dirs, key=os.path.getmtime)

def get_training_log():
    """读取训练日志"""
    log_files = ["train_full_fem.log", "train.log"]
    for log_file in log_files:
        if os.path.exists(log_file):
            return log_file
    return None

def parse_training_log(log_file):
    """解析训练日志，提取最新进度"""
    if not log_file or not os.path.exists(log_file):
        return None

    with open(log_file, 'r', errors='ignore') as f:
        lines = f.readlines()

    # 提取最近的epoch和loss信息
    current_epoch = None
    current_loss = None

    for line in reversed(lines[-100:]):
        if "Unsup Epoch" in line and "Loss:" in line:
            try:
                parts = line.split("Unsup Epoch")[1].split("|")
                current_epoch = int(parts[0].strip())
                current_loss = float(parts[1].split("Loss:")[1].strip())
                break
            except:
                continue

    return {
        "epoch": current_epoch,
        "loss": current_loss,
        "total_epochs": 50,
        "progress": f"{current_epoch}/50" if current_epoch else "N/A"
    }

def get_evaluation_results(epoch=None):
    """获取评估结果"""
    eval_dirs = sorted(glob.glob("results/eval_full_fem_epoch*"), key=os.path.getmtime)

    if not eval_dirs:
        return None

    if epoch:
        target_dir = f"results/eval_full_fem_epoch{epoch}"
        if os.path.exists(target_dir):
            eval_dirs = [target_dir]
        else:
            return None

    latest_dir = eval_dirs[-1]
    metrics_file = os.path.join(latest_dir, "metrics.json")

    if not os.path.exists(metrics_file):
        return None

    with open(metrics_file) as f:
        data = json.load(f)

    return {
        "dir": latest_dir,
        "metrics": data["summary"]
    }

def print_status():
    """打印训练状态"""
    print("=" * 70)
    print(" 🎯 EIT训练监控面板")
    print("=" * 70)
    print(f" 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 训练进度
    checkpoint_dir = get_latest_checkpoint_dir()
    log_file = get_training_log()
    training_status = parse_training_log(log_file)

    print(" 📊 训练进度:")
    if training_status:
        print(f"   当前Epoch: {training_status['progress']}")
        print(f"   训练损失: {training_status['loss']:.6f}" if training_status['loss'] else "   训练损失: N/A")
        if training_status['epoch']:
            progress_pct = (training_status['epoch'] / training_status['total_epochs']) * 100
            bar_len = 30
            filled_len = int(bar_len * training_status['epoch'] // training_status['total_epochs'])
            bar = '█' * filled_len + '░' * (bar_len - filled_len)
            print(f"   [{bar}] {progress_pct:.1f}%")
    else:
        print("   未检测到训练进程")

    print()

    # 检查点文件
    if checkpoint_dir:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "unsup_epoch*.pt"))
        if checkpoints:
            epochs = sorted([int(f.split("epoch")[1].split(".")[0]) for f in checkpoints])
            print(f" 💾 已保存检查点: Epoch {epochs}")

    print()

    # 最新评估结果
    eval_results = get_evaluation_results()

    print(" 🎨 最新评估结果:")
    if eval_results:
        m = eval_results['metrics']
        print(f"   目录: {eval_results['dir']}")
        print(f"   RE  = {m['RE']['mean']:.4f} ± {m['RE']['std']:.4f}")
        print(f"   CC  = {m['CC']['mean']:.4f} ± {m['CC']['std']:.4f}")
        if 'SSIM' in m:
            print(f"   SSIM = {m['SSIM']['mean']:.4f}")
    else:
        print("   暂无评估结果")

    print()

    # 历史对比
    print(" 📈 历史基准对比:")
    print(f"   {'模型':<25} {'RE':<10} {'CC':<10} {'SSIM':<10}")
    print(f"   {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for name, metrics in BASELINES.items():
        re_str = f"{metrics['RE']:.4f}" if metrics['RE'] else "N/A"
        cc_str = f"{metrics['CC']:.4f}" if metrics['CC'] is not None else "N/A"
        ssim_str = f"{metrics['SSIM']:.4f}" if metrics['SSIM'] else "N/A"
        print(f"   {name:<25} {re_str:<10} {cc_str:<10} {ssim_str:<10}")

    # 如果有评估结果，添加对比
    if eval_results:
        print(f"   {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        m = eval_results['metrics']
        print(f"   {'当前训练 (最新)':<25} {m['RE']['mean']:.4f}     {m['CC']['mean']:.4f}     {m.get('SSIM', {}).get('mean', 'N/A')}")

    print()
    print("=" * 70)

    # 建议
    if training_status and training_status['epoch']:
        if training_status['epoch'] % 10 == 0:
            print(f" 💡 提示: 已完成 {training_status['epoch']} 个epoch，建议运行评估:")
            print(f"    bash evaluate_checkpoint.sh {training_status['epoch']}")

    print()

def watch_mode(interval=300):
    """持续监控模式"""
    import time
    try:
        while True:
            os.system('clear')
            print_status()
            print(f" ⏰ 下次更新: {interval}秒后 (Ctrl+C 退出)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n 监控已停止")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="持续监控模式")
    parser.add_argument("--interval", type=int, default=300, help="监控间隔（秒）")
    args = parser.parse_args()

    if args.watch:
        watch_mode(args.interval)
    else:
        print_status()
