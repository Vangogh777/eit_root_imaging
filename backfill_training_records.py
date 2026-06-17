#!/usr/bin/env python3
"""
导入已有训练日志到 training_records/
=====================================
解析 training_v2_both_bs32.log 并生成结构化记录文件。
"""

import os, sys, re, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.recorder import TrainingRecorder, RECORDS_DIR

LOG_FILE = "training_v2_both_bs32.log"
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)

if not os.path.exists(log_path):
    print(f"日志文件不存在: {log_path}")
    sys.exit(1)

with open(log_path) as f:
    lines = f.readlines()

# ============ 从日志解析信息 ============

# 参数量
params = None
for line in lines:
    m = re.search(r'参数量:\s*([0-9,]+)', line)
    if m:
        params = int(m.group(1).replace(',', ''))
        break

# 超参数（从日志中 try 提取）
hidden_dim = 512  # 已知

# 创建记录器
recorder = TrainingRecorder(
    name="v2_both_512",
    run_id="20260616_v2_both_512",
)

meta = {
    "hidden_dim": 512,
    "gnn_hidden": 512,
    "gnn_layers": 4,
    "batch_size": 32,
    "mode": "both",
    "epochs_sup": 50,
    "epochs_unsup": 50,
    "model_params": params or 4195153,
    "best_re": 0.1082,
    "best_cc": 0.9761,
    "start_time": "2026-06-16T22:00:00",
    "end_time": "2026-06-17T06:02:00",
    "log_file": LOG_FILE,
    "status": "completed",
}
recorder.save_meta(meta)

# 解析逐 epoch 数据
sup_phase = True
epoch_count = 0

for line in lines:
    # 阶段切换
    if "阶段 2: 无监督" in line:
        sup_phase = False
        epoch_count = 0

    # 有监督 epoch: "  Epoch  1 | Loss: 0.000040 | Val: 0.000010 | RE: 0.2235"
    m = re.search(r'^\s+Epoch\s+(\d+)\s+\|\s+Loss:\s+([0-9.]+)\s+\|\s+Val:\s+([0-9.]+)\s+\|\s+RE:\s+([0-9.]+)', line)
    if m and "Unsup" not in line:
        phase = "supervised"
        epoch = int(m.group(1))
        loss = float(m.group(2))
        val_loss = float(m.group(3))
        re_val = float(m.group(4))
        recorder.log_epoch(phase=phase, epoch=epoch,
                          loss=loss, val_loss=val_loss, re=re_val)
        continue

    # 有监督 "→ 保存最佳模型 (RE=0.1082)"
    m = re.search(r'保存最佳模型 \(RE=([0-9.]+)\)', line)
    if m:
        recorder.log_event("best_model_saved", re=float(m.group(1)))
        continue

    # 无监督 epoch: "Unsup Epoch 50 | Loss: 0.2488"
    m = re.search(r'Unsup Epoch\s+(\d+)\s+\|\s+Loss:\s+([0-9.]+)', line)
    if m:
        phase = "unsupervised"
        epoch = int(m.group(1))
        loss = float(m.group(2))
        recorder.log_epoch(phase=phase, epoch=epoch, loss=loss)
        continue

    # 无监督 checkpoint
    m = re.search(r'已保存: (checkpoints/conv_spatial_unsup_epoch(\d+)\.pt)', line)
    if m:
        recorder.log_event("checkpoint_saved",
                          path=m.group(1), epoch=int(m.group(2)))
        continue

# 保存最终事件
recorder.log_event("training_completed", best_re=0.1082,
                   best_model="checkpoints/conv_spatial_best.pt")

print(f"✅ 已导入训练记录: {recorder.run_id}")
print(f"   目录: {recorder.run_dir}")
print(f"   元数据: {recorder._meta_file}")
print(f"   epoch 记录: {recorder._epoch_file}")

# 验证
from training.recorder import load_run_data
data = load_run_data(recorder.run_id)
sup = [e for e in data["epochs"] if e["phase"] == "supervised"]
unsup = [e for e in data["epochs"] if e["phase"] == "unsupervised"]
print(f"   有监督 epochs: {len(sup)}")
print(f"   无监督 epochs: {len(unsup)}")
print(f"   事件数: {len(data['events'])}")
