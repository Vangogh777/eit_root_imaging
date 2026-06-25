#!/bin/bash
# 定期评估脚本 - 每10个epoch评估一次训练效果
# 使用方法: bash evaluate_checkpoint.sh epoch_number

EPOCH=$1
CHECKPOINT_DIR="checkpoints/20260622_*_full_fem"  # 自动查找最新训练目录

if [ -z "$EPOCH" ]; then
    echo "用法: bash evaluate_checkpoint.sh <epoch_number>"
    echo "示例: bash evaluate_checkpoint.sh 10"
    exit 1
fi

# 查找最新的checkpoint文件
CHECKPOINT=$(ls -t checkpoints/*/unsup_epoch${EPOCH}.pt 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "错误: 找不到 epoch ${EPOCH} 的checkpoint文件"
    echo "可用的checkpoint:"
    ls -lh checkpoints/*/unsup_epoch*.pt 2>/dev/null
    exit 1
fi

OUTPUT_DIR="results/eval_full_fem_epoch${EPOCH}"
LOG_FILE="${OUTPUT_DIR}/evaluation.log"

echo "=========================================="
echo "评估 Checkpoint: ${CHECKPOINT}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p ${OUTPUT_DIR}

# 使用CPU评估（避免GPU内存冲突）
CUDA_VISIBLE_DEVICES="" python evaluate_current_run.py \
    --checkpoint ${CHECKPOINT} \
    --data data/generated/mixed_dataset.h5 \
    --output ${OUTPUT_DIR} \
    --split val \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "=========================================="
echo "评估完成！结果保存在: ${OUTPUT_DIR}"
echo "=========================================="

# 提取关键指标
if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    echo "关键指标:"
    python -c "
import json
with open('${OUTPUT_DIR}/metrics.json') as f:
    m = json.load(f)['summary']
    print(f\"  RE  = {m['RE']['mean']:.4f} ± {m['RE']['std']:.4f}\")
    print(f\"  CC  = {m['CC']['mean']:.4f} ± {m['CC']['std']:.4f}\")
    if 'SSIM' in m:
        print(f\"  SSIM = {m['SSIM']['mean']:.4f}\")
"
fi

# 对比之前的结果
echo ""
echo "历史对比:"
echo "  监督训练 (best_v2):  RE=0.108, CC=0.976"
echo "  雅可比模式 (epoch20): RE=0.648, CC=-0.01 (失效)"
echo "  训练前 (both_hd256):  RE=0.193, CC=0.955"
