#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# EIT 两阶段训练脚本 (修复反跳问题版本)
# ═══════════════════════════════════════════════════════════════

set -e

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 训练参数 (参考 v2_both_hd512 成功经验)
HIDDEN_DIM=512
EPOCHS_SUP=50
EPOCHS_UNSUP=50
BATCH_SIZE=32
LR=1e-4
EMA_DECAY=0.999  # 长训练用更稳定的 EMA

# 输出日志
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════════════════"
echo "EIT 两阶段训练"
echo "═══════════════════════════════════════════════════════════════"
echo "时间: $TIMESTAMP"
echo "hidden_dim: $HIDDEN_DIM"
echo "epochs_sup: $EPOCHS_SUP"
echo "epochs_unsup: $EPOCHS_UNSUP"
echo "batch_size: $BATCH_SIZE"
echo "ema_decay: $EMA_DECAY"
echo "日志: $LOG_FILE"
echo "═══════════════════════════════════════════════════════════════"

# 运行训练
python train_conv_spatial.py \
    --mode both \
    --hidden_dim $HIDDEN_DIM \
    --epochs_sup $EPOCHS_SUP \
    --epochs_unsup $EPOCHS_UNSUP \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --ema_decay $EMA_DECAY \
    --gnn_layers 4 \
    --n_heads 4 \
    2>&1 | tee $LOG_FILE

echo ""
echo "✅ 训练完成！日志已保存到: $LOG_FILE"
