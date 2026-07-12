#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 顺序执行 5 个 shape 的有监督训练 (ConvSpatialEIT, 11466 mesh)
#
# 用法: nohup bash run_per_shape_supervised.sh > train_per_shape.log 2>&1 &
# ═══════════════════════════════════════════════════════════════

set -e  # 某个训练失败即停止

cd /home/ubuntu/EIT/eit_root_imaging

CONFIG="config/train_config_11466.yaml"
MESH_CONFIG="config/mesh_11466_config.yaml"
EPOCHS=80
BATCH_SIZE=16
GRAD_ACCUM=2
LR=1e-4

echo "════════════════════════════════════════════════════════"
echo "Per-Shape Supervised Training — 5 shapes × ${EPOCHS} epochs"
echo "Start: $(date)"
echo "════════════════════════════════════════════════════════"

for SHAPE in circle ellipse double_circle square near_boundary; do
    DATA_FILE="data/generated/${SHAPE}_dataset_11466.h5"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$(date)] Starting: ${SHAPE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python train_conv_spatial.py \
        --mode supervised \
        --config "${CONFIG}" \
        --mesh_config "${MESH_CONFIG}" \
        --data "${DATA_FILE}" \
        --epochs_sup "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --grad_accum_steps "${GRAD_ACCUM}" \
        --lr "${LR}" \
        --ema_decay 0.99 \
        --no_gat

    echo ""
    echo "[$(date)] Completed: ${SHAPE}"
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "All 5 shapes done! $(date)"
echo "════════════════════════════════════════════════════════"
