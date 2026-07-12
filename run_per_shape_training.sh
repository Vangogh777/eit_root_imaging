#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Per-Shape Sequential Training — 5 个形状数据集逐一训练
# ═══════════════════════════════════════════════════════════════════════
# 用法:
#   bash run_per_shape_training.sh                   # 在当前终端直接跑
#   tmux new -s per_shape 'bash run_per_shape_training.sh'  # 在 tmux 中跑（推荐）
#
# 每个形状训练完成后自动开始下一个，SSH 断连不影响（在 tmux 中运行即可）
# 模型 checkpoint 自动按形状命名：checkpoints/{YYYYMMDD_HHMMSS}_v2_{shape}_hd512/
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 训练配置 ──────────────────────────────────────────────
MESH_CONFIG="config/mesh_11466_config.yaml"
HIDDEN_DIM=512
BATCH_SIZE=12
GRAD_ACCUM=2
EPOCHS_SUP=80
EPOCHS_UNSUP=200
LR=1e-4
UNSUP_LR=1e-5
MCL_MODE="full_fem"
EARLY_STOP=10

# ── 形状列表（按顺序训练）─────────────────────────────────
SHAPES=(
    "circle"
    "ellipse"
    "double_circle"
    "square"
    "near_boundary"
)

# ── 日志目录 ──────────────────────────────────────────────
LOG_DIR="logs/per_shape"
mkdir -p "$LOG_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Per-Shape Sequential Training — 5 形状逐一训练           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Mesh:      $MESH_CONFIG"
echo "║  Model:     ConvSpatialEIT v2 (hidden_dim=$HIDDEN_DIM)"
echo "║  Batch:     ${BATCH_SIZE} × grad_accum ${GRAD_ACCUM} = 等效 $((BATCH_SIZE * GRAD_ACCUM))"
echo "║  MCL mode:  $MCL_MODE"
echo "║  Epochs:    ${EPOCHS_SUP} sup + ${EPOCHS_UNSUP} unsup"
echo "║  Shapes:    ${SHAPES[*]}"
echo "╚══════════════════════════════════════════════════════════════╝"

TOTAL=${#SHAPES[@]}
START_TIME=$(date +%s)

for i in "${!SHAPES[@]}"; do
    SHAPE="${SHAPES[$i]}"
    IDX=$((i + 1))
    DATA_PATH="data/generated/${SHAPE}_dataset_11466.h5"
    LOG_FILE="${LOG_DIR}/${SHAPE}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  [$IDX/$TOTAL] 开始训练: $SHAPE"
    echo "  数据集: $DATA_PATH"
    echo "  日志:   $LOG_FILE"
    echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════"

    SHAPE_START=$(date +%s)

    set +e  # 单个训练失败不中断后续训练
    python train_conv_spatial.py \
        --mesh_config "$MESH_CONFIG" \
        --data "$DATA_PATH" \
        --hidden_dim "$HIDDEN_DIM" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum_steps "$GRAD_ACCUM" \
        --epochs_sup "$EPOCHS_SUP" \
        --epochs_unsup "$EPOCHS_UNSUP" \
        --lr "$LR" \
        --unsup_lr "$UNSUP_LR" \
        --mcl_mode "$MCL_MODE" \
        --use_fixed_weights \
        --early_stop_patience "$EARLY_STOP" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=$?
    set -e

    SHAPE_END=$(date +%s)
    SHAPE_DUR=$((SHAPE_END - SHAPE_START))
    SHAPE_DUR_FMT=$(printf '%dh %dm %ds' $((SHAPE_DUR/3600)) $(((SHAPE_DUR%3600)/60)) $((SHAPE_DUR%60)))

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "  ✅ [$IDX/$TOTAL] $SHAPE 训练完成 ($SHAPE_DUR_FMT)"
    else
        echo ""
        echo "  ❌ [$IDX/$TOTAL] $SHAPE 训练异常退出 (exit code=$EXIT_CODE, $SHAPE_DUR_FMT)"
        echo "  继续下一个形状..."
    fi
done

END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))
TOTAL_DUR_FMT=$(printf '%dh %dm %ds' $((TOTAL_DUR/3600)) $(((TOTAL_DUR%3600)/60)) $((TOTAL_DUR%60)))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  全部训练结束！                                             ║"
echo "║  总耗时: $TOTAL_DUR_FMT"
echo "║  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 汇总 ──────────────────────────────────────────────────
echo ""
echo "训练日志:"
for SHAPE in "${SHAPES[@]}"; do
    LATEST_LOG=$(ls -t "${LOG_DIR}/${SHAPE}"_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        CKPT=$(grep -oP 'checkpoints/\d+_\d+_v2_'${SHAPE}'_hd\d+' "$LATEST_LOG" 2>/dev/null | head -1 || echo "?")
        STATUS="✅"
        grep -q "exit code" "$LATEST_LOG" 2>/dev/null && STATUS="❌"
        echo "  $STATUS  $SHAPE  →  $CKPT"
    else
        echo "  ?  $SHAPE  →  未找到日志"
    fi
done
