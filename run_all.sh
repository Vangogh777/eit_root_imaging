#!/bin/bash
# ========================================
# 两阶段训练脚本（mixed_dataset 版）
# 两次训练均使用 mixed_dataset.h5，各自保存
# ========================================

cd ~/EIT/eit_root_imaging

# 确保数据存在
echo "检查数据..."
if [ ! -f data/generated/mixed_dataset.h5 ]; then
    echo "生成混合数据..."
    python data/generate_mixed_dataset.py --n_train 20000 --n_val 1000 --n_test 500 --workers 32
fi

# ========== 阶段 1 ==========
echo ""
echo "=========================================="
echo "阶段 1: mixed_dataset (run-1)"
echo "=========================================="

python train_conv_spatial.py \
    --epochs_sup 30 \
    --epochs_unsup 100 \
    --batch_size 64 \
    --lr 3e-4 \
    --wandb

# 备份阶段 1 模型
cp checkpoints/conv_spatial_best.pt checkpoints/conv_spatial_best_run1.pt
cp checkpoints/conv_spatial_final.pt checkpoints/conv_spatial_final_run1.pt
echo "✅  阶段1模型: checkpoints/conv_spatial_best_run1.pt"

# ========== 阶段 2 ==========
echo ""
echo "=========================================="
echo "阶段 2: mixed_dataset (run-2)"
echo "=========================================="

# 随机种子不指定 → 自动不同
python train_conv_spatial.py \
    --epochs_sup 30 \
    --epochs_unsup 100 \
    --batch_size 64 \
    --lr 3e-4 \
    --wandb

# 备份阶段 2 模型
cp checkpoints/conv_spatial_best.pt checkpoints/conv_spatial_best_run2.pt
cp checkpoints/conv_spatial_final.pt checkpoints/conv_spatial_final_run2.pt
echo "✅  阶段2模型: checkpoints/conv_spatial_best_run2.pt"

echo ""
echo "=========================================="
echo "全部完成!"
echo "  run1: checkpoints/conv_spatial_best_run1.pt"
echo "  run2: checkpoints/conv_spatial_best_run2.pt"
echo "=========================================="
