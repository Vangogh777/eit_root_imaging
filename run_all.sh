#!/bin/bash
# ========================================
# 两阶段训练脚本：先跑圆形，再跑正方形
# 各自模型分开保存，互不覆盖
# ========================================

cd ~/EIT/eit_root_imaging

# 确保数据存在
echo "检查数据..."
if [ ! -f data/generated/circle_dataset.h5 ]; then
    echo "生成圆形数据..."
    python data/generate_circle_dataset.py --n_train 20000 --n_val 1000 --n_test 500 --workers 32
fi

if [ ! -f data/generated/square_dataset.h5 ]; then
    echo "生成正方形数据..."
    python data/generate_square_dataset.py --n_train 20000 --n_val 1000 --n_test 500 --workers 32
fi

# ========== 阶段 1：圆形 ==========
echo ""
echo "=========================================="
echo "阶段 1: 圆形内含物"
echo "=========================================="

# 确保当前指向圆形
sed -i 's|data/generated/square_dataset.h5|data/generated/circle_dataset.h5|' train_conv_spatial.py 2>/dev/null || true
sed -i 's|from data.generate_square_dataset import generate_dataset|from data.generate_circle_dataset import generate_dataset|' train_conv_spatial.py 2>/dev/null || true

python train_conv_spatial.py \
    --epochs_sup 30 \
    --epochs_unsup 100 \
    --batch_size 64 \
    --lr 3e-4 \
    --wandb

# 备份圆形模型
cp checkpoints/conv_spatial_best.pt checkpoints/conv_spatial_best_circle.pt
cp checkpoints/conv_spatial_final.pt checkpoints/conv_spatial_final_circle.pt
echo "✅ 圆形模型已保存: checkpoints/conv_spatial_best_circle.pt"

# ========== 阶段 2：正方形 ==========
echo ""
echo "=========================================="
echo "阶段 2: 正方形内含物"
echo "=========================================="

# 切换数据路径到正方形
sed -i 's|data/generated/circle_dataset.h5|data/generated/square_dataset.h5|' train_conv_spatial.py
sed -i 's|from data.generate_circle_dataset import|from data.generate_square_dataset import|' train_conv_spatial.py

python train_conv_spatial.py \
    --epochs_sup 30 \
    --epochs_unsup 100 \
    --batch_size 64 \
    --lr 3e-4 \
    --wandb

# 备份正方形模型
cp checkpoints/conv_spatial_best.pt checkpoints/conv_spatial_best_square.pt
cp checkpoints/conv_spatial_final.pt checkpoints/conv_spatial_final_square.pt
echo "✅ 正方形模型已保存: checkpoints/conv_spatial_best_square.pt"

# 路径恢复为圆形（默认）
sed -i 's|data/generated/square_dataset.h5|data/generated/circle_dataset.h5|' train_conv_spatial.py
sed -i 's|from data.generate_square_dataset import|from data.generate_circle_dataset import|' train_conv_spatial.py

echo ""
echo "=========================================="
echo "全部完成!"
echo "  圆形模型:   checkpoints/conv_spatial_best_circle.pt"
echo "  正方形模型: checkpoints/conv_spatial_best_square.pt"
echo "=========================================="
