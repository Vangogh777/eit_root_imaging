#!/bin/bash
# 启动完整FEM无监督训练 (batch_size=8, 等效32)
cd /home/ubuntu/EIT/eit_root_imaging
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python -u train_conv_spatial.py \
  --mode unsupervised \
  --mcl_mode full_fem \
  --resume checkpoints/20260622_015538_v2_both_hd256/best.pt \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --epochs_unsup 50 \
  --hidden_dim 256 \
  --no_gat \
  --ema_decay 0.999 \
  > train_full_fem.log 2>&1 &
echo "PID: $!"
echo "日志: tail -f train_full_fem.log"
