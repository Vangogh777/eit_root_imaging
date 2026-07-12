#!/bin/bash
cd /home/ubuntu/EIT/eit_root_imaging
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 train_conv_spatial.py \
  --epochs_sup 50 \
  --epochs_unsup 200 \
  --batch_size 12 \
  --grad_accum_steps 3 \
  --lr 1e-4 \
  --hidden_dim 256 \
  --gnn_layers 4 \
  --edge_ratio 0.5 \
  --use_model_jacobian \
  --workers 8 \
  > logs/conv_spatial_v3_absvoltage.log 2>&1
