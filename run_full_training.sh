#!/usr/bin/env bash
# Full MP-20 training run: orb-v3 backbone, option-3 adapter
# (no hard force constraint; ORB forces, stress and node representations
#  all enter as conditioning features).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
exec uv run python train.py \
  --model orb \
  --orb_model orb-v3 \
  --train_csv train.csv \
  --test_csv test.csv \
  --hidden_dim 128 \
  --num_layers 2 \
  --batch_size 128 \
  --lr 1e-3 \
  --epochs 1000 \
  --eval_freq 10 \
  --save_freq 25 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --ckpt_path orb_v3_diffcsp_mp20_opt3.pt \
  --wandb --wandb_project diffcsp --wandb_entity symmetry-advantage
