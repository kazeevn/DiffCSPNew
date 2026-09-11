#!/usr/bin/env bash
# Vanilla DiffCSP++ (CSPNet 512/6) on LeMat-Bulk fmax1.
#   E_hull <= 0.1 eV  : 1,584,267 structures preprocessed (32 cached shards)
#   max_atoms <= 128  : keeps 99.42%; CSPNet's intra-cell graph is fully
#                       connected, so a 992-atom conventional cell is ~984k
#                       edges on its own and fits at no batch size.
#   eval on val.csv.gz; test.csv.gz is never seen here.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kna/DiffCSPNew
exec uv run python train.py \
  --model cspnet \
  --hidden_dim 512 --num_layers 6 \
  --train_csv /home/kna/WyckoffTransformer/data/lemat_bulk_fmax1/train.csv.gz \
  --test_csv  /home/kna/WyckoffTransformer/data/lemat_bulk_fmax1/val.csv.gz \
  --max_e_hull 0.1 \
  --max_atoms 128 \
  --cache_dir /home/kna/DiffCSPNew/cache/lemat_bulk_fmax1 \
  --batch_size 256 \
  --lr 1e-3 \
  --epochs 20 \
  --eval_freq 2 \
  --save_freq 5 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --resume auto \
  --ckpt_path cspnet_lemat_fmax1_ehull0p1.pt \
  --wandb --wandb_project diffcsp --wandb_entity symmetry-advantage
