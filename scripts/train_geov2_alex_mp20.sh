#!/usr/bin/env bash
# DiffCSP-GeoV2 (12.3M, 512x6, AdamW + EMA 0.9999 + warmup/cosine) on Alex-MP-20, ASPIRE 2A.
#
#   Alex-MP-20 (MatterGen data release): MP-20 + Alexandria, <= 20 atoms, E_hull < 0.1 eV;
#   train.csv.gz 607,683, val.csv.gz 67,521. Validation (model selection, EMA weights) on val;
#   the release has no test split. NB 6,831 of the 9,046 MP-20 test structures are in this
#   train split, so an MP-20 test benchmark of this model is not a held-out measurement.
#
#   Budget: 150 epochs x ~4.75k steps (bs 128) ~ 712k steps. On MP-20 GeoV2 peaked at ~64k
#   steps / 300 passes (W&B say1phjr); at 20x the data 150 passes should not overfit, and
#   the cosine schedule is fixed by --epochs, so it is set once here. On one A100-40GB
#   (3.2 steps/s measured) that is ~25 min/epoch, ~62 h: three chained 24 h links.
#
# Build the graph caches first (CPU work, resumable):
#   bash scripts/platforms/aspire2a/run_in_singularity.sh python scripts/preprocess_dataset.py \
#       $DATA/val.csv.gz --mode test_sym --cache_dir $CACHE
#   bash scripts/platforms/aspire2a/run_in_singularity.sh python scripts/preprocess_dataset.py \
#       $DATA/train.csv.gz --mode train_sym --cache_dir $CACHE
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
DATA=/home/project/11001786/WyFormer/WyckoffTransformer/data/alex_mp_20
CACHE=/home/project/11001786/WyFormer/DiffCSPNew/cache/alex_mp_20
RUN_NAME=geov2_alex_mp20_150e

for m in train_sym test_sym; do
    [ -f "$CACHE/${m}_shards/manifest.json" ] || { echo "missing graph cache $CACHE/${m}_shards" >&2; exit 1; }
done

TRAIN_ARGS="--model geov2 --train_csv $DATA/train.csv.gz --test_csv $DATA/val.csv.gz --cache_dir $CACHE"
TRAIN_ARGS="$TRAIN_ARGS --epochs 150 --batch_size 128 --lr 5e-4 --eval_freq 5 --save_freq 5"
TRAIN_ARGS="$TRAIN_ARGS --num_workers 6 --prefetch_factor 4"
TRAIN_ARGS="$TRAIN_ARGS --wandb --wandb_project diffcsp --wandb_entity symmetry-advantage"

qsub -N "dcsp_$RUN_NAME" \
    -v "REPO=$repo_root,RUN_NAME=$RUN_NAME,TRAIN_ARGS=$TRAIN_ARGS" \
    "$repo_root/scripts/platforms/aspire2a/train_chain.pbs"
