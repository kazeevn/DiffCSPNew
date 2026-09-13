#!/usr/bin/env bash
# DiffCSP-Geo: 500-Epoch Matched Training & Benchmark Pipeline
# Matches the full 500-epoch training budget of DiffCSP++ on MP-20 with WanDB logging.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mkdir -p runs/mp20_geo runs/bench/mp20 logs

echo "=== [1/5] Training DiffCSP-Geo to 500 Epochs (Resuming from Epoch 210) ==="
uv run diffcsp-train \
  --model geo \
  --resume runs/mp20_geo/geo_mp20_210e.pt \
  --train_csv data/mp-20/train.csv \
  --test_csv data/mp-20/test.csv \
  --epochs 500 \
  --batch_size 128 \
  --lr 5e-4 \
  --eval_freq 10 \
  --save_freq 10 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device cuda \
  --wandb \
  --wandb_project diffcsp \
  --wandb_entity symmetry-advantage \
  --ckpt_path runs/mp20_geo/geo_mp20_500e.pt

echo "=== [2/5] Generating benchmark predictions with 500-epoch DiffCSP-Geo ==="
CKPT="runs/mp20_geo/geo_mp20_500e_best.pt"
if [ ! -f "$CKPT" ]; then
  CKPT="runs/mp20_geo/geo_mp20_500e.pt"
fi
echo "Using checkpoint: $CKPT"

uv run python bench/run_diffusion.py \
  --regime geo \
  --ckpt "$CKPT" \
  --inits runs/bench/mp20/inits.pkl \
  --batch_size 128 \
  --out runs/bench/mp20/pred_geo_500e.pkl

echo "=== [3/5] Evaluating with StructureMatcher against Ground Truth ==="
uv run python bench/evaluate.py \
  --inits runs/bench/mp20/inits.pkl \
  --preds "vanilla-diffcsp=runs/bench/mp20/pred_cspnet.pkl" \
          "diffcsp-geo-210e=runs/bench/mp20/pred_geo.pkl" \
          "diffcsp-geo-500e=runs/bench/mp20/pred_geo_500e.pkl" \
  --n_jobs 20 \
  --out runs/bench/mp20/results_geo_500e.pkl

echo "=== [4/5] Degrees of Freedom Analysis ==="
uv run python bench/dof_analysis.py \
  --results runs/bench/mp20/results_geo_500e.pkl \
  --inits runs/bench/mp20/inits.pkl \
  --out runs/bench/mp20/dof_table_geo_500e.pkl

echo "=== [5/5] Multiplicity Analysis & Paired McNemar Tests ==="
uv run python bench/multiplicity_analysis.py \
  --inits runs/bench/mp20/inits.pkl \
  --results runs/bench/mp20/results_geo_500e.pkl \
  --regimes "vanilla-diffcsp" "diffcsp-geo-500e" \
  --compare vanilla-diffcsp diffcsp-geo-500e

echo "=== [6/6] Structural Diagnostics (Clash Rate & Volume Accuracy) ==="
uv run python bench/structure_diagnostics.py \
  --preds "vanilla=runs/bench/mp20/pred_cspnet.pkl" \
          "diffcsp-geo-500e=runs/bench/mp20/pred_geo_500e.pkl"

echo "=== 500-EPOCH BENCHMARK COMPLETE ==="
