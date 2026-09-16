#!/usr/bin/env bash
# DiffCSP-GeoV2 (GeoNet): 400-Epoch Matched Training & Benchmark Pipeline
# Evaluated strictly under apples-to-apples matching with DiffCSP-Geo (<=12.28M parameters, <=106k steps).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=20

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mkdir -p runs/mp20_geov2 runs/bench/mp20 logs

echo "=== [1/5] Training DiffCSP-GeoV2 on MP-20 ==="
uv run diffcsp-train \
  --model geov2 \
  --train_csv data/mp-20/train.csv \
  --test_csv data/mp-20/test.csv \
  --epochs 400 \
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
  --ckpt_path runs/mp20_geov2/geov2_mp20_400e.pt

echo "=== [2/5] Generating benchmark predictions with DiffCSP-GeoV2 ==="
CKPT="runs/mp20_geov2/geov2_mp20_400e_best.pt"
if [ ! -f "$CKPT" ]; then
  CKPT="runs/mp20_geov2/geov2_mp20_400e.pt"
fi
echo "Using checkpoint: $CKPT"

uv run python bench/run_diffusion.py \
  --regime geov2 \
  --ckpt "$CKPT" \
  --inits runs/bench/mp20/inits.pkl \
  --batch_size 128 \
  --out runs/bench/mp20/pred_geov2_400e.pkl

echo "=== [3/5] Evaluating with StructureMatcher against Ground Truth ==="
uv run python bench/evaluate.py \
  --inits runs/bench/mp20/inits.pkl \
  --preds "vanilla-diffcsp=runs/bench/mp20/pred_cspnet.pkl" \
          "diffcsp-geo-500e=runs/bench/mp20/pred_geo_500e.pkl" \
          "diffcsp-geov2=runs/bench/mp20/pred_geov2_400e.pkl" \
  --n_jobs 20 \
  --out runs/bench/mp20/results_geov2.pkl

echo "=== [4/5] Degrees of Freedom Analysis ==="
uv run python bench/dof_analysis.py \
  --results runs/bench/mp20/results_geov2.pkl \
  --inits runs/bench/mp20/inits.pkl \
  --out runs/bench/mp20/dof_table_geov2.pkl

echo "=== [5/5] Multiplicity Analysis & Paired McNemar Tests ==="
uv run python bench/multiplicity_analysis.py \
  --inits runs/bench/mp20/inits.pkl \
  --results runs/bench/mp20/results_geov2.pkl \
  --regimes "vanilla-diffcsp" "diffcsp-geo-500e" "diffcsp-geov2" \
  --compare diffcsp-geo-500e diffcsp-geov2

echo "=== [6/6] Structural Diagnostics (Clash Rate & Volume Accuracy) ==="
uv run python bench/structure_diagnostics.py \
  --preds "vanilla=runs/bench/mp20/pred_cspnet.pkl" \
          "diffcsp-geo-500e=runs/bench/mp20/pred_geo_500e.pkl" \
          "diffcsp-geov2=runs/bench/mp20/pred_geov2_400e.pkl"

echo "=== DIFFCSP-GEOV2 BENCHMARK COMPLETE ==="
