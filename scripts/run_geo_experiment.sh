#!/usr/bin/env bash
# End-to-end training and benchmark pipeline for DiffCSP-Geo on MP-20
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mkdir -p runs/mp20_geo runs/bench/mp20 logs

echo "=== [1/6] Training DiffCSP-Geo (210 Epochs, MP-20) ==="
uv run diffcsp-train \
  --model geo \
  --train_csv data/mp-20/train.csv \
  --test_csv data/mp-20/test.csv \
  --epochs 210 \
  --batch_size 128 \
  --lr 5e-4 \
  --eval_freq 10 \
  --save_freq 10 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --no-wandb \
  --ckpt_path runs/mp20_geo/geo_mp20_210e.pt

echo "=== [2/6] Generating benchmark predictions with DiffCSP-Geo ==="
CKPT="runs/mp20_geo/geo_mp20_210e_best.pt"
if [ ! -f "$CKPT" ]; then
  CKPT="runs/mp20_geo/geo_mp20_210e.pt"
fi

uv run python bench/run_diffusion.py \
  --regime geo \
  --ckpt "$CKPT" \
  --inits runs/bench/mp20/inits.pkl \
  --batch_size 128 \
  --out runs/bench/mp20/pred_geo.pkl

echo "=== [3/6] Evaluating with StructureMatcher against Ground Truth ==="
uv run python bench/evaluate.py \
  --inits runs/bench/mp20/inits.pkl \
  --preds "vanilla-diffcsp=runs/bench/mp20/pred_cspnet.pkl" \
          "diffcsp-geo=runs/bench/mp20/pred_geo.pkl" \
  --n_jobs 20 \
  --out runs/bench/mp20/results_geo.pkl

echo "=== [4/6] Degrees of Freedom Analysis ==="
uv run python bench/dof_analysis.py \
  --results runs/bench/mp20/results_geo.pkl \
  --inits runs/bench/mp20/inits.pkl \
  --out runs/bench/mp20/dof_table_geo.pkl

echo "=== [5/6] Multiplicity Analysis & McNemar Significance Test ==="
uv run python bench/multiplicity_analysis.py \
  --inits runs/bench/mp20/inits.pkl \
  --results runs/bench/mp20/results_geo.pkl \
  --compare vanilla-diffcsp diffcsp-geo

echo "=== [6/6] Structural Diagnostics (Clash Rate & Cell Volume Error) ==="
uv run python bench/structure_diagnostics.py \
  --preds "vanilla=runs/bench/mp20/pred_cspnet.pkl" \
          "diffcsp-geo=runs/bench/mp20/pred_geo.pkl"

echo "=== PIPELINE COMPLETE ==="

