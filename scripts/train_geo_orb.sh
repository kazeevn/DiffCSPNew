#!/usr/bin/env bash
# GeoOrbCSPNet: Deep MLIP-Conditioned Diffusion with Two-Phase Hybrid Sampling
# Trainable parameters strictly <= 12,283,784, frozen ORB v3 backbone (~25.6M).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=20

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mkdir -p runs/mp20_geo_orb runs/bench/mp20 logs

echo "=== [1/5] Training GeoOrbCSPNet on MP-20 ==="
uv run diffcsp-train \
  --model geo_orb \
  --orb_model orb-v3 \
  --train_csv data/mp-20/train.csv \
  --test_csv data/mp-20/test.csv \
  --epochs 200 \
  --batch_size 64 \
  --lr 5e-4 \
  --eval_freq 10 \
  --save_freq 10 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device cuda \
  --wandb \
  --wandb_project diffcsp \
  --wandb_entity symmetry-advantage \
  --ckpt_path runs/mp20_geo_orb/geo_orb_mp20_200e.pt

echo "=== [2/5] Generating benchmark predictions with Two-Phase Hybrid Sampling ==="
CKPT="runs/mp20_geo_orb/geo_orb_mp20_200e_best.pt"
if [ ! -f "$CKPT" ]; then
  CKPT="runs/mp20_geo_orb/geo_orb_mp20_200e.pt"
fi
echo "Using checkpoint: $CKPT"

# Generative diffusion down to t=0.10, then hand off to symmetry-constrained ORB FIRE relaxation
uv run python bench/run_diffusion.py \
  --regime geo_orb \
  --ckpt "$CKPT" \
  --inits runs/bench/mp20/inits.pkl \
  --batch_size 64 \
  --orb_handoff_t 0.10 \
  --relax_steps 50 \
  --out runs/bench/mp20/pred_geo_orb_hybrid.pkl

echo "=== [3/5] Evaluating with StructureMatcher against Ground Truth ==="
uv run python bench/evaluate.py \
  --inits runs/bench/mp20/inits.pkl \
  --preds "vanilla-diffcsp=runs/bench/mp20/pred_cspnet.pkl" \
          "diffcsp-geo-500e=runs/bench/mp20/pred_geo_500e.pkl" \
          "diffcsp-geov2=runs/bench/mp20/pred_geov2_400e.pkl" \
          "diffcsp-geo-orb=runs/bench/mp20/pred_geo_orb_hybrid.pkl" \
  --n_jobs 20 \
  --out runs/bench/mp20/results_master.pkl

echo "=== [4/5] Degrees of Freedom Analysis ==="
uv run python bench/dof_analysis.py \
  --results runs/bench/mp20/results_master.pkl \
  --inits runs/bench/mp20/inits.pkl \
  --out runs/bench/mp20/dof_table_master.pkl

echo "=== [5/5] Multiplicity Analysis & Paired McNemar Tests ==="
uv run python bench/multiplicity_analysis.py \
  --inits runs/bench/mp20/inits.pkl \
  --results runs/bench/mp20/results_master.pkl \
  --regimes "diffcsp-geo-500e" "diffcsp-geov2" "diffcsp-geo-orb" \
  --compare diffcsp-geo-500e diffcsp-geo-orb

echo "=== [6/6] Structural Diagnostics ==="
uv run python bench/structure_diagnostics.py \
  --preds "diffcsp-geo-500e=runs/bench/mp20/pred_geo_500e.pkl" \
          "diffcsp-geov2=runs/bench/mp20/pred_geov2_400e.pkl" \
          "diffcsp-geo-orb=runs/bench/mp20/pred_geo_orb_hybrid.pkl"

echo "=== GEO-ORB BENCHMARK COMPLETE ==="
