#!/usr/bin/env bash
# DiffCSP-Geo + M=3 Test-Time ORB Candidate Ranking Benchmark
# Generates M=3 parallel candidates per draw, ranks by frozen ORB potential energy,
# and evaluates match rates, DoF breakdown, McNemar tests, and diagnostics on MP-20.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=20

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mkdir -p runs/bench/mp20 logs

echo "=== [1/4] Generating predictions with DiffCSP-Geo + M=3 ORB Ranking ==="
uv run python bench/run_diffusion.py \
  --regime geo \
  --ckpt runs/mp20_geo/geo_mp20_500e_best.pt \
  --inits runs/bench/mp20/inits.pkl \
  --batch_size 128 \
  --rank_candidates 3 \
  --noise_cutoff_t 50 \
  --step_lr 8e-6 \
  --out runs/bench/mp20/pred_geo_rank3.pkl

echo "=== [2/4] Evaluating with StructureMatcher against MP-20 Ground Truth ==="
uv run python bench/evaluate.py \
  --inits runs/bench/mp20/inits.pkl \
  --preds "vanilla-diffcsp=runs/bench/mp20/pred_cspnet.pkl" \
          "diffcsp-geo-500e=runs/bench/mp20/pred_geo_500e.pkl" \
          "diffcsp-geo-rank3=runs/bench/mp20/pred_geo_rank3.pkl" \
  --n_jobs 20 \
  --out runs/bench/mp20/results_geo_rank3.pkl

echo "=== [3/4] Degrees of Freedom Analysis ==="
uv run python bench/dof_analysis.py \
  --results runs/bench/mp20/results_geo_rank3.pkl \
  --inits runs/bench/mp20/inits.pkl \
  --out runs/bench/mp20/dof_table_geo_rank3.pkl

echo "=== [4/4] Multiplicity Analysis & Paired McNemar Tests ==="
uv run python bench/multiplicity_analysis.py \
  --inits runs/bench/mp20/inits.pkl \
  --results runs/bench/mp20/results_geo_rank3.pkl \
  --regimes "diffcsp-geo-500e" "diffcsp-geo-rank3" \
  --compare diffcsp-geo-500e diffcsp-geo-rank3

echo "=== [5/5] Structural Diagnostics (Clash Rate & Volume Accuracy) ==="
uv run python bench/structure_diagnostics.py \
  --preds "diffcsp-geo-500e=runs/bench/mp20/pred_geo_500e.pkl" \
          "diffcsp-geo-rank3=runs/bench/mp20/pred_geo_rank3.pkl"

echo "=== DIFFCSP-GEO M=3 RANKING BENCHMARK COMPLETE ==="
