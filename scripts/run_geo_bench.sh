#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

CKPT="runs/mp20_geo/geo_mp20_210e_best.pt"
if [ ! -f "$CKPT" ]; then
  CKPT="runs/mp20_geo/geo_mp20_210e.pt"
fi

echo "=== [2/6] Generating benchmark predictions with DiffCSP-Geo using $CKPT ==="
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

echo "=== BENCHMARK PIPELINE COMPLETE ==="
