#!/usr/bin/env bash
# Full three-regime benchmark on an MP-20 test subset.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mkdir -p runs/bench/mp20

echo "=== [1/6] pyxtal.from_random draws ==="
uv run python bench/gen_init.py --benchset runs/bench/mp20/benchset.pkl --trials 3 --out runs/bench/mp20/inits.pkl

echo "=== [2/6] control: unrefined draws ==="
uv run python bench/run_none.py --inits runs/bench/mp20/inits.pkl --out runs/bench/mp20/pred_none.pkl

echo "=== [3/6] regime 1: orb-diffcsp (epoch-500 best) ==="
uv run python bench/run_diffusion.py --regime orb \
  --ckpt runs/vuvt0kab/model-vuvt0kab:v19/orb_v3_diffcsp_mp20_opt3.pt --inits runs/bench/mp20/inits.pkl --batch_size 128 \
  --hidden_dim 128 --num_layers 2 --orb_model orb-v3 --out runs/bench/mp20/pred_orb.pkl

echo "=== [4/6] regime 2: vanilla diffcsp (test_ckpt.pt) ==="
uv run python bench/run_diffusion.py --regime cspnet \
  --ckpt data/mp-20/test_ckpt.pt --inits runs/bench/mp20/inits.pkl --batch_size 128 --out runs/bench/mp20/pred_cspnet.pkl

echo "=== [5/6] regime 3: ORB relaxation ==="
uv run python bench/run_relax.py --inits runs/bench/mp20/inits.pkl --out runs/bench/mp20/pred_relax.pkl

echo "=== [6/6] scoring ==="
uv run python bench/evaluate.py --inits runs/bench/mp20/inits.pkl \
  --preds "pyxtal-only (control)=runs/bench/mp20/pred_none.pkl" \
          "orb-diffcsp=runs/bench/mp20/pred_orb.pkl" \
          "vanilla-diffcsp=runs/bench/mp20/pred_cspnet.pkl" \
          "orb-relax=runs/bench/mp20/pred_relax.pkl" \
  --out runs/bench/mp20/results.pkl
echo "=== BENCHMARK COMPLETE ==="
