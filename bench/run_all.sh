#!/usr/bin/env bash
# Full three-regime benchmark on an MP-20 test subset.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /home/kna/DiffCSPNew

echo "=== [1/6] pyxtal.from_random draws ==="
uv run python bench/gen_init.py --benchset bench/benchset.pkl --trials 3 --out bench/inits.pkl

echo "=== [2/6] control: unrefined draws ==="
uv run python bench/run_none.py --inits bench/inits.pkl --out bench/pred_none.pkl

echo "=== [3/6] regime 1: orb-diffcsp (epoch-500 best) ==="
uv run python bench/run_diffusion.py --regime orb \
  --ckpt "/home/kna/DiffCSPNew/artifacts/model-vuvt0kab:v19/orb_v3_diffcsp_mp20_opt3.pt" --inits bench/inits.pkl --batch_size 128 \
  --hidden_dim 128 --num_layers 2 --orb_model orb-v3 --out bench/pred_orb.pkl

echo "=== [4/6] regime 2: vanilla diffcsp (test_ckpt.pt) ==="
uv run python bench/run_diffusion.py --regime cspnet \
  --ckpt test_ckpt.pt --inits bench/inits.pkl --batch_size 128 --out bench/pred_cspnet.pkl

echo "=== [5/6] regime 3: ORB relaxation ==="
uv run python bench/run_relax.py --inits bench/inits.pkl --out bench/pred_relax.pkl

echo "=== [6/6] scoring ==="
uv run python bench/evaluate.py --inits bench/inits.pkl \
  --preds "pyxtal-only (control)=bench/pred_none.pkl" \
          "orb-diffcsp=bench/pred_orb.pkl" \
          "vanilla-diffcsp=bench/pred_cspnet.pkl" \
          "orb-relax=bench/pred_relax.pkl" \
  --out bench/results.pkl
echo "=== BENCHMARK COMPLETE ==="
