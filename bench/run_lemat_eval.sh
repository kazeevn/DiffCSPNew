#!/usr/bin/env bash
# Evaluate the LeMat-trained CSPNet on held-out LeMat test, against three references:
#   lemat-cspnet : the model just trained (val 0.2902, epoch 20)
#   mp20-cspnet  : the same architecture trained on MP-20 -- does LeMat training help?
#   orb-relax    : no learning at all, symmetry-constrained ORB relaxation
#   pyxtal-only  : the unrefined from_random draw
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mkdir -p runs/bench/lemat

echo "=== [1/4] lemat-cspnet (this run's best checkpoint) ==="
uv run python bench/run_diffusion.py --regime cspnet \
  --ckpt runs/mqk461a6/cspnet_lemat_fmax1_ehull0p1_best.pt \
  --inits runs/bench/lemat/lemat_inits.pkl --batch_size 128 --out runs/bench/lemat/lemat_pred_lemat.pkl

echo "=== [2/4] mp20-cspnet (test_ckpt.pt) ==="
uv run python bench/run_diffusion.py --regime cspnet \
  --ckpt data/mp-20/test_ckpt.pt \
  --inits runs/bench/lemat/lemat_inits.pkl --batch_size 128 --out runs/bench/lemat/lemat_pred_mp20.pkl

echo "=== [3/4] ORB relaxation ==="
uv run python bench/run_relax.py --inits runs/bench/lemat/lemat_inits.pkl --out runs/bench/lemat/lemat_pred_relax.pkl

echo "=== [4/4] scoring ==="
uv run python bench/evaluate.py --inits runs/bench/lemat/lemat_inits.pkl \
  --preds "pyxtal-only (control)=runs/bench/lemat/lemat_pred_none.pkl" \
          "mp20-cspnet=runs/bench/lemat/lemat_pred_mp20.pkl" \
          "lemat-cspnet=runs/bench/lemat/lemat_pred_lemat.pkl" \
          "orb-relax=runs/bench/lemat/lemat_pred_relax.pkl" \
  --out runs/bench/lemat/lemat_results.pkl
echo "=== EVAL COMPLETE ==="
