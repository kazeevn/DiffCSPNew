#!/usr/bin/env bash
# Evaluate the LeMat-trained CSPNet on held-out LeMat test, against three references:
#   lemat-cspnet : the model just trained (val 0.2902, epoch 20)
#   mp20-cspnet  : the same architecture trained on MP-20 -- does LeMat training help?
#   orb-relax    : no learning at all, symmetry-constrained ORB relaxation
#   pyxtal-only  : the unrefined from_random draw
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /home/kna/DiffCSPNew

echo "=== [1/4] lemat-cspnet (this run's best checkpoint) ==="
uv run python bench/run_diffusion.py --regime cspnet \
  --ckpt cspnet_lemat_fmax1_ehull0p1_best.pt \
  --inits bench/lemat_inits.pkl --batch_size 128 --out bench/lemat_pred_lemat.pkl

echo "=== [2/4] mp20-cspnet (test_ckpt.pt) ==="
uv run python bench/run_diffusion.py --regime cspnet \
  --ckpt test_ckpt.pt \
  --inits bench/lemat_inits.pkl --batch_size 128 --out bench/lemat_pred_mp20.pkl

echo "=== [3/4] ORB relaxation ==="
uv run python bench/run_relax.py --inits bench/lemat_inits.pkl --out bench/lemat_pred_relax.pkl

echo "=== [4/4] scoring ==="
uv run python bench/evaluate.py --inits bench/lemat_inits.pkl \
  --preds "pyxtal-only (control)=bench/lemat_pred_none.pkl" \
          "mp20-cspnet=bench/lemat_pred_mp20.pkl" \
          "lemat-cspnet=bench/lemat_pred_lemat.pkl" \
          "orb-relax=bench/lemat_pred_relax.pkl" \
  --out bench/lemat_results.pkl
echo "=== EVAL COMPLETE ==="
