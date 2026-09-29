#!/usr/bin/env bash
# DiffCSP-GeoV2 (512x6, EMA 0.9999) on LeMat-Bulk fmax1_stress, one 4-GPU node per run, ASPIRE 2A.
#
#   bash scripts/train_geov2_lemat.sh <variant> [qsub options]
#
#   a_ehull01    unconditional, E_hull <= 0.1 eV/atom only (= lemat_bulk_fmax1_stress_ehull01)
#   b_ehull      conditioned on energy_above_hull, all structures
#   c_ehull_cfg  as b, condition dropped 20% of the time -> classifier-free guidance
#   d_eform      conditioned on formation_energy_per_atom, all structures
#   e_eform_cfg  as d, condition dropped 20% of the time -> classifier-free guidance
#
# Data: packed from $SRC (the CSVs behind the WyFormer cache lemat_bulk_fmax1_stress) by
# scripts/platforms/aspire2a/pack_dataset.pbs. Train on train/, select on a fixed random
# 20k subset of val/ (the E_hull filter applies to it for a_ehull01); test/ is untouched.
# The b-e runs also report val loss on E_hull <= 0.1 val structures, comparable with a.
#
# Batches: <= 64k full-cell edges (sum N^2) per GPU, ~160 structures at LeMat's mean;
# cells above 128 atoms are dropped (0.44% of train). LR 1e-3, linear warmup over 3000
# steps then cosine to 1e-6, per step. Epoch counts give each run ~20-24 h on 4 A100s.
set -euo pipefail
variant=${1:?variant: a_ehull01 | b_ehull | c_ehull_cfg | d_eform | e_eform_cfg}
shift

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
DATA=/home/project/11001786/WyFormer/DiffCSPNew/cache/lemat_bulk_fmax1_stress_packed

common="--model geov2 --data_dir $DATA --max_atoms 128 --max_edges_per_batch 64000"
common="$common --lr 1e-3 --warmup_steps 3000 --max_test_samples 20000"
common="$common --num_workers 8 --prefetch_factor 4"
common="$common --wandb --wandb_project diffcsp --wandb_entity symmetry-advantage --wandb_tags lemat geov2"
full="--epochs 40 --eval_freq 2 --save_freq 4 --extra_val_max_e_hull 0.1"

case $variant in
    a_ehull01)   args="--max_e_hull 0.1 --epochs 100 --eval_freq 4 --save_freq 8" ;;
    b_ehull)     args="$full --cond_props energy_above_hull" ;;
    c_ehull_cfg) args="$full --cond_props energy_above_hull --cond_drop_prob 0.2" ;;
    d_eform)     args="$full --cond_props formation_energy_per_atom" ;;
    e_eform_cfg) args="$full --cond_props formation_energy_per_atom --cond_drop_prob 0.2" ;;
    *) echo "unknown variant $variant" >&2; exit 1 ;;
esac
RUN_NAME=geov2_lemat_$variant
for split in train val; do
    [ -f "$DATA/$split/meta.json" ] || { echo "missing packed split $DATA/$split" >&2; exit 1; }
done

TRAIN_ARGS="$common $args --wandb_name $RUN_NAME"
# Extra arguments go to qsub, e.g. `-W depend=afterok:<job>`.
qsub -N "dcsp_$variant" -l select=1:ngpus=4:ncpus=64:mem=440gb "$@" \
    -v "REPO=$repo_root,RUN_NAME=$RUN_NAME,TRAIN_ARGS=$TRAIN_ARGS,NGPUS=4" \
    "$repo_root/scripts/platforms/aspire2a/train_chain.pbs"
