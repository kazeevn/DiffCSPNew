# GeoV2 on LeMat-Bulk: stability slice, property conditioning, classifier-free guidance

**Launched:** 2026-09-29, ASPIRE 2A, one 4x A100-40GB node per run (`aiq3`, chained 24 h links).
**Code:** branch `lemat-geov2-cond`; model and training code at `6dba3dc`, the runs at the
commit that adds this document (each run's W&B config records its `git_commit`).
**W&B:** project `symmetry-advantage/diffcsp`, run names `geov2_lemat_<variant>`, tag `lemat`.
Run ids are in each checkpoint's `wandb_run_id` and are added here when the runs finish.

## Variants

| variant | train data | condition | condition dropout | epochs |
|---|---|---|---|---|
| `a_ehull01` | E_hull <= 0.1 eV/atom only (= `lemat_bulk_fmax1_stress_ehull01`) | none | - | 100 |
| `b_ehull` | all | `energy_above_hull` | 0 | 40 |
| `c_ehull_cfg` | all | `energy_above_hull` | 0.2 (CFG) | 40 |
| `d_eform` | all | `formation_energy_per_atom` | 0 | 40 |
| `e_eform_cfg` | all | `formation_energy_per_atom` | 0.2 (CFG) | 40 |

Model: DiffCSP-GeoV2, 512x6, EMA 0.9999 — the same architecture as the Alex-MP-20 run
(`alex-mp20-geov2`). It was chosen over DiffCSP-Geo, the only model with a published match rate (88.4% best-of-3 on MP-20),
because of its lower MP-20 validation loss; it has no match-rate benchmark yet.

## Data

`WyFormer/data/lemat_bulk_fmax1_stress/{train,val,test}.csv.gz` (the CIFs behind the
WyFormer cache `lemat_bulk_fmax1_stress`), packed by `scripts/platforms/aspire2a/pack_dataset.pbs`
(job 25629806.pbs101) into `DiffCSPNew/cache/lemat_bulk_fmax1_stress_packed`. Each split's
`meta.json` records the source sha256 and the packing commit.

- val and test are identical, id for id, to WyFormer's `split_ids.json` (100,000 each), and
  their E_hull <= 0.1 rows to the `_ehull01` slice's (30,543 / 30,818). Variant a filters by
  the same rule, so it trains on exactly the WyFormer slice.
- Cells above 128 atoms are dropped (0.44% of train, 0.37% of the slice): the full-cell
  graph costs N^2 edges.
- Model selection: a fixed random 20k subset of val (seed 0), E_hull-filtered for variant a.
  The b-e runs also log `val_loss_ehull0.1` on val's stable structures, the same population
  as a's validation, and the CFG runs log `*_uncond` losses with the condition nulled.
  test is not touched.

## Training

- Batches of <= 64,000 full-cell edges (sum N^2) per GPU, ~160 structures at LeMat's mean;
  every rank runs the same number of steps. One A100 sustains ~290k edges/s above ~40k
  edges per batch, so larger budgets buy nothing.
- AdamW, lr 1e-3, weight decay 1e-4, per-step linear warmup (3000 steps), cosine to 1e-6;
  gradient value clipping 0.4 (unchanged from GeoV2).
- Conditioning: each property is clamped (E_hull 0..3, E_form -5..4 eV/atom), expanded on
  64 Gaussians, passed through a zero-initialised MLP and added to the time embedding; a
  learned null token stands for a dropped condition. At init a conditional model equals
  its unconditional counterpart.
- Sampling: `python -m diffcsp.cli.inference genes.json.gz --model geov2 --ckpt_path <ckpt>
  --condition energy_above_hull=0 --guidance_scale <w>`; w = 1 is conditional, w = 0
  unconditional, w > 1 stronger guidance (b and d support w = 1 only).

## Pilot

W&B `gwjo8lgm` (`pilot_geov2_lemat_1gpu`), 2026-09-29, commit `6dba3dc`: the c configuration on one
A100, trained on packed val and validated on packed test. 711 structures/s, peak 6.4 GiB,
no skipped steps. Projected on 4 GPUs: ~32 min/epoch on full LeMat, ~14 min on the slice,
so ~21-23 h per run. The pilot ran one rank, so the 4-rank DDP path was first exercised by
the real runs.

## Restart of variant a (2026-09-30)

`a_ehull01` (W&B `t6l9qjp7`) diverged: train loss fell to 0.399 at epoch 14, then rose to
0.43, 0.42, 0.56 and 0.89 by epoch 18, with no non-finite steps, at lr ~9.3e-4. Best
checkpoint: epoch 12, val loss 0.382 (kept in `runs/diffcsp/geov2_lemat_a_ehull01/`). The run
was stopped at epoch 19 and restarted from scratch as `a_ehull01_lr5e4`: identical except
peak lr 5e-4 (the Alex-MP-20 GeoV2 value), from branch `lemat-geov2-a-lr5e4`. The slice's
larger cells mean ~100 structures per GPU batch against ~160 for the full data, i.e. noisier
gradients at the same lr. The b-e runs stayed at 1e-3.
