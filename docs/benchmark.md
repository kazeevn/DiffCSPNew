# Benchmarking crystal structure prediction

The harness in `bench/` measures how often a model recovers a known crystal from its
Wyckoff representation, and compares that against references that do less work — so
a match rate can be read against what the representation alone already gives away.

## The pipeline

```
test structure -> Wyckoff representation -> pyxtal.from_random x3 -> regime -> StructureMatcher
                 (pyxtal.from_seed)        (3 instantiations)                 (vs ground truth)
```

Every regime runs on **identical draws**: `crystal_graph_from_pyxtal` (split out of
`build_crystal_graph`) derives model inputs from one specific pyXtal instance rather
than triggering a fresh random draw per regime.

Scored with `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)`, reported two ways:

- **match/trial** — fraction of all generated structures that match.
- **best-of-3** — fraction of representations matched by at least one of their trials.

### Stages

| script | does |
|---|---|
| `build_set.py` | samples test structures, derives Wyckoff representations, keeps matched ground truth |
| `gen_init.py` | the three `from_random` draws per representation |
| `run_none.py` | control: the unrefined draw |
| `run_diffusion.py` | `--regime cspnet` or `orb`, from any checkpoint |
| `run_relax.py` | symmetry-constrained ORB relaxation |
| `evaluate.py` | scores every regime against ground truth |
| `dof_analysis.py` | splits results by Wyckoff degrees of freedom |

`run_all.sh` and `run_lemat_eval.sh` drive the two comparisons below.

## What pyXtal supplies, and what it does not

`sample()` draws initial coordinates from `torch.rand` and the initial lattice from
`torch.randn`, then projects both onto the space group — coordinates through
`batch.ops`/`batch.anchor_index`, the lattice through `proj_k_to_spacegroup`. So
pyXtal supplies the **symmetry scaffold** (space group, Wyckoff orbits, affine
operations, atom count, species); the continuous values are noise. A `from_random`
draw's own coordinates are computed and then discarded.

This has a consequence for the three trials. The `sites` spec fixes the orbits, so
all three draws give an identical scaffold: for the diffusion regimes the trials are
three **sampler seeds**, while for relaxation and the control they are three
genuinely different starting geometries. The comparison is still fair — each regime
gets three attempts per representation — but the trials are not the same object.

---

## Result 1 — MP-20 test, four regimes

1000 representations x 3 trials, `bench/run_all.sh`.

| regime | match/trial | best-of-3 | mean RMS | params |
|---|---|---|---|---|
| pyxtal-only (control) | 22.6% | 40.2% | 0.1824 | — |
| orb-relax | 55.0% | 73.0% | 0.0585 | 0 (25.6M frozen) |
| orb-diffcsp | 77.4% | 84.5% | 0.0415 | 561k trainable |
| **vanilla-diffcsp** | **81.3%** | **86.6%** | **0.0387** | 12.3M |

Vanilla beats the ORB adapter (paired McNemar, per-trial 219 vs 100 discordant,
p = 2.3e-11). The adapter reaches 95% of vanilla's match rate with 22x fewer
trainable parameters — efficient, not better.

### Split by Wyckoff degrees of freedom

DoF is the number of free continuous coordinates summed over sites: what the model
actually has to determine. The aggregate above hides three different regimes.

| DoF | n | atoms | control | orb-relax | orb-diffcsp | vanilla |
|---|---|---|---|---|---|---|
| 0 | 125 | 4.3 | 89.6% | 100.0% | 100.0% | 100.0% |
| 1 | 125 | 7.3 | 55.2% | 94.4% | 100.0% | 100.0% |
| 2 | 191 | 9.3 | 55.0% | 98.4% | 99.0% | 99.5% |
| 3 | 56 | 12.0 | 42.9% | 92.9% | 98.2% | 96.4% |
| 4-5 | 86 | 13.6 | 41.9% | 80.2% | 96.5% | 94.2% |
| 6-8 | 153 | 14.4 | 28.8% | 69.9% | 89.5% | 91.5% |
| 9+ | 264 | 16.1 | 4.5% | 26.9% | 49.6% | 57.2% |

- **Below DoF 3 the potential is enough.** At DoF 0 every regime hits 100% and even
  the unrefined draw manages 89.6%: the representation fixes the coordinates and only
  the cell is free. Relaxation stays within 6 points of the generative models to DoF 2.
- **Between DoF 4 and 8 the diffusion earns its keep.** Its advantage over relaxation
  grows monotonically: +0.0 at DoF 0, +5.6 at 1, +16.3 at 4-5, +19.6 at 6-8, +22.7 at
  9+. Search in high-dimensional configuration space is what it buys, and only that.
- **Above DoF 9 capacity decides.** 51% of vanilla's entire margin over the adapter
  (61 of 119 net discordant trials) comes from this one bin, 26% of the set. Within
  bins, the two are statistically indistinguishable at DoF 3 and 4-5.

Published as an artifact: *Match Rate by Wyckoff DoF*.

---

## Result 2 — LeMat-Bulk test, does more data help?

1000 representations x 3 trials from held-out `test.csv.gz`, filtered to the training
distribution (`E_hull <= 0.1 eV`, `<= 128` atoms). `bench/run_lemat_eval.sh`.

| regime | match/trial | best-of-3 | mean RMS |
|---|---|---|---|
| pyxtal-only (control) | 19.7% | 40.7% | 0.1972 |
| orb-relax | 56.6% | 72.2% | **0.0371** |
| mp20-cspnet | 75.0% | 80.6% | 0.0418 |
| **lemat-cspnet** | **77.4%** | **85.6%** | 0.0404 |

**Yes.** Same 12.3M CSPNet, same sampler, same evaluation — only the training set
differs, and the LeMat-trained model gains 5.0 points best-of-3. Paired McNemar:
best-of-3 77 vs 27 discordant (p = 9.7e-07), per-trial 219 vs 145, net +74
(p = 1.2e-04).

Note **orb-relax has the best mean RMS despite the worst match rate** among trained
regimes. Relaxation either lands in the right basin and refines it precisely or
misses entirely; it has no mechanism for choosing a basin. The diffusion models match
more often and settle slightly less precisely.

The 85.6% here is not comparable to vanilla's 86.6% on MP-20 — different test set,
different filters, 68 vs 78 space groups. The like-for-like comparison is 85.6 vs 80.6
within this table.

---

## Caveats that apply to both results

- **The sets are the round-trippable subsets.** Reaching 1000 representations screened
  2560 candidates on MP-20 and 2304 on LeMat: only ~40% survive
  `from_seed` -> `from_random` at the same site count. Neither is a uniform sample of
  its test split.
- **The ORB backbone has seen this data.** `orb_v3_direct_20_mpa` is trained on
  MPTraj + Alexandria. MP-20 derives from Materials Project and LeMat-Bulk aggregates
  MP + Alexandria + OQMD, so any orb-* number is not a clean generalization
  measurement. For Result 1 this cuts in the reported direction's favour: the adapter
  ran with an advantage vanilla did not have and still lost.
- **`lemat-cspnet` was not trained to convergence.** Its validation loss was still
  falling when the 20-epoch budget ended, so 85.6% is a floor for that configuration
  rather than its ceiling. See `training-stability.md`.
- **Training loss is not match rate.** Vanilla's MP-20 validation loss (0.2505) is
  lower than the ORB adapter's (0.2551) and its match rate is also higher, so the two
  agreed there — but nothing guarantees that, and loss curves alone should not be read
  as generative quality.
