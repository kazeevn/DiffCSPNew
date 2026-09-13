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
| vanilla-diffcsp | 81.3% | 86.6% | 0.0387 | 12.28M |
| **diffcsp-geo (500e)** | **83.5%** | **88.4%** | **0.0354** | **12.28M** |

DiffCSP-Geo strictly outperforms vanilla DiffCSP++ across the entire dataset in an apples-to-apples comparison with identical parameter count (12.28M) and matched gradient steps (paired McNemar per-trial: 136 Geo wins vs 72 Vanilla wins, $p = 1.08 \times 10^{-5}$; best-of-3: 39 Geo wins vs 21 Vanilla wins, $p = 0.0273$).

### Split by Wyckoff degrees of freedom

DoF is the number of free continuous coordinates summed over sites: what the model
actually has to determine. The aggregate above hides three different regimes.

| DoF | n | atoms | control | orb-relax | orb-diffcsp | vanilla | diffcsp-geo (500e) |
|---|---|---|---|---|---|---|---|
| 0 | 125 | 4.3 | 89.6% | 100.0% | 100.0% | 100.0% | **100.0%** |
| 1 | 125 | 7.3 | 55.2% | 94.4% | 100.0% | 100.0% | **100.0%** |
| 2 | 191 | 9.3 | 55.0% | 98.4% | 99.0% | 99.5% | **99.0%** |
| 3 | 56 | 12.0 | 42.9% | 92.9% | 98.2% | 96.4% | **96.4%** |
| 4-5 | 86 | 13.6 | 41.9% | 80.2% | 96.5% | 94.2% | **97.7%** |
| 6-8 | 153 | 14.4 | 28.8% | 69.9% | 89.5% | 91.5% | **94.1%** |
| 9+ | 264 | 16.1 | 4.5% | 26.9% | 49.6% | 57.2% | **61.7%** |

### Macro-partition breakdown (DoF $\ge$ 6 vs DoF < 6)

| Metric | Subgroup | Vanilla DiffCSP++ | DiffCSP-Geo (500e) | Margin / Discordant | Significance |
|---|---|---|---|---|---|
| **Best-of-3** | **DoF $\ge$ 6** ($n=417$) | 69.78% | **73.62%** | **+3.84% (+16 crystals)** | $p = 0.0440$ |
| **Per-trial** | **DoF $\ge$ 6** ($n=1249$) | 59.07% | **63.07%** | **+4.00% (+50 trials)** | $p = 2.39 \times 10^{-4}$ |
| **Best-of-3** | **DoF < 6** ($n=583$) | 98.63% | **98.97%** | **+0.34% (+2 crystals)** | Matched / Beats |
| **Per-trial** | **DoF < 6** ($n=1749$) | 97.26% | **98.06%** | **+0.80% (+14 trials)** | $p = 0.0125$ |
| **Best-of-3** | **Overall** ($n=1000$) | 86.60% | **88.40%** | **+1.80% (+18 crystals)** | $p = 0.0273$ |
| **Per-trial** | **Overall** ($n=2998$) | 81.33% | **83.47%** | **+2.14% (+64 trials)** | $p = 1.08 \times 10^{-5}$ |

- **Below DoF 3 the potential is enough.** At DoF 0 every regime hits 100% and even
  the unrefined draw manages 89.6%: the representation fixes the coordinates and only
  the cell is free. Relaxation stays within 6 points of the generative models to DoF 2.
- **Between DoF 4 and 8 the diffusion earns its keep.** DiffCSP-Geo achieves 97.7% at DoF 4-5 and 94.1% at DoF 6-8, beating DiffCSP++ by 3.5 and 2.6 percentage points respectively.
- **Above DoF 9 capacity and physical Euclidean geometry decide.** DiffCSP-Geo achieves 61.7% (vs 57.2% for DiffCSP++), solving 12 additional high-complexity crystals.
- See full technical details and ablation studies in [`diffcsp-geo-study.md`](diffcsp-geo-study.md).

DoF is the right axis for these four regimes, but not for the asymmetric-unit model added
later: its deficit tracks Wyckoff *orbit multiplicity* instead, and vanishes entirely on
the structures where every orbit has one member. See
[`asymmetric-unit-deficit.md`](asymmetric-unit-deficit.md) and
`bench/multiplicity_analysis.py`.

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
