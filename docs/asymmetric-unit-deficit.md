# Why the asymmetric-unit model loses the high-DoF structures

`docs/wyckoff-innovations-study.md` reports that the asymmetric-unit model (Innovations
1 and 2) matches or beats vanilla DiffCSP++ up to DoF 5 and then falls behind, by 4.8
points per-trial at DoF 6-8 and 5.9 at DoF 9+. It attributes the loss to **multiplicity
ensembling**: the full-cell model emits m score vectors per orbit and averages them,
cutting score variance by ~1/sqrt(m), while the asymmetric model emits one.

That explanation is wrong. This note establishes what the cause actually is.

**Summary.** The deficit is not a function of degrees of freedom but of **orbit
multiplicity**, and it has nothing to do with averaging the score at the output. Vanilla
with its orbit averaging ablated loses nothing. What the full cell actually supplies is
`m` differently-oriented *hidden states* per orbit: the anchor's first message-passing
step is identical in both graphs, and they diverge only from the second layer on, because
a replica's node state is not the same vector as its anchor's under a featurisation that
is not equivariant. The divergence compounds with depth and with orbit size, and lands
hardest on the coordinate head. It shows up in the generated structures as too-short
contacts. The lattice head is unaffected -- it is in fact better than vanilla's.

Two genuine biases *do* exist in the asymmetric-unit code (§5). Neither causes this.

---

## 1. The deficit tracks orbit multiplicity, not DoF

A Wyckoff site of multiplicity `m` is one anchor repeated `m` times, so Innovation 1
replaces `m` graph nodes with one. Where **every** orbit in a structure has multiplicity
1, the asymmetric unit *is* the conventional cell: `K = N`, `inverse_site_map` is the
identity, the bipartite `K x N` edge set is the full `N x N` graph, and all projectors are
symmetric so `pinv(M) == M`. The two models are then the same computation, and that subset
is the control for anything attributed to the asymmetric unit.

MP-20 benchmark set, 1000 representations x 3 draws, per-trial match rate:

| subset | n | vanilla | vanilla, rerun | vanilla, no orbit avg | asymm-inno2 |
|---|---:|---:|---:|---:|---:|
| ALL | 1000 | 81.3% | 81.9% | 82.4% | 79.5% |
| **multiplicity == 1** (architectures identical) | 110 | 47.9% | 49.4% | 49.7% | 47.3% |
| &nbsp;&nbsp;...DoF <= 5 | 32 | 100.0% | 100.0% | 100.0% | 100.0% |
| &nbsp;&nbsp;...DoF 6-8 | 11 | 84.8% | 84.8% | 84.8% | 75.8% |
| &nbsp;&nbsp;...DoF >= 9 | 67 | 16.9% | 19.4% | 19.9% | 17.4% |
| **multiplicity > 1** (orbits collapsed) | 890 | 85.5% | 85.9% | 86.5% | 83.4% |
| &nbsp;&nbsp;...DoF <= 5 | 551 | 97.1% | 97.5% | 97.9% | 97.9% |
| &nbsp;&nbsp;...DoF 6-8 | 142 | 84.3% | 84.3% | 85.7% | 79.8% |
| &nbsp;&nbsp;...DoF >= 9 | 197 | 53.8% | 55.0% | 55.2% | **45.7%** |

`vanilla` is the checkpoint sampled by `run_all.sh`; `vanilla, rerun` is the same weights
resampled in the same session as the ablation, and the 0.6-point spread between them is
the sampler's run-to-run noise -- the paired control for §2.

Paired McNemar, vanilla vs asymm-inno2, per-trial:

| subset | gap | discordant | p |
|---|---:|---|---:|
| ALL | +1.9 | 160 vs 104 | 6.8e-04 |
| multiplicity == 1 | +0.6 | 16 vs 14 | 0.86 |
| multiplicity == 1, DoF >= 9 | -0.5 | 13 vs 14 | 1.0 |
| multiplicity > 1 | +2.0 | 144 vs 90 | 5.0e-04 |
| multiplicity > 1, DoF >= 9 | **+8.1** | 91 vs 43 | **4.1e-05** |

The effect switches off exactly where the architectures coincide. It replicates across
all three asymmetric checkpoints, trained differently and to different lengths:

| checkpoint | gap vs vanilla, mult == 1 | gap vs vanilla, mult > 1 & DoF >= 9 |
|---|---:|---:|
| asymm-inno1, zero-shot transfer | +3.9 | -6.3 |
| asymm-inno1, from scratch ep 210 | +0.6 | -12.5 |
| asymm-inno2, converged ep 510 | -0.6 | -8.1 |

So it is not training budget, capacity, or checkpoint selection. It is the graph.

Two caveats on this control. The multiplicity-1 subset is small (110 structures, 330
trials) and its DoF >= 9 half is harder still (16.9% base rate), so it cannot resolve a
gap below roughly ±7 points there -- the zero-shot row, which is literally vanilla's own
weights in the asymmetric graph and therefore an exact null, comes in at +3.9/+7.0 and
calibrates that floor. What the subset establishes is the *absence* of the 8-12 point
deficit seen at multiplicity > 1, not the absence of any effect at all. And the zero-shot
row at multiplicity > 1 is a model run off-distribution, so it demonstrates that the two
graphs compute different functions, not that one is better; the from-scratch and
converged rows carry that claim.

## 2. It is not multiplicity ensembling

`CSPDiffusion.sample` averages the per-atom scores over each orbit before the update:

```python
pred_x_proj  = torch.einsum("bij, bj -> bi", batch.ops_inv, pred_x)
pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce="mean")[batch.anchor_index]
```

Setting `orbit_average=False` replaces that with `pred_x_proj[batch.anchor_index]` -- the
anchor replica's own score, the single viewpoint the asymmetric model has. Same weights,
same draws, same sampler, nothing else touched -- scored against its own paired control
(same script, same seed, averaging on vs off):

| subset | averaging on | averaging off | gap | discordant | p |
|---|---:|---:|---:|---|---:|
| ALL | 81.9% | 82.4% | -0.5 | 81 vs 96 | 0.29 |
| multiplicity == 1 | 49.4% | 49.7% | -0.3 | 10 vs 11 | 1.0 |
| multiplicity > 1, DoF >= 9 | 55.0% | 55.2% | -0.2 | 51 vs 52 | **1.0** |

Removing the averaging **costs vanilla nothing** -- 51 vs 52 discordant trials in the very
regime that carries the deficit. And single-viewpoint vanilla still beats the asymmetric
model by **+9.5 points** there (100 vs 44 discordant, p = 3.5e-06); the paired control
gives +9.3 (91 vs 36, p = 1.1e-06). Whatever the full cell buys, it is not bought at the
output.

The 1/sqrt(m) reasoning also sits badly with the data: the deficit does not grow with `m`.
Within DoF >= 9 it is +11.2 points where the free orbits have multiplicity 2 and +5.2 where
they have 3-4 -- the wrong direction, though those two bins differ in base rate (32% vs
75%), so read this as failing to support the hypothesis rather than as refuting it on its
own. The ablation above is the decisive test.

## 3. Where the two graphs diverge

`bench/layer_divergence.py` loads one vanilla checkpoint into **both** decoders -- the
Wyckoff layers reuse `CSPLayer`'s parameter names, so the transfer is exact -- and runs
them on identical inputs (512 crystals, 5843 atoms, 2499 sites), comparing the anchor
node after each layer:

| after layer | anchor state | coord score | state, m == 1 | state, m > 1 |
|---:|---:|---:|---:|---:|
| 1 | 6.1e-08 | 1.4e-07 | 6.0e-08 | 6.2e-08 |
| 2 | 8.6e-03 | 1.6e-02 | 4.2e-03 | 1.1e-02 |
| 3 | 5.1e-02 | 9.5e-02 | 1.6e-02 | 7.3e-02 |
| 4 | 9.2e-02 | 1.9e-01 | 2.7e-02 | 1.3e-01 |
| 5 | 1.7e-01 | 3.8e-01 | 4.5e-02 | 2.5e-01 |
| 6 | 2.5e-01 | **5.8e-01** | 7.4e-02 | 3.7e-01 |

(relative L2 difference, mean over sites)

**One hop is exactly equivalent.** The anchor's neighbour set is unchanged -- it still
receives an edge from every one of the `N` cell atoms, so every symmetry-distinct contact
is present, and the anchor sees the same multiset of neighbour displacements any replica
would. At layer 0 all replicas of a site carry the same species embedding. So the layer-1
output agrees to machine precision, and no *geometric* information is lost by the
collapse.

From layer 2 the graphs part company. In the full cell, neighbour `j`'s state was built
from `j`'s own viewpoint (`layers.py:97`, `hj = node_features[edge_index[1]]`); in the
collapsed graph it is its anchor's state (`layers.py:233`,
`hj = site_features[source_sites]`). By symmetry those environments are congruent, but
`CSPNet` embeds edges as `sin/cos` of fractional differences and is not equivariant, so
congruent-but-rotated environments do not produce the same vector. The full cell therefore
hands the anchor `m` genuinely different descriptions of its orbit-mates where the
collapsed graph hands it `m` copies of one. The error compounds with depth, is ~5x larger
when orbits have replicas (0.37 vs 0.07 at layer 6), and is worst in the coordinate score,
the output that matters.

That is the mechanism: the redundancy in the `N`-atom cell is not wasted computation. It
is an implicit ensemble over the symmetry-equivalent views of the same crystal, consumed
*inside* the network rather than at the readout -- which is why ablating the readout
average (§2) does not reproduce it.

## 4. The symptom, and what is not wrong

Shortest interatomic distance in the generated structures, paired on the same draws:

| subset | GT | vanilla | asymm-inno2 | draws with d < 0.9 x GT |
|---|---:|---:|---:|---|
| all | 2.39 Å | 2.36 | 2.35 | vanilla 3.7%, asymm 6.8% |
| DoF <= 5 | 2.55 | 2.53 | 2.55 | vanilla 1.4%, asymm 0.3% |
| DoF >= 9, mult == 1 | 2.15 | 2.01 | 2.01 | vanilla 23.9%, asymm 29.9% |
| DoF >= 9, mult > 1 | 2.07 | 2.01 | 1.97 | vanilla 4.6%, **asymm 18.8%** |

A 4x increase in too-short contacts, confined to the same multiplicity > 1 / high-DoF
cell as everything else.

The **lattice head is not implicated**. It pools over `K` sites rather than `N` atoms, so
it loses vanilla's implicit multiplicity weighting, but that turns out to help: median
relative volume error is 0.61x vanilla's overall, 0.80x at DoF >= 9. The deficit is
entirely in the coordinates.

## 5. Two real biases in the asymmetric-unit code

Neither explains §1-4. Both are worth fixing on their own.

### 5.1 Oblique projectors -- asymmetric-unit specific

`wyckoff_diffusion.py` injects coordinate noise as `site_P @ eps`; `diffusion.py` injects
`ops_inv[anchor] @ eps`, where `ops_inv = pinv(rotation_matrix)`. For a symmetric
projector `pinv(M) == M` and the two agree. But pyXtal's `wp.ops[0].rotation_matrix` is
idempotent and **not always symmetric** -- the Reynolds average `(1/|H|) sum R` is a
symmetric projector only when the `R` are orthogonal, which they are not in the fractional
basis of trigonal and hexagonal groups. For a site like `(x, 2x, z)`:

```
M = [[1,0,0],[2,0,0],[0,0,1]]      M @ M == M  (idempotent)   M != M.T  (oblique)
per-component noise std:  wyckoff  (1.00, 2.00, 1.00)
                          vanilla  (0.45, 0.00, 1.00)
```

The target `d_log_p_wrapped_normal(sigma * M @ eps, sigma)` assumes each component has
std `sigma`; the second component has `2 sigma`. That is a genuinely mis-specified score
target, and it also means the two models are trained on different effective noise
schedules on these sites.

Scope: 240 of 4903 sites in the benchmark set (4.9%), present in **22.0%** of DoF < 9
structures but only **1.1%** of DoF >= 9. It lives in the bins where the asymmetric model
wins, so it cannot be the cause of §1 -- but fixing it should be free upside there. Use
`pinv(M)` for the noise, matching `diffusion.py`, or symmetrise the projector.

### 5.2 Non-uniform sampler prior -- shared with vanilla

Both samplers initialise coordinates as `u = rand(K, 3); x = (M @ u + x0) % 1`. Projecting
a uniform cube is not uniform on the subspace unless the tangent space is axis-aligned.
The free parameter's distribution, decile mass (uniform = 0.100 each):

| site form | std (uniform = 0.289) | min decile | max decile |
|---|---:|---:|---:|
| `(x, y, z)` | 0.289 | 0.099 | 0.101 |
| `(x, 0, 1/2)` | 0.289 | 0.099 | 0.101 |
| `(x, x, z)` | 0.204 | 0.020 | 0.180 |
| `(x, x, x)` | 0.166 | **0.004** | 0.217 |

For `(x, x, x)` the outer deciles are 25x under-sampled -- the sampler essentially never
starts such a site near 0 or 1. The forward process has the matching defect: the
per-component noise std on that site is `sigma/sqrt(3)`, so at `sigma_end = 0.5` the
terminal distribution retains a ~19% first Fourier mode instead of converging to uniform.

Both models do this identically (`diffusion.py` and `wyckoff_diffusion.py` alike), so it
does not move the comparison, but it costs both on trigonal/hexagonal/cubic special
positions. The fix is to draw the free parameters uniformly in the tangent basis rather
than projecting a uniform cube.

## 6. What follows

The diagnosis argues for repairing Innovation 1 rather than abandoning it -- the speedup
is real and the geometric information is provably intact after one hop.

- **The edge is already individuated; the node state is not.** `frac_diff` carries the
  replica's actual position, so the message knows which replica it came from -- but
  `hj = site_features[source_sites]` throws that away in the source state. Conditioning
  the message on the replica's operation, e.g. `hj = MLP([h_{s(j)}, embed(R_j)])` in
  `WyckoffCSPLayer`, restores per-replica distinguishability at unchanged `K x N` cost.
- **Or make the featurisation equivariant**, which is the premise Innovation 1 was written
  under. Innovation 3's Cartesian/Bessel route gives invariant `|r|` and equivariant
  direction, under which congruent environments *do* produce the same state and the
  collapse becomes genuinely lossless.
- Fix §5.1 and §5.2 independently; they are cheap and affect the low-DoF bins.

## Reproducing

```bash
# ablation: vanilla with orbit score-averaging removed
uv run python bench/run_diffusion.py --regime cspnet --no-orbit-average \
  --ckpt data/mp-20/test_ckpt.pt --inits runs/bench/mp20/inits.pkl \
  --out runs/bench/mp20/pred_cspnet_noavg.pkl
uv run python bench/evaluate.py --inits runs/bench/mp20/inits.pkl \
  --preds "vanilla-no-orbit-avg=runs/bench/mp20/pred_cspnet_noavg.pkl" \
  --out runs/bench/mp20/results_noavg.pkl

# the multiplicity split and paired tests
uv run python bench/multiplicity_analysis.py \
  --results runs/bench/mp20/results_inno2_converged.pkl runs/bench/mp20/results_noavg.pkl \
  --regimes "vanilla-diffcsp" "vanilla-no-orbit-avg" "asymm-inno2 (converged)" \
  --compare "vanilla-diffcsp" "asymm-inno2 (converged)"

# layer-by-layer divergence, one checkpoint in both decoders
uv run python bench/layer_divergence.py

# clash and cell-volume diagnostics (§4)
uv run python bench/structure_diagnostics.py \
  --preds "vanilla=runs/bench/mp20/pred_cspnet.pkl" \
          "asymm-inno2=runs/bench/mp20/pred_wyckoff_inno2_converged.pkl"
```
