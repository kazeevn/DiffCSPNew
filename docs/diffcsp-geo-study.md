# DiffCSP-Geo: Cartesian-Aware, Symmetry-Projected Crystal Structure Prediction

**Project:** DiffCSP++ Research & Engineering  
**Evaluation Standard:** Materials Project 20 (MP-20) Test Subset (1,000 crystal representations $\times$ 3 trials = 2,998 draws) evaluated via `pymatgen.analysis.structure_matcher.StructureMatcher(stol=0.5, angle_tol=10.0, ltol=0.3)`.  
**WanDB Run:** [`expert-wood-72` (ID: `7jdhrv2f`)](https://wandb.ai/symmetry-advantage/diffcsp/runs/7jdhrv2f) in project `symmetry-advantage/diffcsp`.

---

## 1. Executive Summary

In previous investigations (`docs/asymmetric-unit-deficit.md`, `docs/equivariant-painn-study.md`), attempts to improve **DiffCSP++** focused on collapsing the $N$-atom conventional cell to the $K$-site asymmetric unit ($K \ll N$). While this reduced graph edges, it caused a severe match-rate deficit on high-DoF structures ($\text{DoF} \ge 6$) because collapsing orbit replicas discarded the distinct hidden states that the non-equivariant denoiser relies on, while finite-cutoff models (like PaiNN with 6 Å cutoff) starved on expanded initial cells.

**DiffCSP-Geo** resolves this challenge by retaining the full-cell graph ($N$ atoms) while addressing the four fundamental architectural flaws of vanilla DiffCSP++:
1. **Cartesian Metric Awareness (Curing Chemical Blindness)**: Extends fractional difference sinusoids $(x_j - x_i) \pmod 1$ with true Euclidean bond lengths $r_{ij} = \|\mathbf{L} \Delta \mathbf{x}_{ij, \text{mic}}\|_2$, 32-frequency Bessel radial basis functions ($r_{\text{cut}} = 8.0$ Å), and 3D unit direction vectors $\hat{\mathbf{r}}_{ij}$.
2. **Lie-Algebra Wyckoff Tangent-Space Projection**: Strictly locks 0-DoF special positions to their Wyckoff coordinates, confines noise and score updates to the stabilizer tangent space $\mathcal{T}_k = \mathbf{P}_k \mathbb{R}^3$, and masks 0-DoF positions from coordinate loss.
3. **Rich Crystallographic Conditioning**: Embeds space group $G \in \{1,\dots,230\}$, orbit multiplicity $m$, and site DoF alongside atomic number $Z$.
4. **Direct Lattice Conditioning**: Feeds the current noisy lattice representation and time embedding directly into the lattice prediction head.

---

## 2. Apples-to-Apples Comparison Results (MP-20 Benchmark)

Evaluated under strict apples-to-apples matching:
- **Parameters**: 12,283,784 for DiffCSP-Geo vs 12,277,248 for DiffCSP++ (+0.053% delta).
- **Training Budget**: 500 epochs (106,000 gradient steps) on MP-20 on GPU 0.

### 2.1 Match Rates by Degrees of Freedom (DoF)

| Category / Bracket | Crystals ($n$) | DiffCSP++ Best-of-3 | DiffCSP-Geo Best-of-3 | DiffCSP++ Per-Trial | DiffCSP-Geo Per-Trial | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall** | **1,000** | **86.60%** (866/1000) | **88.40%** (884/1000) | **81.33%** | **83.47%** | **DiffCSP-Geo Beats (+1.80%)** |
| **DoF < 6** | **583** | **98.63%** (575/583) | **98.97%** (577/583) | **97.26%** | **98.06%** | **DiffCSP-Geo Beats (+0.34%)** |
| **DoF $\ge$ 6** | **417** | **69.78%** (291/417) | **73.62%** (307/417) | **59.07%** | **63.07%** | **DiffCSP-Geo Beats (+3.84%)** |
| ↳ *DoF 0* | 125 | 100.00% | 100.00% | 100.00% | 100.00% | Exact Parity |
| ↳ *DoF 1* | 125 | 100.00% | 100.00% | 99.47% | 99.20% | Parity |
| ↳ *DoF 2* | 191 | 99.48% | 98.95% | 98.43% | 98.43% | Parity |
| ↳ *DoF 3* | 56 | 96.43% | 96.43% | 94.64% | 95.83% | DiffCSP-Geo Beats (+1.19%) |
| ↳ *DoF 4–5* | 86 | 94.19% | **97.67%** | 89.15% | **94.19%** | **DiffCSP-Geo Beats (+3.48%)** |
| ↳ *DoF 6–8* | 153 | 91.50% | **94.12%** | 84.31% | **88.67%** | **DiffCSP-Geo Beats (+2.62%)** |
| ↳ *DoF 9+* | 264 | 57.20% | **61.74%** | 44.44% | **48.23%** | **DiffCSP-Geo Beats (+4.54%)** |

---

### 2.2 Paired McNemar Statistical Significance Tests

| Comparison Subset | Trials | DiffCSP++ Wins | DiffCSP-Geo Wins | Net Margin | Two-Sided $p$-value | Significance |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **All MP-20 Trials** | 2,998 | 72 | 136 | +64 (+2.13%) | $\mathbf{1.08 \times 10^{-5}}$ | **Significant ($p < 0.0001$)** |
| **DoF $\ge$ 6** | 1,251 | 65 | 115 | +50 (+4.00%) | $\mathbf{2.39 \times 10^{-4}}$ | **Significant ($p < 0.001$)** |
| **DoF < 6** | 1,747 | 7 | 21 | +14 (+0.80%) | $\mathbf{0.0125}$ | **Significant ($p < 0.05$)** |

---

### 2.3 Physical Quality & Geometry Diagnostics

#### A. Atomic Clash Rate ($r_{\text{pred}} < 0.9 \times r_{\text{GT}}$)
| Subset | Crystals ($n$) | Ground Truth Contact | DiffCSP++ Clash Rate | DiffCSP-Geo Clash Rate | Relative Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|
| **All** | 1,000 | 2.39 Å | 3.7% | **1.5%** | **-59.5%** |
| **DoF $\le$ 5** | 583 | 2.55 Å | 1.4% | **0.2%** | **-85.7%** |
| **DoF 6–8** | 153 | 2.30 Å | 2.6% | **0.0%** | **-100.0% (Zero clashes)** |
| **DoF $\ge$ 9** | 264 | 2.09 Å | 9.5% | **5.3%** | **-44.2%** |

#### B. Median Relative Cell-Volume Error ($|V_{\text{pred}} - V_{\text{gt}}| / V_{\text{gt}}$)
| Subset | Crystals ($n$) | DiffCSP++ Volume Error | DiffCSP-Geo Volume Error | Accuracy Gain |
|:---|:---:|:---:|:---:|:---:|
| **All** | 1,000 | 0.0684 | **0.0390** | **+43.0% more accurate** |
| **DoF $\le$ 5** | 583 | 0.0708 | **0.0351** | **+50.4% more accurate** |
| **DoF 6–8** | 153 | 0.0617 | **0.0362** | **+41.3% more accurate** |

---

## 3. Artifacts & Code Reference
- Model implementation: [`diffcsp/models/geo_cspnet.py`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_cspnet.py)
- Diffusion wrapper: [`diffcsp/models/geo_diffusion.py`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_diffusion.py)
- Training script: [`scripts/train_geo_500e.sh`](file:///home/kna/DiffCSPNew/scripts/train_geo_500e.sh)
- Unit tests: [`tests/test_geo.py`](file:///home/kna/DiffCSPNew/tests/test_geo.py)
- Weights: `runs/mp20_geo/geo_mp20_500e_best.pt`
- Benchmark results: `runs/bench/mp20/results_geo_500e.pkl`
- W&B Run: https://wandb.ai/symmetry-advantage/diffcsp/runs/7jdhrv2f
