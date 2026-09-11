# Architectural Innovations for Wyckoff-Conditioned DiffCSP++

This document outlines architectural and mathematical innovations to improve **DiffCSP++** when operating as the continuous structure solver conditioned on the **Wyckoff gene** produced by **WyFormer** (or equivalent discrete symmetry generators).

---

## 1. Problem Formulation & Scope

In the WyFormer $\to$ DiffCSP++ pipeline, the generation of crystal structures is factorized into two distinct stages:

```
Chemical Composition / Target
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│ WyFormer (Discrete Crystallographic Generator)          │
│ - Space Group G ∈ {1, ..., 230}                         │
│ - Occupied Wyckoff sites {(Z_i, letter_i, mult_i)}      │
│ - Atom counts & Stoichiometry                           │
└─────────────────────────┬───────────────────────────────┘
                          │ Wyckoff Gene (Discrete Scaffold)
                          ▼
┌─────────────────────────────────────────────────────────┐
│ DiffCSP++ (Continuous Geometric Denoiser / Flow)        │
│ - Free Wyckoff fractional coordinates (Wyckoff DoF)     │
│ - Unit cell lattice matrix L ∈ R^(3x3) (Lattice DoF)    │
└─────────────────────────┬───────────────────────────────┘
                          │ Fully Realized Crystal
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Downstream Relaxation / Verification (ORB MLIP / DFT)   │
└─────────────────────────────────────────────────────────┘
```

Because the discrete crystallographic identity (space group, site assignments, atomic species, and orbit multiplicity) is **fixed upstream by WyFormer**, DiffCSP++ does **not** need to perform unconstrained *ab initio* generation or site-assignment search. Its sole objective is to solve for:
1. The **continuous fractional coordinates** of each unique Wyckoff site within its symmetry-permitted subspace.
2. The **continuous lattice parameters** ($a, b, c, \alpha, \beta, \gamma$) within the constraints of the crystal system.

---

## 2. Bottlenecks in the Current Architecture

Analysis of the current codebase ([`diffcsp/models/cspnet.py`](file:///home/kna/DiffCSPNew/diffcsp/models/cspnet.py), [`diffcsp/models/diffusion.py`](file:///home/kna/DiffCSPNew/diffcsp/models/diffusion.py)) and benchmark results ([`docs/benchmark.md`](file:///home/kna/DiffCSPNew/docs/benchmark.md), [`docs/training-stability.md`](file:///home/kna/DiffCSPNew/docs/training-stability.md)) highlights six major structural limitations:

1. **The $N$-Atom Full-Cell Graph Bottleneck:**
   DiffCSP++ expands the Wyckoff gene into all $N$ atoms in the conventional unit cell (e.g. 64–128+ atoms) via PyXtal and builds a fully connected intra-crystal graph `fc_graph = torch.block_diag(...)`. As documented in [`docs/training-stability.md`](file:///home/kna/DiffCSPNew/docs/training-stability.md), **14.8% of structures account for 73.7% of all edges**, causing severe gradient spikes, memory blowups, and training instability.
2. **Coordinate Over-Parameterization on Special Positions:**
   In [`diffcsp/models/diffusion.py`](file:///home/kna/DiffCSPNew/diffcsp/models/diffusion.py), 3D Gaussian noise is injected into all coordinates, and the network predicts 3D vector updates that are projected post-hoc via `batch.ops_inv`. For 0-DoF sites (e.g. $(0, 0, 0)$) or 1-DoF lines (e.g. $(x, 0, 1/2)$), the model wastes expressive capacity learning to cancel noise in symmetry-forbidden directions.
3. **Cartesian "Chemical Blindness":**
   [`CSPNet`](file:///home/kna/DiffCSPNew/diffcsp/models/cspnet.py) operates purely on fractional difference sinusoids $(x_j - x_i) \pmod 1$ and a global 6D lattice representation. It **never computes real Euclidean bond lengths** $r_{ij} = \|\mathbf{L}(\mathbf{x}_j - \mathbf{x}_i + \mathbf{n})\|_2$ in Ångströms. Consequently, the network cannot directly evaluate whether atoms are chemically bonded (~1.5 Å) or catastrophically overlapping (< 0.8 Å). This is the primary driver of the **performance collapse on DoF 9+ structures** (dropping to 49.6%–57.2% match rate in [`docs/benchmark.md`](file:///home/kna/DiffCSPNew/docs/benchmark.md)).
4. **Information Bottleneck in Lattice Prediction:**
   Lattice updates are computed via global mean pooling over atom features (`scatter(..., reduce="mean")` in [`diffcsp/models/cspnet.py`](file:///home/kna/DiffCSPNew/diffcsp/models/cspnet.py#L124-L126)), which severs the fine-grained coupling between cell strain and individual Wyckoff site coordinates.
5. **Slow 1,000-Step SDE Sampling:**
   Sampling requires 1,000 predictor-corrector diffusion steps. For a standard 1,000-structure $\times$ 3-trial benchmark, this incurs **3,000,000 GNN forward evaluations**, making inference and large-scale screening slow.
6. **MLIP Capacity & Basin Disconnect:**
   The frozen ORB adapter has only 561k parameters vs. 12.3M in vanilla CSPNet, choking global configuration search at high DoF. Conversely, ORB relaxation produces significantly better local RMS accuracy (0.0371 Å vs. 0.0404 Å) once inside the true energy basin.

---

## 3. Six Architectural Innovations

### Innovation 1: Asymmetric Unit (Wyckoff-Only) Message Passing

*(References: **SymmCD** [NeurIPS 2024], **SGEquiDiff** [Princeton/UIUC 2024])*

#### Concept
Rather than instantiating all $N$ atoms in the conventional cell, the graph nodes in DiffCSP++ should represent **only the $K$ unique Wyckoff sites (the asymmetric unit)**, where typically $K \in [1, 8]$:

$$\mathcal{V}_{\text{model}} = \{ W_1, W_2, \dots, W_K \}$$

Interactions with symmetry replicas across the unit cell and periodic boundaries are handled via **Wyckoff-to-Wyckoff multi-edges**:

$$\mathbf{m}_{i} = \sum_{j=1}^K \sum_{(R, \mathbf{t}, \mathbf{n}) \in \mathcal{N}_{ij}} \text{Message}\left(\mathbf{h}_i, \mathbf{h}_j, \mathbf{r}_{ij}^{(R, \mathbf{t}, \mathbf{n})}\right)$$

where $(R, \mathbf{t})$ are the space group operations mapping anchor $j$ to its orbit images, and $\mathbf{n} \in \mathbb{Z}^3$ are periodic unit cell translation shifts within a distance cutoff $r_{\text{cut}}$.

#### Benefits
- **Eliminates Edge Explosion:** For a cell with 128 atoms derived from 4 Wyckoff sites, graph size drops from 16,384 edges to $< 50$ edges.
- **Resolves Training Instability:** Uniform edge distribution across batches eliminates the heavy-tailed gradient spikes documented in [`docs/training-stability.md`](file:///home/kna/DiffCSPNew/docs/training-stability.md).
- **10x–30x Speedup & Memory Reduction:** The forward pass evaluates message passing on only $K \ll N$ nodes.

---

### Innovation 2: Tangent-Space Subspace Coordinate Parameterization

*(Reference: **SGEquiDiff** [2024/2025])*

#### Concept
Each Wyckoff site $W_k$ has an associated site-symmetry stabilizer group $H_k \le G$. The coordinates of site $W_k$ are confined to an affine subspace $\mathcal{A}_k = \mathbf{x}_0^{(k)} + \mathcal{T}_k$, where $\mathcal{T}_k \subseteq \mathbb{R}^3$ is the linear tangent space:
- **0 DoF (Special Position):** $\dim(\mathcal{T}_k) = 0$ (e.g. $(0, 0, 0)$ or $(1/2, 1/2, 1/2)$). $\mathbf{x}^{(k)}$ is constant.
- **1 DoF (Line):** $\dim(\mathcal{T}_k) = 1$ (e.g. $(x, 0, 1/2)$ or $(x, x, x)$). Parameterized by scalar $u \in \mathbb{T}^1$.
- **2 DoF (Plane):** $\dim(\mathcal{T}_k) = 2$ (e.g. $(x, y, 0)$). Parameterized by $(u, v) \in \mathbb{T}^2$.
- **3 DoF (General Position):** $\dim(\mathcal{T}_k) = 3$. Parameterized by $(u, v, w) \in \mathbb{T}^3$.

#### Implementation
1. Project noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$ directly onto $\mathcal{T}_k$:
   $$\boldsymbol{\epsilon}_{\mathcal{T}_k} = \mathbf{P}_k \boldsymbol{\epsilon}$$
   where $\mathbf{P}_k = \frac{1}{|H_k|} \sum_{R \in H_k} R$ is the projection matrix of the stabilizer group.
2. For 0-DoF sites, fix $\mathbf{x}^{(k)} = \mathbf{x}_0^{(k)}$; do not inject noise and do not predict scores.
3. The coordinate prediction head outputs vectors strictly in $\mathcal{T}_k$:
   $$\hat{\mathbf{v}}_k = \mathbf{P}_k \text{Head}(\mathbf{h}_k)$$

#### Benefits
- Symmetry is strictly guaranteed **by construction** at every continuous step, rather than enforced as a post-hoc correction.
- Prevents the network from spending capacity suppressing out-of-subspace noise.

---

### Innovation 3: Cartesian Multi-Graph Awareness & Bessel RBFs

*(References: **MatterGen** [Microsoft Research], **GemNet-dT**, **EquiformerV2**)*

#### Concept
The primary cause of failure on structures with Wyckoff DoF $\ge 9$ in [`docs/benchmark.md`](file:///home/kna/DiffCSPNew/docs/benchmark.md) is that the network cannot measure physical distance in Ångströms.

Replace the fractional difference sinusoid embedding with explicit Cartesian geometry:
1. Reconstruct the real Euclidean displacement vector for each edge between site $i$ and symmetry image $(R, \mathbf{t}, \mathbf{n})$ of site $j$:
   $$\mathbf{r}_{ij}^{(R, \mathbf{t}, \mathbf{n})} = \mathbf{L} \left( R \mathbf{x}_j + \mathbf{t} + \mathbf{n} - \mathbf{x}_i \right)$$
2. Compute the physical distance $d = \|\mathbf{r}_{ij}^{(R, \mathbf{t}, \mathbf{n})}\|_2$.
3. Expand $d$ using **Bessel Radial Basis Functions (RBF)** with a smooth polynomial cutoff $f_{\text{cut}}(d; r_{\text{cut}})$ at $r_{\text{cut}} \approx 6.0\text{ \AA}$:
   $$e_n(d) = \sqrt{\frac{2}{r_{\text{cut}}}} \frac{\sin\left(\frac{n \pi d}{r_{\text{cut}}}\right)}{d} \cdot f_{\text{cut}}(d)$$
4. Incorporate the unit direction vector $\hat{\mathbf{r}} = \mathbf{r} / d$ into equivariant message passing.
5. In training and inference, apply the analytic repulsive potential from [`diffcsp/models/repulsion.py`](file:///home/kna/DiffCSPNew/diffcsp/models/repulsion.py) when $d < \eta (R_i + R_j)$ to penalize steric overlaps.

#### Benefits
- Directly resolves the DoF 9+ collapse by equipping the model with physical bond-length awareness (e.g. C–C ~ 1.54 Å, Ti–O ~ 1.95 Å).
- Drastically reduces unphysical steric clashes in generated candidates.

---

### Innovation 4: Rich Wyckoff Conditioning & WyFormer Latent Cross-Attention

*(References: **WyFormer** [ICML 2025], **WyckoffDiff-Adaptor** [2025])*

#### Concept
In the existing codebase, [`CSPNet`](file:///home/kna/DiffCSPNew/diffcsp/models/cspnet.py#L50) initializes atom nodes using only their scalar atomic number $Z$:
```python
self.node_embedding = nn.Embedding(max_atoms + 1, hidden_dim)
```
This throws away critical crystallographic metadata provided by WyFormer.

#### Implementation
Enrich the site representation before message passing:
1. **Wyckoff Discrete Embeddings:**
   $$\mathbf{h}_k^{(0)} = \text{Embed}_Z(Z_k) + \text{Embed}_{\text{letter}}(\text{letter}_k) + \text{Embed}_{\text{mult}}(\text{mult}_k) + \text{Embed}_{\text{site\_sym}}(\text{PointGroup}_k)$$
2. **WyFormer Cross-Attention:**
   Rather than treating WyFormer as an external black box that outputs text tokens, expose WyFormer's penultimate transformer hidden representations $\mathbf{H}_{\text{WyFormer}} \in \mathbb{R}^{K \times d_{\text{trans}}}$. In DiffCSP++, insert cross-attention layers:
   $$\mathbf{h}_k \leftarrow \mathbf{h}_k + \text{MultiHeadAttention}(\text{query}=\mathbf{h}_k, \text{key}=\mathbf{H}_{\text{WyFormer}}, \text{value}=\mathbf{H}_{\text{WyFormer}})$$

#### Benefits
- Gives the continuous denoiser direct access to the global coordination and chemical context already learned by the discrete transformer.

---

### Innovation 5: Riemannian Flow Matching (RFM) on the Wyckoff Subspace

*(References: **FlowMM** [ICML 2024], **CrystalFlow** [2025], **uFlowCSP** [2026])*

#### Concept
DiffCSP++'s 1,000-step predictor-corrector diffusion can be replaced with **Riemannian Flow Matching (RFM)**. RFM defines probability paths along geodesics with deterministic vector fields:

1. **Lattice Flow:**
   Define the lattice path on the Lie algebra $\mathfrak{gl}(3, \mathbb{R})$ or the metric manifold of symmetric matrices $\mathcal{S}_{++}^3$, constrained by space group projection $\mathbf{P}_G^{\text{lattice}}$:
   $$\mathbf{L}_t = \exp\left( (1 - t) \mathbf{v}_0 + t \mathbf{v}_1 \right), \quad \mathbf{v} \in \mathfrak{g}_G$$
2. **Coordinate Flow on Torus Subspace:**
   For site $k$ with free parameters $\mathbf{u}_k \in \mathbb{T}^{\dim(\mathcal{T}_k)}$, define the conditional vector field along the shortest toroidal geodesic:
   $$\mathbf{u}_t^{(k)} = \left( (1 - t) \mathbf{u}_0^{(k)} + t \mathbf{u}_1^{(k)} \right) \pmod 1$$
   The target velocity is simply the constant displacement vector $\mathbf{v}^* = (\mathbf{u}_1^{(k)} - \mathbf{u}_0^{(k)})_{\mathbb{T}}$.

#### Benefits
- **20x–50x Inference Speedup:** High-quality generation requires only **20 to 50 ODE solver steps** (using Midpoint, RK4, or adaptive Dormand-Prince) instead of 1,000 SDE steps.
- **Stable Training Objective:** Replaces wrapped-normal score matching (which diverges at $t \to 0$) with a clean mean-squared error regression on the target vector field $\mathbf{v}_t$.

---

### Innovation 6: Two-Phase Hybrid Sampling (Basin Search $\to$ MLIP Relaxation)

*(Insight from [`docs/benchmark.md`](file:///home/kna/DiffCSPNew/docs/benchmark.md))*

#### Benchmark Finding
In the MP-20 and LeMat benchmarks:
- **Generative Diffusion** excelled at finding the correct structural basin (+22.7% margin over relaxation on high-DoF structures), but settled with a slightly higher mean RMS error (~0.0404 Å).
- **ORB Relaxation** had the lowest mean RMS error (**0.0371 Å**), but could not navigate barrier crossings to locate the correct basin from random initializations.

#### Hybrid Protocol
Combine the complementary strengths of both methods:

```
t = 1.0 (Random Noise)
   │
   │  Phase 1: Flow / Generative Search (Global Basin Discovery)
   │  - 25–40 steps of Flow Matching or DiffCSP++
   │  - Navigates configuration space & avoids false minima
   ▼
t = 0.1 (Near-Equilibrium Scaffold)
   │
   │  Phase 2: Hand-off to Symmetry-Constrained ORB Relaxation
   │  - Minimizes exact DFT-surrogate potential within space group
   ▼
t = 0.0 (High-Precision Ground Truth Match, sub-0.038 Å RMS)
```

#### Implementation
In `diffcsp/cli/inference.py` and `bench/run_diffusion.py`, implement an `--orb-handoff-t <float>` flag (e.g. `0.10`). When sampling reaches $t \le 0.10$, the intermediate fractional coordinates and lattice are passed directly to `run_relax.py`'s symmetry-constrained L-BFGS / FIRE optimizer.

---

## 4. Implementation Roadmap & Prioritization

The following matrix organizes the innovations by implementation effort and expected performance impact:

| Phase | Innovation | Primary Files Touched | Effort | Expected Gain |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Two-Phase Sampling Hand-off**<br>Early exit from diffusion at $t=0.10$ into symmetry-constrained ORB relaxation | [`bench/run_diffusion.py`](file:///home/kna/DiffCSPNew/bench/run_diffusion.py), [`diffcsp/cli/inference.py`](file:///home/kna/DiffCSPNew/diffcsp/cli/inference.py) | Low (1–2 days) | Combines 85%+ best-of-3 match rate with sub-0.038 Å RMS accuracy. |
| **2** | **Cartesian Multi-Graph & Bessel RBFs**<br>Compute real $\mathbf{r}_{ij} = \mathbf{L}(\Delta \mathbf{x} + \mathbf{n})$, Bessel RBF distance embeddings, and steric repulsion | [`diffcsp/models/layers.py`](file:///home/kna/DiffCSPNew/diffcsp/models/layers.py), [`diffcsp/models/cspnet.py`](file:///home/kna/DiffCSPNew/diffcsp/models/cspnet.py) | Medium (3–5 days) | Resolves the DoF 9+ match rate collapse (targets +10–15% in high-DoF bins). |
| **3** | **Asymmetric Unit Message Passing & Tangent Subspaces**<br>Denoise $K$ Wyckoff sites instead of $N$ atoms; project vector fields onto site stabilizer subspaces | [`diffcsp/models/layers.py`](file:///home/kna/DiffCSPNew/diffcsp/models/layers.py), [`diffcsp/models/diffusion.py`](file:///home/kna/DiffCSPNew/diffcsp/models/diffusion.py) | Medium (1 week) | Eliminates $O(N^2)$ edge explosion, fixes training instability, 10x faster forward pass. |
| **4** | **Riemannian Flow Matching**<br>Formulate continuous vector fields on $\mathbb{T}^{\text{DoF}} \times \mathfrak{g}_G$ with ODE integration | [`diffcsp/core/schedulers.py`](file:///home/kna/DiffCSPNew/diffcsp/core/schedulers.py), [`diffcsp/models/diffusion.py`](file:///home/kna/DiffCSPNew/diffcsp/models/diffusion.py) | Medium-High (1–2 weeks) | 20x–50x speedup in sampling (30 steps vs. 1,000 steps); smoother training curves. |
| **5** | **WyFormer Cross-Attention & Rich Wyckoff Tokens**<br>Embed Wyckoff letter, multiplicity, and cross-attend into WyFormer latent states | [`diffcsp/models/cspnet.py`](file:///home/kna/DiffCSPNew/diffcsp/models/cspnet.py), [`diffcsp/data/dataset.py`](file:///home/kna/DiffCSPNew/diffcsp/data/dataset.py) | Medium (3–5 days) | Tighter conditioning between discrete Wyckoff scaffold and continuous coordinates. |
