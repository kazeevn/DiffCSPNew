# DiffCSP++

**DiffCSP++: Crystal Structure Prediction via Space Group Constrained Diffusion with MLIP Backbone**

DiffCSP++ generates periodic crystal structures by diffusing atomic fractional coordinates on the 3-torus and lattice vectors on the Lie algebra $\mathfrak{gl}(3, \mathbb{R})$, enforcing exact space group symmetries by design. It supports both the original deep GNN denoiser (CSPNet) and a frozen pretrained Machine Learning Interatomic Potential backbone (ORB MLIP) with a lightweight adapter network.

**Results and notes:** [`docs/benchmark.md`](docs/benchmark.md) for structure-prediction
benchmarks, [`docs/training-stability.md`](docs/training-stability.md) for known
training-stability issues and proposed fixes.

---

## 1. Architecture & Package Structure

The codebase is organized as a modular Python package (`diffcsp`) with clean entrypoints:

```
diffcsp/
├── core/               # Mathematical foundations & diffusion math
│   ├── crystal_family.py  # 6D Lie algebra basis & 230 space group linear constraints
│   ├── matrix.py          # Numerically stabilized logm, expm, sqrtm on SO(3)
│   ├── scatter.py         # Scatter & segment operations with zero C++ wheel dependency
│   └── schedulers.py      # BetaScheduler, SigmaScheduler, wrapped normal distributions
├── data/               # Graph extraction, caching, and datasets
│   ├── dataset.py         # CrystDataset (MP-20 CIF tables) & WyckoffDataset (pyXtal JSON/GZ)
│   ├── graph.py           # CrystalNN / pymatgen graph extraction & PBC neighbor lists
│   └── transforms.py      # Coordinate (frac/cart), force, and lattice matrix transforms
├── models/             # Neural network architectures
│   ├── cspnet.py          # Original 6-layer GNN denoiser
│   ├── cspnet_orb.py      # Lightweight adapter on frozen ORB MLIP (forces, stress & representations as conditioning)
│   ├── diffusion.py       # Base CSPDiffusion with predictor-corrector sampler
│   ├── diffusion_orb.py   # CSPDiffusionORB subclassing base diffusion with adapter head
│   ├── layers.py          # CSPLayer message passing and SinusoidsEmbedding
│   └── orb_wrapper.py     # Frozen ORB MLIP potential wrapper with MockOrbBackbone fallback
└── cli/                # Structured command-line interfaces
    ├── train.py           # Unified training CLI (both CSPNet and ORB adapter)
    └── inference.py       # Unified structure generation CLI
data/                   # Dataset repository (data/mp-20/, data/carbon-24/, etc.)
scripts/                # Training and utility scripts
├── platforms/          # Platform-specific scripts (e.g. iapetus, zeus)
bench/                  # Structure-prediction benchmark harness (outputs to runs/bench/)
runs/                   # Experiment artifacts, checkpoints, and logs (gitignored)
docs/                   # Benchmark results and training notes
tests/                  # pytest suite (27 unit & integration tests)
```

---

## 2. Environment & Execution

We use **`uv`** for dependency and environment management.

```bash
# Sync environment with all dependencies (including dev, orb, wandb)
uv sync --all-extras

# Run test suite
uv run pytest
```

### Platform-specific Setup

Platform configurations and convenience scripts are housed under `scripts/platforms/<platform>`:

- **Zeus**: Host-specific `uv.toml` and initialization script:
  ```bash
  scripts/platforms/zeus/env_init.sh
  ```
- **Iapetus**: PyTorch 2.14 universal container execution:
  ```bash
  scripts/platforms/iapetus/run_container.sh uv run pytest
  ```

---

## 3. Testing & Quality Assurance

The codebase includes an extensive suite of automated tests verifying numerical operations, symmetry constraints, model forward passes, diffusion sampling, and CLI interfaces:

```bash
uv run pytest
```

To run individual test modules:
```bash
uv run pytest tests/test_matrix.py
uv run pytest tests/test_crystal_family.py
uv run pytest tests/test_models.py
uv run pytest tests/test_diffusion.py
```

To run specific tests with verbose output:
```bash
uv run pytest -v -k "orb or diffusion"
```

---

## 4. Structure Generation (Inference)

Generate periodic crystal structures from Wyckoff representations (`.json` or `.json.gz`):

### With Frozen ORB MLIP Adapter:
```bash
uv run diffcsp-inference data/mp-20/WyckoffTransformer_mp_20.json.gz \
  --model orb \
  --ckpt_path runs/vuvt0kab/orb_v3_diffcsp_mp20_opt3.pt \
  --device cuda
```

### With Standard DiffCSP++ (CSPNet):
```bash
uv run diffcsp-inference data/mp-20/WyckoffTransformer_mp_20.json.gz \
  --model cspnet \
  --ckpt_path data/mp-20/test_ckpt.pt \
  --device cuda
```

The output structures are saved in gzip-compressed JSON (`*.diffcsp-orb.json.gz` or `*.diffcsp-cspnet.json.gz`).

### Reading Output Structures:
```python
import gzip
import json
from monty.json import MontyDecoder

decoder = MontyDecoder()
with gzip.open("data/mp-20/WyckoffTransformer_mp_20.diffcsp-orb.json.gz", "rt") as f:
    data_raw = json.load(f)
structures = [decoder.process_decoded(d) for d in data_raw]
print(f"Loaded {len(structures)} generated pymatgen structures.")
```

---

## 5. Training

### Train Frozen ORB MLIP Adapter:
```bash
uv run diffcsp-train \
  --model orb \
  --orb_model orb-v3 \
  --train_csv data/mp-20/train.csv \
  --test_csv data/mp-20/test.csv \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-3 \
  --device cuda
```

For fast local verification or testing without GPU weights, pass `--mock_orb`:
```bash
uv run diffcsp-train --model orb --mock_orb --epochs 5 --batch_size 16 --device cpu
```

### Train Standard DiffCSP++ (CSPNet):
```bash
uv run diffcsp-train \
  --model cspnet \
  --train_csv data/mp-20/train.csv \
  --test_csv data/mp-20/test.csv \
  --batch_size 256 \
  --epochs 500 \
  --lr 1e-3 \
  --device cuda
```

Or run the predefined scripts under `scripts/`:
```bash
scripts/run_full_training.sh
scripts/run_lemat_cspnet.sh
```

### Dataset options

Large CSV tables are preprocessed once into shards under `--cache_dir` and reused; an
interrupted preprocessing run resumes from the shards already written rather than
restarting. A complete shard set loads without re-reading the source CSV.

```bash
uv run diffcsp-train --model cspnet \
  --train_csv data/train.csv.gz --test_csv data/val.csv.gz \
  --max_e_hull 0.1 \        # keep structures within 0.1 eV of the hull
  --max_atoms 128 \         # drop larger conventional cells (see below)
  --cache_dir cache/my_dataset
```

`--max_atoms` matters for any dataset with large cells. CSPNet connects every atom in
a cell to every other, so cost and memory grow as the square of the cell size: a
992-atom conventional cell is ~984k edges on its own, more than a whole batch of
typical ones, and fits at no batch size. The cap is applied to the loaded cache, so
one cache serves any cap.

`--hidden_dim` and `--num_layers` default per model — 128/2 for `--model orb` (a small
head on a frozen 25.6M potential), 512/6 for `--model cspnet` (the whole denoiser).
The effective values are printed at startup and logged to W&B.

### ORB backbone options

`--orb_model` accepts an alias (`orb-v3`, `orb-v3-direct-omat`, `orb-v2`, ...) or any
`orb_models.pretrained` loader name. Unknown names raise rather than silently
substituting a different backbone.

The adapter conditions on the potential's forces, stress **and** its learned atomic
representations. Three flags change that:

| flag | effect |
|---|---|
| `--no_orb_node_features` | condition on forces and stress only, without the representations |
| `--enforce_zero_force` | constrain the coordinate score to `gamma*f_frac + v_perp`, `gamma > 0` |
| `--force_residual` | add a time-gated force residual instead of the hard constraint |

Both constraint forms are off by default: they can express only one direction along
the force, and training drives `gamma` to its floor. See `docs/benchmark.md`.

---

## 6. CLI Execution

The CLI tools can be invoked via:

1. **Installed package entrypoints:**
   ```bash
   uv run diffcsp-train --model orb --epochs 100
   uv run diffcsp-inference data/mp-20/WyckoffTransformer_mp_20.json.gz --model orb
   ```

2. **Standard Python module execution:**
   ```bash
   uv run python -m diffcsp.cli.train --model orb --epochs 100
   uv run python -m diffcsp.cli.inference data/mp-20/WyckoffTransformer_mp_20.json.gz --model orb
   ```

