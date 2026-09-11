# DiffCSP++

**DiffCSP++: Crystal Structure Prediction via Space Group Constrained Diffusion with MLIP Backbone**

DiffCSP++ generates periodic crystal structures by diffusing atomic fractional coordinates on the 3-torus and lattice vectors on the Lie algebra $\mathfrak{gl}(3, \mathbb{R})$, enforcing exact space group symmetries by design. It supports both the original deep GNN denoiser (CSPNet) and a frozen pretrained Machine Learning Interatomic Potential backbone (ORB MLIP) with a lightweight adapter network.

**Results and notes:** [`docs/benchmark.md`](docs/benchmark.md) for structure-prediction
benchmarks, [`docs/training-stability.md`](docs/training-stability.md) for known
training-stability issues and proposed fixes.

---

## 1. Architecture & Package Structure

The codebase is organized as a modular Python package (`diffcsp`) with backward-compatible root script facades:

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
bench/                  # Structure-prediction benchmark harness (see docs/benchmark.md)
docs/                   # Benchmark results and training notes
tests/                  # pytest suite (27 unit & integration tests)
```

---

## 2. Environment & Container Execution

On machines with legacy Kepler (`sm_35`) and Maxwell (`sm_50`) GPUs (Tesla K20c / GTX 750 Ti), code is executed within the custom-compiled PyTorch 2.14 Docker container from `/home/kna/pytorch-research` (`pytorch:2.14.0-cuda11.8-py312-universal`).

A convenience script [`run_container.sh`](run_container.sh) is provided in the repository root:

```bash
# Run tests
./run_container.sh pytest

# Run linting and formatting
./run_container.sh ruff check diffcsp/ tests/
./run_container.sh ruff format --check diffcsp/ tests/

# Run interactive bash
./run_container.sh bash
```

Alternatively, run directly with Docker:
```bash
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  pytorch:2.14.0-cuda11.8-py312-universal \
  <command>
```

---

## 3. Testing & Quality Assurance

The codebase includes an extensive suite of automated tests verifying numerical operations, symmetry constraints, model forward passes, diffusion sampling, and CLI interfaces:

```bash
./run_container.sh pytest
```

To run individual test modules:
```bash
./run_container.sh pytest tests/test_matrix.py
./run_container.sh pytest tests/test_crystal_family.py
./run_container.sh pytest tests/test_models.py
./run_container.sh pytest tests/test_diffusion.py
```

To run specific tests with verbose output:
```bash
./run_container.sh pytest -v -k "orb or diffusion"
```

---

## 4. Structure Generation (Inference)

Generate periodic crystal structures from Wyckoff representations (`.json` or `.json.gz`):

### With Frozen ORB MLIP Adapter:
```bash
./run_container.sh python inference.py WyckoffTransformer_mp_20.json.gz \
  --model orb \
  --ckpt_path orb_diffcsp_ckpt.pt \
  --device cuda
```

### With Standard DiffCSP++ (CSPNet):
```bash
./run_container.sh python inference.py WyckoffTransformer_mp_20.json.gz \
  --model cspnet \
  --ckpt_path test_ckpt.pt \
  --device cuda
```

The output structures are saved in gzip-compressed JSON (`*.diffcsp-orb.json.gz` or `*.diffcsp-cspnet.json.gz`).

### Reading Output Structures:
```python
import gzip
import json
from monty.json import MontyDecoder

decoder = MontyDecoder()
with gzip.open("WyckoffTransformer_mp_20.diffcsp-orb.json.gz", "rt") as f:
    data_raw = json.load(f)
structures = [decoder.process_decoded(d) for d in data_raw]
print(f"Loaded {len(structures)} generated pymatgen structures.")
```

---

## 5. Training

### Train Frozen ORB MLIP Adapter:
```bash
./run_container.sh python train.py \
  --model orb \
  --orb_model orb-v3 \
  --train_csv train.csv \
  --test_csv test.csv \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-3 \
  --device cuda
```

For fast local verification or testing without GPU weights, pass `--mock_orb`:
```bash
./run_container.sh python train.py --model orb --mock_orb --epochs 5 --batch_size 16 --device cpu
```

### Train Standard DiffCSP++ (CSPNet):
```bash
./run_container.sh python train.py \
  --model cspnet \
  --train_csv train.csv \
  --test_csv test.csv \
  --batch_size 256 \
  --epochs 500 \
  --lr 1e-3 \
  --device cuda
```

### Dataset options

Large CSV tables are preprocessed once into shards under `--cache_dir` and reused; an
interrupted preprocessing run resumes from the shards already written rather than
restarting. A complete shard set loads without re-reading the source CSV.

```bash
python train.py --model cspnet \
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

The CLI tools can be invoked through multiple equivalent methods:

1. **Directly via Python scripts in root:**
   ```bash
   ./run_container.sh python train.py --model orb --epochs 100
   ./run_container.sh python inference.py WyckoffTransformer_mp_20.json.gz --model orb
   ```

2. **Via standard Python module execution:**
   ```bash
   ./run_container.sh python -m diffcsp.cli.train --model orb --epochs 100
   ./run_container.sh python -m diffcsp.cli.inference WyckoffTransformer_mp_20.json.gz --model orb
   ```

3. **Via installed package entrypoints:**
   ```bash
   ./run_container.sh diffcsp-train --model orb --epochs 100
   ./run_container.sh diffcsp-inference WyckoffTransformer_mp_20.json.gz --model orb
   ```

