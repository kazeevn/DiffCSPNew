## Installation
1. [Install poetry](https://python-poetry.org/docs/)
2. Copy `pyproject.toml.zeus` to `pyproject.toml` and change the `torch` wheels as appropriate for your system.
3. Run `poetry install`
## Generate structures for Wyckoff representations
```bash
poetry run python inference.py WyckoffTransformer_mp_20.json.gz
```
The model weights are stored in `test_ckpt.pt`, the preprocessed input data are cached in `cache/`.

To read the structures:
```python
import json
import gzip
from monty.json import MontyDecoder
decoder = MontyDecoder()
with gzip.open('WyckoffTransformer_mp_20.diffcsp-pp.json.gz', 'rt') as f:
    data_raw = json.load(f)
structures = [decoder.process_decoded(d) for d in data_raw]
```
## Train on MP-20
```bash
poetry run python train.py
```
The data are cached in `*.pth` files in the root folder.

## Frozen ORB MLIP + Lightweight Adapter (Blueprint A)

DiffCSP++ with a frozen pretrained ORB MLIP backbone and a lightweight adapter head. Instead of training a 6-layer 512-dim GNN from scratch, this leverages ORB's pre-trained atomic representations, Cartesian forces, and virial stresses:

### Verification / Mock Mode (Lightweight / Local Machine):
```bash
python test_orb_adapter.py
python train_orb.py --mock_orb --epochs 5 --batch_size 16 --device cpu
```

### Full Training on Cluster / GPU with Pretrained ORB:
```bash
pip install orb-models
python train_orb.py --orb_model orb-v2 --batch_size 64 --device cuda --lr 1e-3 --epochs 100
```

### Inference with ORB Adapter:
```bash
python inference_orb.py WyckoffTransformer_mp_20.json.gz --ckpt_path orb_diffcsp_ckpt.pt --device cuda
```
