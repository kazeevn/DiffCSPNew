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
