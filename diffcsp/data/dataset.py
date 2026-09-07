"""PyTorch Geometric datasets for crystal structures and Wyckoff representations.

Provides CrystDataset for CIF/MP-20 datasets and WyckoffDataset for
pyXtal Wyckoff representation JSON/GZ archives.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import trange

from diffcsp.data.graph import process_one

logger = logging.getLogger(__name__)


def graph_arrays_to_pyg_data(data_dict: dict[str, Any]) -> Data:
    """Converts cached graph arrays or Wyckoff dictionary into a PyG Data object.

    Args:
        data_dict: Dictionary containing either 'graph_arrays' and optional 'spacegroup'.

    Returns:
        PyTorch Geometric Data instance.
    """
    if "spacegroup" not in data_dict:
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
            operation,
            inv_rotation,
            anchor_idxs,
            space_group,
        ) = data_dict["graph_arrays"]
    else:
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["graph_arrays"]
        space_group = torch.tensor([data_dict["spacegroup"]], dtype=torch.long)
        operation = torch.tensor(data_dict["wyckoff_ops"], dtype=torch.float32)
        anchor_idxs = torch.tensor(data_dict["anchors"], dtype=torch.long)
        inv_rotation = torch.linalg.pinv(operation[:, :3, :3])

    edge_index_tensor = (
        torch.tensor(edge_indices.T, dtype=torch.long).contiguous()
        if edge_indices.ndim == 2 and edge_indices.shape[1] == 2
        else torch.empty((2, 0), dtype=torch.long)
    )

    return Data(
        frac_coords=torch.tensor(frac_coords, dtype=torch.float32),
        atom_types=torch.tensor(atom_types, dtype=torch.long),
        lengths=torch.tensor(lengths, dtype=torch.float32).view(1, -1),
        angles=torch.tensor(angles, dtype=torch.float32).view(1, -1),
        edge_index=edge_index_tensor,
        to_jimages=torch.tensor(to_jimages, dtype=torch.long),
        num_atoms=int(num_atoms),
        num_bonds=edge_indices.shape[0] if edge_indices.ndim == 2 else 0,
        num_nodes=int(num_atoms),
        ops=torch.tensor(operation, dtype=torch.float32),
        ops_inv=torch.tensor(inv_rotation, dtype=torch.float32),
        anchor_index=torch.tensor(anchor_idxs, dtype=torch.long),
        spacegroup=torch.tensor(space_group, dtype=torch.long).squeeze(),
    )


class CrystDataset(Dataset):
    """Dataset for crystal structures from CSV tables containing CIF strings."""

    def __init__(
        self,
        path: str | Path,
        mode: str,
        niggli: bool = True,
        primitive: bool = False,
        graph_method: str = "crystalnn",
        cache_dir: str | Path | None = None,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.max_samples = max_samples

        suffix = f"_{max_samples}" if max_samples is not None else ""
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"{mode}{suffix}.pth"
        else:
            cache_path = self.path.parent / f"{mode}{suffix}.pth"

        self.cache_path = cache_path
        self.cached_data: list[dict[str, Any]] = []
        self._load_or_preprocess()

    def _load_or_preprocess(self) -> None:
        if self.cache_path.exists():
            logger.info("Loading cached dataset from %s", self.cache_path)
            self.cached_data = torch.load(self.cache_path, weights_only=False)
            if self.max_samples is not None:
                self.cached_data = self.cached_data[: self.max_samples]
            return

        logger.info("Preprocessing %s ...", self.path)
        df = pd.read_csv(self.path)
        if self.max_samples is not None:
            df = df.iloc[: self.max_samples]

        results = Parallel(n_jobs=-1)(
            delayed(process_one)(
                df.iloc[idx],
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
            )
            for idx in trange(len(df), desc="Extracting graphs")
        )
        self.cached_data = [r for r in results if r is not None]
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.cached_data, self.cache_path)
        logger.info("Saved cache to %s (%d records)", self.cache_path, len(self.cached_data))

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index: int) -> Data:
        return graph_arrays_to_pyg_data(self.cached_data[index])


class WyckoffDataset(Dataset):
    """Dataset for pyXtal Wyckoff representation files (.json or .json.gz)."""

    def __init__(
        self,
        path: str | Path,
        mode: str = "transformer",
        structure_count: int | None = 1000,
        graph_method: str = "crystalnn",
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.structure_count = structure_count
        self.graph_method = graph_method

        if cache_dir is not None:
            self.cache_path = Path(cache_dir) / f"{mode}.pth"
        else:
            self.cache_path = self.path.parent / "cache" / self.path.stem / f"{mode}.pth"

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cached_data: list[dict[str, Any]] = []
        self._load_or_preprocess()

    def _load_or_preprocess(self) -> None:
        if self.cache_path.exists():
            logger.info("Loading cached Wyckoff data from %s", self.cache_path)
            self.cached_data = torch.load(self.cache_path, weights_only=False)
            return

        logger.info("Preprocessing Wyckoff file: %s", self.path)
        if self.path.suffix == ".json":
            with open(self.path, encoding="utf-8") as f:
                raw_data = json.load(f)
        elif self.path.suffixes[-2:] == [".json", ".gz"]:
            with gzip.open(self.path, "rt", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            raise ValueError(f"Unknown file format for Wyckoff data: {self.path}")

        count = len(raw_data) if self.structure_count is None else min(self.structure_count, len(raw_data))
        results = Parallel(n_jobs=-1)(
            delayed(process_one)(
                raw_data[idx],
                niggli=True,
                primitive=False,
                graph_method=self.graph_method,
            )
            for idx in trange(count, desc="Extracting Wyckoff structures")
        )
        self.cached_data = [r for r in results if r is not None]
        torch.save(self.cached_data, self.cache_path)
        logger.info("Saved cached data to %s (%d records)", self.cache_path, len(self.cached_data))

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index: int) -> Data:
        return graph_arrays_to_pyg_data(self.cached_data[index])

