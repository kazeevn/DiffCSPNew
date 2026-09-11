"""PyTorch Geometric datasets for crystal structures and Wyckoff representations.

Provides CrystDataset for CIF/MP-20 datasets and WyckoffDataset for
pyXtal Wyckoff representation JSON/GZ archives.
"""

import gzip
import json
import logging
import math
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
        max_energy_above_hull: float | None = None,
        max_atoms: int | None = None,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.max_samples = max_samples
        self.max_energy_above_hull = max_energy_above_hull
        self.max_atoms = max_atoms

        # The filter belongs in the cache name: two runs over the same CSV with
        # different stability cuts are different datasets and must not collide.
        suffix = f"_{max_samples}" if max_samples is not None else ""
        if max_energy_above_hull is not None:
            suffix += f"_ehull{max_energy_above_hull:g}".replace(".", "p")
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"{mode}{suffix}.pth"
        else:
            cache_path = self.path.parent / f"{mode}{suffix}.pth"

        self.cache_path = cache_path
        self.cached_data: list[dict[str, Any]] = []
        self._load_or_preprocess()
        self._apply_atom_cap()

    #: Structures per preprocessing shard. Graph extraction runs at roughly
    #: 200/s, so a shard is a couple of minutes of work -- the most a crash,
    #: an OOM or a reboot can cost.
    SHARD_SIZE = 50_000

    @property
    def _shard_dir(self) -> Path:
        return self.cache_path.parent / f"{self.cache_path.stem}_shards"

    def _shard_path(self, index: int) -> Path:
        return self._shard_dir / f"shard_{index:05d}.pth"

    def _load_shards(self, n_shards: int) -> None:
        self.cached_data = []
        for i in trange(n_shards, desc="Loading shards"):
            self.cached_data.extend(torch.load(self._shard_path(i), weights_only=False))

    def _apply_atom_cap(self) -> None:
        """Drops structures with more atoms than ``max_atoms``.

        CSPNet connects every atom in a cell to every other, so cost and memory
        grow as the square of the cell size: one 992-atom conventional cell is
        ~984k edges on its own, more than a whole batch of typical ones, and no
        batch size makes that fit. The cap is applied to the loaded cache rather
        than at preprocessing time, so a single cache serves any cap without
        re-extracting graphs.
        """
        if self.max_atoms is None or not self.cached_data:
            return
        before = len(self.cached_data)
        self.cached_data = [
            r for r in self.cached_data if int(r["graph_arrays"][6]) <= self.max_atoms
        ]
        dropped = before - len(self.cached_data)
        if dropped:
            print(
                f"max_atoms <= {self.max_atoms}: kept {len(self.cached_data):,} of {before:,} "
                f"({100 * len(self.cached_data) / before:.2f}%, dropped {dropped:,})"
            )

    def _load_or_preprocess(self) -> None:
        # A single-file cache from an earlier run stays valid.
        if self.cache_path.exists():
            logger.info("Loading cached dataset from %s", self.cache_path)
            self.cached_data = torch.load(self.cache_path, weights_only=False)
            if self.max_samples is not None:
                self.cached_data = self.cached_data[: self.max_samples]
            return

        # A complete shard set is loadable without touching the source CSV,
        # which for a million-row gzipped table is minutes and gigabytes saved.
        manifest_path = self._shard_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            n_shards = manifest["n_shards"]
            if all(self._shard_path(i).exists() for i in range(n_shards)):
                print(f"Loading {n_shards} cached shards from {self._shard_dir}")
                self._load_shards(n_shards)
                print(f"Loaded {len(self.cached_data):,} structures from cache")
                return

        self._preprocess_sharded()

    def _preprocess_sharded(self) -> None:
        """Extracts graphs shard by shard, skipping shards already on disk.

        Preprocessing a large dataset is hours of CPU work whose result was
        previously held entirely in memory until one final save. Any failure --
        including a single structure raising from a worker -- discarded all of
        it. Shards are written as they complete and skipped on a rerun, so an
        interrupted job resumes where it stopped.
        """
        logger.info("Preprocessing %s ...", self.path)
        df = pd.read_csv(self.path)
        n_read = len(df)

        if self.max_energy_above_hull is not None:
            if "energy_above_hull" not in df.columns:
                raise ValueError(
                    f"{self.path} has no 'energy_above_hull' column, so "
                    f"max_energy_above_hull={self.max_energy_above_hull} cannot be applied. "
                    f"Columns: {list(df.columns)}"
                )
            keep = df["energy_above_hull"] <= self.max_energy_above_hull
            df = df[keep.fillna(False)].reset_index(drop=True)
            print(
                f"E_hull <= {self.max_energy_above_hull} eV: kept {len(df):,} of {n_read:,} "
                f"({100 * len(df) / max(n_read, 1):.1f}%)"
            )

        if self.max_samples is not None:
            df = df.iloc[: self.max_samples]

        n_shards = max(1, math.ceil(len(df) / self.SHARD_SIZE))
        self._shard_dir.mkdir(parents=True, exist_ok=True)
        (self._shard_dir / "manifest.json").write_text(
            json.dumps({
                "source": str(self.path),
                "n_records": int(len(df)),
                "n_shards": int(n_shards),
                "shard_size": int(self.SHARD_SIZE),
                "max_energy_above_hull": self.max_energy_above_hull,
                "max_samples": self.max_samples,
                "graph_method": self.graph_method,
            }, indent=2)
        )

        done = sum(self._shard_path(i).exists() for i in range(n_shards))
        if done:
            print(f"Resuming: {done}/{n_shards} shards already on disk")

        n_dropped = 0
        for i in range(n_shards):
            shard_path = self._shard_path(i)
            if shard_path.exists():
                continue
            chunk = df.iloc[i * self.SHARD_SIZE : (i + 1) * self.SHARD_SIZE]
            results = Parallel(n_jobs=-1)(
                delayed(process_one)(
                    chunk.iloc[j],
                    niggli=self.niggli,
                    primitive=self.primitive,
                    graph_method=self.graph_method,
                )
                for j in trange(len(chunk), desc=f"Shard {i + 1}/{n_shards}")
            )
            kept = [r for r in results if r is not None]
            n_dropped += len(results) - len(kept)
            # Write via a temporary file: a shard interrupted mid-write must not
            # be mistaken for a finished one on the next run.
            tmp = shard_path.with_suffix(".tmp")
            torch.save(kept, tmp)
            tmp.replace(shard_path)

        self._load_shards(n_shards)
        print(
            f"Preprocessed {len(self.cached_data):,} structures into {n_shards} shards"
            + (f" ({n_dropped:,} unconvertible structures dropped)" if n_dropped else "")
        )

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
        self._apply_atom_cap()

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

