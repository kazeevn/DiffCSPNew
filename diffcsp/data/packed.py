"""Packed, memory-mapped crystal datasets for multi-million-structure training.

``CrystDataset`` keeps one Python dict of NumPy arrays per structure. At LeMat-Bulk
scale (5M structures) that is tens of GB per process, and under DDP every rank --
and, through reference-count copy-on-write, every DataLoader worker -- holds its own
copy. A packed split instead stores each field as one flat array on disk:

    <root>/<split>/
        meta.json                  field list, counts, provenance
        ids.npy                    (S,)   structure ids (immutable_id)
        num_atoms.npy              (S,)   int32
        atom_offset.npy            (S,)   int64, start of each structure in the atom arrays
        lengths.npy, angles.npy    (S, 3) float32
        spacegroup.npy             (S,)   int16
        prop_<name>.npy            (S,)   float32, one per scalar property column
        frac_coords.npy            (A, 3) float32
        atom_types.npy             (A,)   uint8
        ops.npy                    (A, 4, 4) float32, Wyckoff affine operations
        ops_inv.npy                (A, 3, 3) float32
        anchor_index.npy           (A,)   int32, relative to the structure's first atom

Every array is opened with ``mmap_mode="r"``, so all ranks and workers share one copy
in the page cache. The values are exactly those of ``process_one`` (the same
``build_crystal`` / pyXtal path, cast to the dtypes ``graph_arrays_to_pyg_data``
produces anyway); only the CrystalNN edge list is dropped, which the full-cell models
(CSPNet, Geo, GeoV2) never read -- they build their own intra-cell graph.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data

STRUCT_FIELDS = ("num_atoms", "atom_offset", "lengths", "angles", "spacegroup")
ATOM_FIELDS = ("frac_coords", "atom_types", "ops", "ops_inv", "anchor_index")


def pack_records(records: Sequence[dict[str, Any]], prop_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Packs ``process_one`` outputs (each extended with ``props``) into flat arrays.

    ``atom_offset`` is relative to this chunk; ``concatenate_packed`` rebases it.
    """
    n_atoms = np.array([int(r["graph_arrays"][6]) for r in records], dtype=np.int32)
    offsets = np.zeros(len(records), dtype=np.int64)
    if len(records):
        offsets[1:] = np.cumsum(n_atoms[:-1], dtype=np.int64)
    out: dict[str, np.ndarray] = {
        "ids": np.array([str(r["id"]) for r in records], dtype=object),
        "num_atoms": n_atoms,
        "atom_offset": offsets,
        "lengths": np.array([r["graph_arrays"][2] for r in records], dtype=np.float32).reshape(-1, 3),
        "angles": np.array([r["graph_arrays"][3] for r in records], dtype=np.float32).reshape(-1, 3),
        "spacegroup": np.array([int(r["graph_arrays"][10]) for r in records], dtype=np.int16),
    }
    for name in prop_names:
        out[f"prop_{name}"] = np.array([r["props"][name] for r in records], dtype=np.float32)

    def cat(i: int, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        if not records:
            return np.empty((0, *shape), dtype=dtype)
        return np.concatenate([np.asarray(r["graph_arrays"][i]).reshape(-1, *shape) for r in records]).astype(dtype)

    out["frac_coords"] = cat(0, np.float32, (3,))
    out["atom_types"] = cat(1, np.uint8, ())
    out["ops"] = cat(7, np.float32, (4, 4))
    out["ops_inv"] = cat(8, np.float32, (3, 3))
    out["anchor_index"] = cat(9, np.int32, ())
    return out


def concatenate_packed(chunks: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenates packed chunks, rebasing ``atom_offset`` onto the joined atom arrays."""
    out: dict[str, np.ndarray] = {}
    base = 0
    rebased = []
    for c in chunks:
        rebased.append(c["atom_offset"] + base)
        base += int(c["num_atoms"].sum(dtype=np.int64))
    for key in chunks[0]:
        out[key] = np.concatenate([c[key] for c in chunks]) if key != "atom_offset" else np.concatenate(rebased)
    return out


def write_packed(split_dir: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    """Writes a packed split, ``meta.json`` last so that its presence marks completion."""
    split_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in arrays.items():
        if key == "ids":
            np.save(split_dir / "ids.npy", arr.astype(str))
        else:
            np.save(split_dir / f"{key}.npy", np.ascontiguousarray(arr))
    meta = dict(meta)
    meta.update(
        n_structures=int(len(arrays["num_atoms"])),
        n_atoms=int(arrays["num_atoms"].sum(dtype=np.int64)),
        props=sorted(k[len("prop_"):] for k in arrays if k.startswith("prop_")),
    )
    (split_dir / "meta.json").write_text(json.dumps(meta, indent=2))


class PackedCrystDataset(Dataset):
    """A packed split, optionally filtered, yielding the same ``Data`` as ``CrystDataset``.

    Args:
        split_dir: ``<root>/<split>`` written by ``scripts/pack_dataset.py``.
        cond_props: property columns returned as ``data.props`` (shape ``(1, P)``), in order.
        max_atoms: drop structures with more atoms (the full-cell graph costs N^2).
        max_e_hull: keep only ``energy_above_hull <= max_e_hull``.
        max_samples: keep a seeded random subset of this size (after filtering).
        subset_seed: seed of that subset, so a validation subset is fixed across runs.
    """

    def __init__(
        self,
        split_dir: str | Path,
        cond_props: Sequence[str] = (),
        max_atoms: int | None = None,
        max_e_hull: float | None = None,
        max_samples: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        super().__init__()
        self.split_dir = Path(split_dir)
        meta_path = self.split_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} missing: {self.split_dir} is not a complete packed split")
        self.meta = json.loads(meta_path.read_text())
        self.cond_props = list(cond_props)

        def load(name: str) -> np.ndarray:
            return np.load(self.split_dir / f"{name}.npy", mmap_mode="r")

        # Per-structure arrays are small; hold them in memory. Atom arrays stay mapped.
        self.num_atoms = np.asarray(load("num_atoms"))
        self.atom_offset = np.asarray(load("atom_offset"))
        self.lengths = np.asarray(load("lengths"))
        self.angles = np.asarray(load("angles"))
        self.spacegroup = np.asarray(load("spacegroup"))
        self._atoms = {name: load(name) for name in ATOM_FIELDS}
        missing = [p for p in self.cond_props if p not in self.meta["props"]]
        if missing:
            raise ValueError(f"{self.split_dir} has no properties {missing}; it has {self.meta['props']}")
        self.props = (
            np.stack([np.asarray(load(f"prop_{p}")) for p in self.cond_props], axis=1).astype(np.float32)
            if self.cond_props
            else None
        )

        keep = np.ones(len(self.num_atoms), dtype=bool)
        self.filter_log: list[str] = []
        if max_e_hull is not None:
            e_hull = np.asarray(load("prop_energy_above_hull"))
            m = e_hull <= max_e_hull
            self.filter_log.append(f"E_hull <= {max_e_hull:g}: {int(m.sum()):,} of {len(m):,}")
            keep &= m
        if max_atoms is not None:
            m = self.num_atoms <= max_atoms
            self.filter_log.append(f"atoms <= {max_atoms}: {int((m & keep).sum()):,} of {int(keep.sum()):,}")
            keep &= m
        self.indices = np.flatnonzero(keep)
        if max_samples is not None and max_samples < len(self.indices):
            rng = np.random.default_rng(subset_seed)
            self.indices = np.sort(rng.choice(self.indices, size=max_samples, replace=False))
            self.filter_log.append(f"random subset (seed {subset_seed}): {max_samples:,}")

    def __len__(self) -> int:
        return len(self.indices)

    def sizes(self) -> np.ndarray:
        """Atom counts of the kept structures, in dataset order (for batch sampling)."""
        return self.num_atoms[self.indices]

    def __getitem__(self, index: int) -> Data:
        i = int(self.indices[index])
        n = int(self.num_atoms[i])
        a = slice(int(self.atom_offset[i]), int(self.atom_offset[i]) + n)
        data = Data(
            frac_coords=torch.from_numpy(np.array(self._atoms["frac_coords"][a])),
            atom_types=torch.from_numpy(np.array(self._atoms["atom_types"][a], dtype=np.int64)),
            lengths=torch.from_numpy(np.array(self.lengths[i])).view(1, -1),
            angles=torch.from_numpy(np.array(self.angles[i])).view(1, -1),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            to_jimages=torch.empty((0, 3), dtype=torch.long),
            num_atoms=n,
            num_bonds=0,
            num_nodes=n,
            ops=torch.from_numpy(np.array(self._atoms["ops"][a])),
            ops_inv=torch.from_numpy(np.array(self._atoms["ops_inv"][a])),
            anchor_index=torch.from_numpy(np.array(self._atoms["anchor_index"][a], dtype=np.int64)),
            spacegroup=torch.tensor(int(self.spacegroup[i]), dtype=torch.long),
        )
        if self.props is not None:
            data.props = torch.from_numpy(self.props[i]).view(1, -1)
        return data


class EdgeBudgetBatchSampler(Sampler[list[int]]):
    """Batches whose full-cell edge count, sum of N^2, stays within a budget.

    The full-cell denoisers connect every pair of atoms in a cell, so memory and time
    per batch grow with sum(N^2), not with the number of structures. A fixed batch
    size either wastes the GPU on batches of small cells or runs out of memory on a
    batch of large ones. Batches are formed greedily over a seeded shuffle.

    Under DDP every rank builds the same global list of batches from the same seed
    and takes every ``world_size``-th one; the list is truncated to a multiple of
    ``world_size``, so all ranks run the same number of steps and none waits at a
    collective for a rank that has run out of data.

    Call ``set_epoch`` before each epoch: the shuffle depends only on (seed, epoch),
    so a resumed run sees the same batches as an uninterrupted one.
    """

    def __init__(
        self,
        sizes: np.ndarray,
        max_edges: int,
        shuffle: bool = True,
        seed: int = 17,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.cost = np.asarray(sizes, dtype=np.int64) ** 2
        if len(self.cost) and self.cost.max() > max_edges:
            raise ValueError(
                f"A structure of {int(math.isqrt(int(self.cost.max())))} atoms needs {int(self.cost.max())} edges, "
                f"above max_edges={max_edges}: lower --max_atoms or raise the budget"
            )
        self.max_edges = max_edges
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self._batches: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = None

    def _global_batches(self) -> list[list[int]]:
        if self.shuffle:
            order = np.random.default_rng(self.seed + 1_000_003 * self.epoch).permutation(len(self.cost))
        else:
            order = np.arange(len(self.cost))
        batches: list[list[int]] = []
        cur: list[int] = []
        load = 0
        for i in order:
            c = int(self.cost[i])
            if cur and load + c > self.max_edges:
                batches.append(cur)
                cur, load = [], 0
            cur.append(int(i))
            load += c
        if cur:
            batches.append(cur)
        n = len(batches) - len(batches) % self.world_size if len(batches) >= self.world_size else len(batches)
        return batches[:n]

    def _mine(self) -> list[list[int]]:
        if self._batches is None:
            self._batches = self._global_batches()[self.rank :: self.world_size]
        return self._batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._mine()

    def __len__(self) -> int:
        return len(self._mine())
