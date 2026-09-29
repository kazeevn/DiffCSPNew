"""Converts a CIF table into a packed, memory-mapped split (diffcsp.data.packed).

CPU work, run it on a CPU node rather than on GPU time:

    python scripts/pack_dataset.py $DATA/train.csv.gz $OUT/train \
        --props energy_above_hull formation_energy_per_atom --n_jobs $NCPUS

Each row goes through ``process_one`` -- the same Niggli reduction and pyXtal
symmetrisation as ``CrystDataset`` -- with ``graph_method="none"``. Chunks are written
to ``<out>.parts/`` as they finish, so an interrupted run resumes; the final split is
assembled from them and ``meta.json`` written last.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from diffcsp.data.graph import process_one
from diffcsp.data.packed import concatenate_packed, pack_records, write_packed

CHUNK_ROWS = 100_000


def _one(args: tuple[str, str, dict[str, float]]) -> dict | None:
    sid, cif, props = args
    rec = process_one({"cif": cif, "immutable_id": sid}, graph_method="none")
    if rec is None:
        return None
    return {"id": sid, "graph_arrays": rec["graph_arrays"], "props": props}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("out", type=Path, help="split directory to create, e.g. cache/lemat/train")
    parser.add_argument("--props", nargs="*", default=[], help="scalar columns to keep as prop_<name>")
    parser.add_argument("--id_column", default="immutable_id")
    parser.add_argument("--n_jobs", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    if (args.out / "meta.json").exists():
        print(f"{args.out} is already complete")
        return
    parts = args.out.with_name(args.out.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_read = n_failed = 0
    part_files = []
    reader = pd.read_csv(args.csv, usecols=[args.id_column, "cif", *args.props], chunksize=CHUNK_ROWS)
    with mp.Pool(args.n_jobs) as pool:
        for k, df in enumerate(reader):
            part = parts / f"part_{k:05d}.npz"
            part_files.append(part)
            n_read += len(df)
            if part.exists():
                continue
            if df[args.props].isna().any().any():
                raise ValueError(f"missing values in {args.props} in rows {n_read - len(df)}..{n_read}")
            jobs = [
                (str(sid), cif, {p: float(v) for p, v in zip(args.props, vals)})
                for sid, cif, *vals in df[[args.id_column, "cif", *args.props]].itertuples(index=False)
            ]
            recs = pool.map(_one, jobs, chunksize=64)
            kept = [r for r in recs if r is not None]
            n_failed += len(recs) - len(kept)
            arrays = pack_records(kept, args.props)
            tmp = part.with_suffix(".tmp.npz")
            np.savez(tmp, **{key: (v.astype(str) if key == "ids" else v) for key, v in arrays.items()})
            tmp.replace(part)
            print(f"[{time.time() - t0:7.0f}s] chunk {k}: {len(kept):,}/{len(df):,} kept ({n_read:,} rows read)", flush=True)

    chunks = []
    for part in part_files:
        with np.load(part) as z:
            chunks.append({key: z[key] for key in z.files})
    arrays = concatenate_packed(chunks)
    n_kept = len(arrays["num_atoms"])
    write_packed(
        args.out,
        arrays,
        {
            "source": str(args.csv),
            "source_sha256": _sha256(args.csv),
            "n_rows": n_read,
            "n_unconvertible": n_read - n_kept,
            "graph_method": "none",
            "code_commit": os.environ.get("DIFFCSP_GIT_COMMIT"),
            "niggli": True,
            "primitive": False,
        },
    )
    for part in part_files:
        part.unlink()
    parts.rmdir()
    print(f"{args.csv}: {n_kept:,} of {n_read:,} structures packed into {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
