"""Builds the CrystDataset graph cache for a CSV split ahead of training.

Training builds a missing cache itself, but on a GPU node that is hours of CPU work
billed as GPU time. This does the same work -- the same class, the same cache path --
on whatever node runs it, and resumes shard by shard if interrupted:

    python scripts/preprocess_dataset.py data.csv.gz --mode train_sym --cache_dir cache/x
"""

import argparse
from pathlib import Path

from diffcsp.data.dataset import CrystDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--mode", required=True, help="train_sym for --train_csv, test_sym for --test_csv")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--max_e_hull", type=float, default=None)
    args = parser.parse_args()

    ds = CrystDataset(args.csv, mode=args.mode, cache_dir=args.cache_dir, max_energy_above_hull=args.max_e_hull)
    print(f"{args.csv} [{args.mode}]: {len(ds):,} structures cached under {ds.cache_path.parent}")


if __name__ == "__main__":
    main()
