"""Stage 1: sample MP-20 test structures and derive their Wyckoff representations.

Writes bench/benchset.pkl holding, per entry, the ground-truth structure and the
pyXtal representation (group / species / numIons / sites) that the three regimes
are all conditioned on.
"""
import argparse, gzip, json, pickle, re, warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")


def wyckoff_representation(structure, tol: float = 0.1):
    """Derives the pyXtal from_random representation of a structure."""
    from pyxtal import pyxtal

    c = pyxtal()
    c.from_seed(structure, tol=tol)
    grouped = OrderedDict()
    for site in c.atom_sites:
        grouped.setdefault(site.specie, []).append(site.wp.get_label())
    species = list(grouped)
    sites = [grouped[sp] for sp in species]
    num_ions = [sum(int(re.match(r"(\d+)", lab).group(1)) for lab in grouped[sp]) for sp in species]
    return {"group": c.group.number, "species": species, "numIons": num_ions, "sites": sites}


def _one(cif: str, mp_id: str, max_atoms=None):
    from diffcsp.data.graph import build_crystal

    try:
        gt = build_crystal(cif, niggli=True, primitive=False)
        rep = wyckoff_representation(gt)
        # a representation is only usable if from_random can instantiate it
        from pyxtal import pyxtal

        probe = pyxtal()
        probe.from_random(**rep, max_count=30)
        n_conv = probe.to_pymatgen(resort=False).num_sites
        if n_conv != gt.num_sites:
            return None
        # Match the cell-size cut the model was trained under; a representation
        # the denoiser never saw at training time is not a fair test of it.
        if max_atoms is not None and n_conv > max_atoms:
            return None
        return {"mp_id": mp_id, "gt": gt, "rep": rep, "n_atoms": n_conv}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="data/mp-20/test.csv")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/bench/mp20/benchset.pkl")
    ap.add_argument("--max_e_hull", type=float, default=None)
    ap.add_argument("--max_atoms", type=int, default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.test_csv)
    print(f"{a.test_csv}: {len(df):,} rows")
    if a.max_e_hull is not None:
        n0 = len(df)
        df = df[df["energy_above_hull"].fillna(np.inf) <= a.max_e_hull].reset_index(drop=True)
        print(f"  E_hull <= {a.max_e_hull} eV: kept {len(df):,} of {n0:,} ({100 * len(df) / n0:.1f}%)")
    rng = np.random.default_rng(a.seed)
    # oversample: some representations fail to round-trip through from_random
    order = rng.permutation(len(df))
    picked, idx = [], 0
    pbar = tqdm(total=a.n, desc="Building benchmark set")
    while len(picked) < a.n and idx < len(order):
        chunk = order[idx : idx + 256]
        idx += 256
        rows = [
            (
                df.iloc[int(i)]["cif"],
                str(df.iloc[int(i)].get("material_id") or df.iloc[int(i)].get("immutable_id") or i),
            )
            for i in chunk
        ]
        got = Parallel(n_jobs=-1)(delayed(_one)(c, m, a.max_atoms) for c, m in rows)
        for g in got:
            if g is not None and len(picked) < a.n:
                picked.append(g)
                pbar.update(1)
    pbar.close()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(picked, f)
    print(f"kept {len(picked)} entries (screened {idx} candidates) -> {a.out}")

    reps = [p["rep"] for p in picked]
    with gzip.open(a.out.replace(".pkl", "_wyckoff.json.gz"), "wt") as f:
        json.dump(reps, f)
    sizes = [p["gt"].num_sites for p in picked]
    sgs = [p["rep"]["group"] for p in picked]
    print(f"atoms/structure: mean={np.mean(sizes):.1f} min={min(sizes)} max={max(sizes)}")
    print(f"distinct space groups: {len(set(sgs))}")


if __name__ == "__main__":
    main()
