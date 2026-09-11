"""Breaks the benchmark results down by the Wyckoff degrees of freedom of each representation.

Wyckoff DoF is the number of free continuous coordinates the model has to
determine: the sum of per-site DoF (0 for a fully fixed special position, up to
3 for a general position). Lattice DoF -- the free cell parameters allowed by
the crystal family -- is reported alongside it, since it is the other half of
the search space.
"""
import argparse, pickle, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# free lattice parameters per crystal system
LATTICE_DOF = {"triclinic": 6, "monoclinic": 4, "orthorhombic": 3,
               "tetragonal": 2, "trigonal": 2, "hexagonal": 2, "cubic": 1}


def representation_dof(rep: dict) -> tuple[int, int]:
    """Returns (wyckoff_dof, lattice_dof) for a pyXtal representation."""
    from pyxtal.symmetry import Group

    g = Group(rep["group"])
    by_label = {w.get_label(): w for w in g.Wyckoff_positions}
    wdof = 0
    for site_list in rep["sites"]:
        for label in site_list:
            wdof += by_label[label].get_dof()
    return int(wdof), int(LATTICE_DOF.get(g.lattice_type, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/bench/mp20/results.pkl")
    ap.add_argument("--inits", default="runs/bench/mp20/inits.pkl")
    ap.add_argument("--out", default="runs/bench/mp20/dof_table.pkl")
    a = ap.parse_args()

    table = pickle.load(open(a.results, "rb"))
    entries = pickle.load(open(a.inits, "rb"))["entries"]
    labels = list(table)
    n = len(entries)

    dofs = [representation_dof(e["rep"]) for e in entries]
    wdof = np.array([d[0] for d in dofs])
    ldof = np.array([d[1] for d in dofs])
    natoms = np.array([e["gt"].num_sites for e in entries])

    # per-entry best-of-trials success and per-trial success, per regime
    best = {k: np.array([any(r is not None for r in table[k]["per_entry"][i]) for i in range(n)]) for k in labels}
    trial = {k: np.array([np.mean([r is not None for r in table[k]["per_entry"][i]]) for i in range(n)]) for k in labels}

    print(f"Wyckoff DoF distribution over {n} representations:")
    vals, cnts = np.unique(wdof, return_counts=True)
    print("   " + "  ".join(f"{v}:{c}" for v, c in zip(vals, cnts)))
    print(f"   mean={wdof.mean():.2f} median={np.median(wdof):.0f} max={wdof.max()}")
    print(f"lattice DoF: " + "  ".join(f"{v}:{c}" for v, c in zip(*np.unique(ldof, return_counts=True))))
    print(f"correlation(wyckoff DoF, n atoms) = {np.corrcoef(wdof, natoms)[0,1]:.3f}\n")

    bins = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 100)]
    def blabel(lo, hi):
        return f"{lo}" if lo == hi else (f"{lo}+" if hi == 100 else f"{lo}-{hi}")

    rows = []
    print("BEST-OF-3 match rate by Wyckoff DoF")
    hdr = f"{'DoF':>6} {'n':>5} {'atoms':>6} " + " ".join(f"{k[:18]:>18}" for k in labels)
    print(hdr); print("-" * len(hdr))
    for lo, hi in bins:
        m = (wdof >= lo) & (wdof <= hi)
        if m.sum() == 0:
            continue
        cells = [100 * best[k][m].mean() for k in labels]
        rows.append({"dof": blabel(lo, hi), "n": int(m.sum()), "atoms": float(natoms[m].mean()),
                     "best": dict(zip(labels, cells)),
                     "trial": {k: 100 * trial[k][m].mean() for k in labels}})
        print(f"{blabel(lo,hi):>6} {int(m.sum()):>5} {natoms[m].mean():>6.1f} " +
              " ".join(f"{c:>17.1f}%" for c in cells))

    print("\nPER-TRIAL match rate by Wyckoff DoF")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['dof']:>6} {r['n']:>5} {r['atoms']:>6.1f} " +
              " ".join(f"{r['trial'][k]:>17.1f}%" for k in labels))

    # margin of the two diffusion regimes over plain relaxation
    print("\nadvantage of the learned models over ORB relaxation (best-of-3, percentage points)")
    for r in rows:
        o = r["best"].get("orb-diffcsp", float("nan")) - r["best"].get("orb-relax", float("nan"))
        v = r["best"].get("vanilla-diffcsp", float("nan")) - r["best"].get("orb-relax", float("nan"))
        print(f"   DoF {r['dof']:>4}: orb-diffcsp {o:+6.1f}   vanilla {v:+6.1f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"rows": rows, "labels": labels, "wdof": wdof, "ldof": ldof,
                 "natoms": natoms, "best": best, "trial": trial}, open(a.out, "wb"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
