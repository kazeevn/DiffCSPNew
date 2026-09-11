"""Geometry of the generated structures, independent of whether they matched.

`evaluate.py` reports a binary match and an RMS over the matches only, which hides
*how* a regime fails. Two diagnostics that do not condition on success:

- **shortest interatomic distance** -- whether the model is packing atoms into each
  other, against the same quantity in the ground truth;
- **relative cell-volume error** -- whether a deficit sits in the lattice head or the
  coordinate head.

Both are paired: every regime refines the same `from_random` draws. Split by Wyckoff DoF
and by orbit multiplicity, matching `multiplicity_analysis.py`. See
`docs/asymmetric-unit-deficit.md` §4.
"""
import argparse, pickle, warnings

import numpy as np

warnings.filterwarnings("ignore")


def shortest_distance(structure) -> float:
    """Shortest interatomic distance in Angstroms, periodic images included."""
    d = structure.distance_matrix.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", default="runs/bench/mp20/inits.pkl")
    ap.add_argument("--preds", nargs="+", required=True, help="label=path.pkl")
    ap.add_argument("--clash_frac", type=float, default=0.9,
                    help="a draw counts as clashing below this fraction of the true shortest contact")
    a = ap.parse_args()

    from pyxtal.symmetry import Group

    entries = pickle.load(open(a.inits, "rb"))["entries"]
    n = len(entries)

    cache = {}
    dof = np.zeros(n, int)
    mult = np.zeros(n, int)
    for i, e in enumerate(entries):
        rep = e["rep"]
        g = cache.setdefault(rep["group"], Group(rep["group"]))
        by_label = {w.get_label(): w for w in g.Wyckoff_positions}
        ws = [by_label[label] for site_list in rep["sites"] for label in site_list]
        dof[i] = sum(w.get_dof() for w in ws)
        mult[i] = max(w.multiplicity for w in ws)

    gt_dist = np.array([shortest_distance(e["gt"]) for e in entries])
    gt_vol = np.array([e["gt"].volume for e in entries])

    labels, dist, vol = [], {}, {}
    for spec in a.preds:
        label, path = spec.split("=", 1)
        labels.append(label)
        per_dist = [[] for _ in range(n)]
        per_vol = [[] for _ in range(n)]
        for p in pickle.load(open(path, "rb")):
            if p["pred"] is None:
                continue
            per_dist[p["entry"]].append(shortest_distance(p["pred"]))
            per_vol[p["entry"]].append(abs(p["pred"].volume - gt_vol[p["entry"]]) / gt_vol[p["entry"]])
        dist[label] = np.array([np.mean(v) if v else np.nan for v in per_dist])
        vol[label] = np.array([np.mean(v) if v else np.nan for v in per_vol])

    subsets = [
        ("all", np.ones(n, bool)),
        ("DoF <= 5", dof <= 5),
        ("DoF 6-8", (dof >= 6) & (dof <= 8)),
        ("DoF >= 9", dof >= 9),
        ("DoF >= 9, mult == 1", (dof >= 9) & (mult == 1)),
        ("DoF >= 9, mult > 1", (dof >= 9) & (mult > 1)),
    ]
    ok = ~np.isnan(np.stack([dist[k] for k in labels])).any(0)

    print(f"shortest interatomic distance (A), mean over trials; "
          f"'clash' = below {a.clash_frac:g} x the ground-truth contact\n")
    head = f"{'subset':<24}{'n':>5}{'GT':>7}  " + " ".join(f"{k[:20]:>20}" for k in labels)
    print(head); print("-" * len(head))
    for name, m in subsets:
        s = m & ok
        if s.sum() == 0:
            continue
        cells = " ".join(
            f"{dist[k][s].mean():>13.2f}{100 * np.mean(dist[k][s] < a.clash_frac * gt_dist[s]):>6.1f}%"
            for k in labels
        )
        print(f"{name:<24}{s.sum():>5}{gt_dist[s].mean():>7.2f}  {cells}")

    print("\nmedian relative cell-volume error |V_pred - V_gt| / V_gt\n")
    head = f"{'subset':<24}{'n':>5}  " + " ".join(f"{k[:20]:>20}" for k in labels)
    print(head); print("-" * len(head))
    for name, m in subsets:
        s = m & ok
        if s.sum() == 0:
            continue
        print(f"{name:<24}{s.sum():>5}  " + " ".join(f"{np.median(vol[k][s]):>20.4f}" for k in labels))


if __name__ == "__main__":
    main()
