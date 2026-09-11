"""Splits the benchmark by Wyckoff orbit multiplicity, not just by degrees of freedom.

`dof_analysis.py` asks how much the model has to determine. This asks how much
*redundancy* the conventional cell carries while it does so: a site of
multiplicity m is one anchor repeated m times, so the asymmetric-unit model
replaces m graph nodes with one. Where every orbit has multiplicity 1 the
asymmetric unit is the whole cell and the two architectures are the same
computation -- which makes that subset the natural control for anything
attributed to the asymmetric unit.

Reports per-trial match rates over the DoF x multiplicity grid and paired
McNemar tests between any two regimes. See `docs/asymmetric-unit-deficit.md`.
"""
import argparse, pickle, warnings

import numpy as np

warnings.filterwarnings("ignore")


def representation_stats(rep: dict, group_cache: dict) -> tuple[int, int, int]:
    """Returns (wyckoff_dof, max_orbit_multiplicity, n_sites) for a pyXtal representation."""
    from pyxtal.symmetry import Group

    g = group_cache.setdefault(rep["group"], Group(rep["group"]))
    by_label = {w.get_label(): w for w in g.Wyckoff_positions}
    ws = [by_label[label] for site_list in rep["sites"] for label in site_list]
    return (
        int(sum(w.get_dof() for w in ws)),
        int(max(w.multiplicity for w in ws)),
        len(ws),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", default="runs/bench/mp20/inits.pkl")
    ap.add_argument("--results", nargs="+", required=True,
                    help="one or more results.pkl from evaluate.py; labels are merged")
    ap.add_argument("--regimes", nargs="+", default=None,
                    help="subset of regime labels to report (default: all found)")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"),
                    help="two regime labels to test pairwise with McNemar")
    a = ap.parse_args()

    entries = pickle.load(open(a.inits, "rb"))["entries"]
    n = len(entries)

    table = {}
    for path in a.results:
        for label, v in pickle.load(open(path, "rb")).items():
            table.setdefault(label, v)
    labels = a.regimes if a.regimes else list(table)
    missing = [k for k in labels if k not in table]
    if missing:
        raise SystemExit(f"regimes not found in results: {missing}\navailable: {list(table)}")

    # Flatten to paired per-trial outcomes. Regimes can disagree on trial count
    # (a draw that failed to build), so each entry is truncated to the common length.
    per_entry_len = [min(len(table[k]["per_entry"][i]) for k in labels) for i in range(n)]
    hit = {
        k: np.array([table[k]["per_entry"][i][j] is not None
                     for i in range(n) for j in range(per_entry_len[i])])
        for k in labels
    }
    entry_of_trial = np.array([i for i in range(n) for j in range(per_entry_len[i])])

    cache = {}
    stats = [representation_stats(e["rep"], cache) for e in entries]
    dof = np.array([s[0] for s in stats])
    mult = np.array([s[1] for s in stats])
    nsites = np.array([s[2] for s in stats])
    natoms = np.array([e["gt"].num_sites for e in entries])

    subsets = [
        ("ALL", np.ones(n, bool)),
        ("multiplicity == 1  (architectures identical)", mult == 1),
        ("  ...DoF <= 5", (mult == 1) & (dof <= 5)),
        ("  ...DoF 6-8", (mult == 1) & (dof >= 6) & (dof <= 8)),
        ("  ...DoF >= 9", (mult == 1) & (dof >= 9)),
        ("multiplicity > 1   (orbits collapsed)", mult > 1),
        ("  ...DoF <= 5", (mult > 1) & (dof <= 5)),
        ("  ...DoF 6-8", (mult > 1) & (dof >= 6) & (dof <= 8)),
        ("  ...DoF >= 9", (mult > 1) & (dof >= 9)),
    ]

    print(f"{n} representations; per-trial match rate (%)\n")
    head = f"{'subset':<46}{'n':>5}{'trials':>7}{'sites':>6}{'atoms':>6}  " + \
           " ".join(f"{k[:20]:>20}" for k in labels)
    print(head)
    print("-" * len(head))
    for name, m in subsets:
        if m.sum() == 0:
            continue
        t = m[entry_of_trial]
        cells = " ".join(f"{100 * hit[k][t].mean():>19.1f}%" for k in labels)
        print(f"{name:<46}{m.sum():>5}{t.sum():>7}{nsites[m].mean():>6.1f}{natoms[m].mean():>6.1f}  {cells}")

    if a.compare:
        from scipy.stats import binomtest

        k1, k2 = a.compare
        print(f"\npaired McNemar: {k1} vs {k2}  (discordant trials, + favours {k1})\n")
        print(f"{'subset':<46}{'gap':>8}{'wins':>7}{'losses':>8}{'p':>12}")
        print("-" * 81)
        for name, m in subsets:
            if m.sum() == 0:
                continue
            t = m[entry_of_trial]
            wins = int((hit[k1][t] & ~hit[k2][t]).sum())
            losses = int((hit[k2][t] & ~hit[k1][t]).sum())
            p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
            gap = 100 * (hit[k1][t].mean() - hit[k2][t].mean())
            print(f"{name:<46}{gap:>+8.1f}{wins:>7}{losses:>8}{p:>12.2g}")


if __name__ == "__main__":
    main()
