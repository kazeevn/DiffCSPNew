"""Stage 2: instantiate each Wyckoff representation with pyxtal.from_random.

Every regime is then run on exactly these draws, so the three are compared on
identical initializations rather than on independent random draws.
"""
import argparse, pickle, warnings
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _draw(rep: dict, seed: int):
    """One from_random draw, returned as graph arrays plus the initial structure."""
    from pyxtal import pyxtal
    from diffcsp.data.graph import crystal_graph_from_pyxtal

    try:
        c = pyxtal()
        c.from_random(**rep, max_count=30, random_state=seed)
        arrays = crystal_graph_from_pyxtal(c, graph_method="crystalnn")
        return {"graph_arrays": arrays, "init": c.to_pymatgen(resort=False)}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchset", default="bench/benchset.pkl")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="bench/inits.pkl")
    a = ap.parse_args()

    with open(a.benchset, "rb") as f:
        entries = pickle.load(f)
    if a.limit:
        entries = entries[: a.limit]
    print(f"{len(entries)} representations x {a.trials} trials")

    jobs = [(i, t) for i in range(len(entries)) for t in range(a.trials)]
    got = Parallel(n_jobs=-1)(
        delayed(_draw)(entries[i]["rep"], 1000 * t + i) for i, t in tqdm(jobs, desc="from_random draws")
    )

    out, n_fail = [], 0
    for (i, t), g in zip(jobs, got):
        if g is None:
            n_fail += 1
            continue
        out.append({"entry": i, "trial": t, **g})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"entries": entries, "draws": out, "trials": a.trials}, f)
    print(f"{len(out)} draws kept, {n_fail} failed -> {a.out}")


if __name__ == "__main__":
    main()
