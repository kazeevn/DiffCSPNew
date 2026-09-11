"""Stage 4: score every regime against MP-20 ground truth with StructureMatcher."""
import os

# Matching is pure CPU work spread over many worker processes; hiding the GPU
# keeps a stray tensor in a prediction from making every worker initialize CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse, pickle
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def _score(gt, pred, stol, angle_tol, ltol):
    if pred is None:
        return None
    from pymatgen.analysis.structure_matcher import StructureMatcher

    m = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
    try:
        r = m.get_rms_dist(gt, pred)
        return None if r is None else float(r[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", default="bench/inits.pkl")
    ap.add_argument("--preds", nargs="+", required=True, help="label=path.pkl")
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle_tol", type=float, default=10.0)
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--out", default="bench/results.pkl")
    a = ap.parse_args()

    with open(a.inits, "rb") as f:
        blob = pickle.load(f)
    entries, trials = blob["entries"], blob["trials"]

    table = {}
    for spec in a.preds:
        label, path = spec.split("=", 1)
        with open(path, "rb") as f:
            preds = pickle.load(f)
        rms = Parallel(n_jobs=-1)(
            delayed(_score)(entries[p["entry"]]["gt"], p["pred"], a.stol, a.angle_tol, a.ltol)
            for p in tqdm(preds, desc=f"matching {label}")
        )
        per_entry = defaultdict(list)
        for p, r in zip(preds, rms):
            per_entry[p["entry"]].append(r)
        n = len(entries)
        all_r = [r for r in rms if r is not None]
        per_trial = np.mean([r is not None for r in rms]) * 100
        best_of_k = np.mean([any(r is not None for r in per_entry[i]) for i in range(n)]) * 100
        table[label] = {
            "per_trial_match_rate": per_trial,
            "best_of_k_match_rate": best_of_k,
            "mean_rms_matched": float(np.mean(all_r)) if all_r else float("nan"),
            "n_generated": len(preds),
            "n_failed": sum(p["pred"] is None for p in preds),
            "per_entry": {i: per_entry[i] for i in range(n)},
        }

    with open(a.out, "wb") as f:
        pickle.dump(table, f)

    print(f"\nMP-20 test subset: {len(entries)} representations x {trials} pyxtal.from_random trials")
    print(f"StructureMatcher(stol={a.stol}, angle_tol={a.angle_tol}, ltol={a.ltol})\n")
    w = max(len(k) for k in table)
    print(f"{'regime':<{w}}  {'match/trial':>12}  {'best-of-' + str(trials):>10}  {'mean RMS':>9}  {'failed':>7}")
    for k, v in table.items():
        print(
            f"{k:<{w}}  {v['per_trial_match_rate']:>11.1f}%  {v['best_of_k_match_rate']:>9.1f}%  "
            f"{v['mean_rms_matched']:>9.4f}  {v['n_failed']:>7}"
        )


if __name__ == "__main__":
    main()
