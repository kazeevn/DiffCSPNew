"""Control regime: the pyxtal.from_random draw itself, with no refinement.

Establishes how much of any regime's match rate comes from the Wyckoff
representation rather than from the model acting on it.
"""
import argparse, pickle
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--inits", default="bench/inits.pkl")
ap.add_argument("--out", default="bench/pred_none.pkl")
a = ap.parse_args()

with open(a.inits, "rb") as f:
    blob = pickle.load(f)
out = [{"entry": d["entry"], "trial": d["trial"], "pred": d["init"]} for d in blob["draws"]]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "wb") as f:
    pickle.dump(out, f)
print(f"wrote {len(out)} unrefined draws -> {a.out}")
