"""Stage 3c: relax each pyxtal.from_random draw with the ORB potential.

Uses variable-cell relaxation under a symmetry constraint, so the relaxed
structure keeps the space group of the Wyckoff representation it started from --
the same constraint the two diffusion regimes satisfy by construction. Without
it the comparison would let this regime leave the symmetry sector the other two
are confined to.
"""
import argparse, pickle, time, warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _detach(structure):
    """Rebuilds a Structure with no reference to the ASE calculator.

    A Structure handed back by AseAtomsAdaptor stays transitively reachable from
    the attached calculator, so pickling one drags the whole ORB model -- CUDA
    tensors included -- along with it. That inflates the output file and, worse,
    makes every worker that unpickles a prediction allocate GPU memory.
    """
    from pymatgen.core import Structure

    return Structure(
        lattice=structure.lattice.matrix.copy(),
        species=[site.species_string for site in structure],
        coords=structure.frac_coords.copy(),
        coords_are_cartesian=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", default="bench/inits.pkl")
    ap.add_argument("--orb_model", default="orb_v3_direct_20_mpa")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--no_fix_symmetry", action="store_true")
    ap.add_argument("--out", default="bench/pred_relax.pkl")
    a = ap.parse_args()

    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    model, adapter = getattr(pretrained, a.orb_model)(device="cuda")
    calc = ORBCalculator(model, adapter, device="cuda")

    with open(a.inits, "rb") as f:
        blob = pickle.load(f)
    draws = blob["draws"]
    print(f"{len(draws)} draws to relax (fix_symmetry={not a.no_fix_symmetry})")

    out, n_fail, t0 = [], 0, time.perf_counter()
    for d in tqdm(draws, desc="ORB relaxation"):
        pred = None
        try:
            atoms = AseAtomsAdaptor.get_atoms(d["init"])
            atoms.calc = calc
            if not a.no_fix_symmetry:
                from ase.constraints import FixSymmetry

                atoms.set_constraint(FixSymmetry(atoms))
            FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=a.fmax, steps=a.steps)
            pred = _detach(AseAtomsAdaptor.get_structure(atoms))
        except Exception:
            n_fail += 1
        out.append({"entry": d["entry"], "trial": d["trial"], "pred": pred})

    el = time.perf_counter() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print(f"relaxed {len(out) - n_fail}/{len(out)} in {el/60:.1f} min ({el/len(out):.2f} s each) -> {a.out}")


if __name__ == "__main__":
    main()
