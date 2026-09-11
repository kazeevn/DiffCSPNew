"""Which symmetry gauge does the asymmetric-unit collapse actually discard?

Two gauges are easy to confuse, and only one of them matters here.

**Setting gauge.** A crystal has several equally valid Wyckoff descriptions related by
the space group's Euclidean normalizer -- a different origin or axis choice relabels the
Wyckoff letters. This is what WyckoffTransformer augments over
(`preprocess_wychoffs.get_augmentation_dict`, from `Group(sg).get_alternatives()`). Most
of it is pure origin shifts, and CSPNet only ever consumes `(x_j - x_i) mod 1`, so those
are an exact identity on its inputs -- augmenting over them produces identical gradients.

**Viewpoint gauge.** Any member of a Wyckoff orbit can serve as the anchor; all of them
reconstruct the same crystal. This is the gauge Innovation 1 quotients out when it
replaces an orbit's m nodes with one, and an equivariant denoiser would be blind to it.

Part A counts how much of the setting gauge acts on CSPNet's inputs at all. Part B
(needs `--ckpt`) measures how far a trained asymmetric-unit model is from viewpoint
invariance -- the headroom that representative-randomised augmentation could close.
See `docs/asymmetric-unit-deficit.md` §7.
"""
import argparse, pickle, re, warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")


def acts_on_cspnet(coset_representative: str) -> bool:
    """True if the setting change is more than an origin shift.

    CSPNet is exactly translation invariant, so `x+1/2,y,z` is an identity on its
    inputs; a sign flip or axis mixing such as `-x,-y,-z` or `y,x,z` is not.
    """
    parts = coset_representative.split(",")
    if len(parts) != 3:
        return True
    return not all(
        re.fullmatch(rf"{axis}([+-][\d/.]+)?", p.strip().replace(" ", ""))
        for p, axis in zip(parts, "xyz")
    )


def representation_stats(rep: dict, group_cache: dict) -> tuple[int, int]:
    from pyxtal.symmetry import Group

    g = group_cache.setdefault(rep["group"], Group(rep["group"]))
    by_label = {w.get_label(): w for w in g.Wyckoff_positions}
    ws = [by_label[label] for site_list in rep["sites"] for label in site_list]
    return int(sum(w.get_dof() for w in ws)), int(max(w.multiplicity for w in ws))


def part_a(entries):
    """How much of the setting gauge is visible to CSPNet at all?"""
    from pyxtal.symmetry import Group

    cache = {}
    stats = [representation_stats(e["rep"], cache) for e in entries]
    dof = np.array([s[0] for s in stats])
    mult = np.array([s[1] for s in stats])
    sg = np.array([e["rep"]["group"] for e in entries])

    alts = {}
    for s in sorted(set(sg.tolist())):
        cosets = Group(s).get_alternatives()["Coset Representative"]
        acting = [c for c in cosets if acts_on_cspnet(c)]
        alts[s] = (len(cosets), acting)

    total = np.array([alts[s][0] for s in sg])
    acting = np.array([len(alts[s][1]) for s in sg])

    print("A. setting gauge (what WyckoffTransformer augments over)\n")
    print(f"{'subset':<28}{'n':>5}{'alt settings':>14}{'pure origin shifts':>20}"
          f"{'acting on CSPNet':>18}{'no-op structures':>18}")
    print("-" * 103)
    for label, m in [
        ("all", np.ones(len(sg), bool)),
        ("DoF <= 5", dof <= 5),
        ("DoF >= 9", dof >= 9),
        ("DoF >= 9, mult == 1", (dof >= 9) & (mult == 1)),
        ("DoF >= 9, mult > 1", (dof >= 9) & (mult > 1)),
    ]:
        if m.sum() == 0:
            continue
        print(f"{label:<28}{m.sum():>5}{total[m].mean():>14.1f}"
              f"{(total - acting)[m].mean():>20.1f}{acting[m].mean():>18.1f}"
              f"{100 * np.mean(acting[m] == 0):>17.0f}%")

    contested = (dof >= 9) & (mult > 1)
    print("\n   space groups carrying the DoF>=9 / mult>1 deficit:")
    print(f"   {'sg':>4}{'n':>5}{'alts':>6}{'acting':>8}   non-shift coset representatives")
    for s, n in Counter(sg[contested].tolist()).most_common(8):
        t, act = alts[s]
        shown = ", ".join(act[:4]) + (", ..." if len(act) > 4 else "")
        print(f"   {s:>4}{n:>5}{t:>6}{len(act):>8}   {shown if act else '(none)'}")


def part_b(entries, args):
    """How viewpoint-dependent is a trained asymmetric-unit model?"""
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from torch_geometric.loader import DataLoader

    from run_diffusion import DrawSet
    from diffcsp.core.schedulers import SinusoidalTimeEmbeddings
    from diffcsp.models.layers import generate_asymmetric_edges
    from diffcsp.models.wyckoff_cspnet import WyckoffCSPNet
    from diffcsp.models.wyckoff_diffusion import WyckoffDiffusion

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    draws = pickle.load(open(args.inits, "rb"))["draws"][: args.n_crystals]
    model = WyckoffDiffusion(
        device=dev, decoder=WyckoffCSPNet(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    ).to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    W = model.decoder
    print(f"\n\nB. viewpoint gauge -- checkpoint epoch={ck.get('epoch')}\n")

    batch = next(iter(DataLoader(DrawSet(draws), batch_size=len(draws), shuffle=False))).to(dev)
    n_graphs = batch.num_graphs
    t_emb = SinusoidalTimeEmbeddings(256).to(dev)(
        torch.full((n_graphs,), args.timestep, device=dev, dtype=torch.long)
    )
    x = batch.frac_coords % 1.0
    lattices = torch.randn(n_graphs, 6, device=dev)

    anchors, inv = torch.unique(batch.anchor_index, return_inverse=True, sorted=True)
    site2graph = batch.batch[anchors]
    num_sites = torch.bincount(site2graph, minlength=n_graphs)
    multiplicity = torch.bincount(inv)
    # order[start[k] + r] is the r-th atom of orbit k
    order = torch.argsort(inv, stable=True)
    start = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.long, device=dev), multiplicity[:-1]]), 0
    )

    targets, sources = generate_asymmetric_edges(num_sites, batch.num_atoms, dev)
    source_sites = inv[sources]
    edge2graph = site2graph[targets]

    scores = []
    with torch.no_grad():
        for r in range(args.n_viewpoints):
            rep_atom = order[start + (r % multiplicity.clamp(min=1))]
            ref_pos = x[rep_atom]
            frac_diff = (x[sources] - ref_pos[targets]) % 1.0
            h = W.atom_latent_emb(
                torch.cat(
                    [W.node_embedding(batch.atom_types[anchors].clamp(0, 100)),
                     t_emb.repeat_interleave(num_sites, dim=0)], dim=-1
                )
            )
            for i in range(args.num_layers):
                h = W.csp_layers[i](h, ref_pos, x, lattices, targets, source_sites,
                                    sources, edge2graph, frac_diff=frac_diff)
            s = W.coord_out(h)
            # back to the canonical anchor frame so all viewpoints are comparable
            scores.append((batch.ops_inv[rep_atom] @ s.unsqueeze(-1)).squeeze(-1))

    stacked = torch.stack(scores)
    mean = stacked.mean(0)
    spread = (stacked - mean).norm(dim=-1).mean(0) / (mean.norm(dim=-1) + 1e-9)
    pairwise = (stacked[0] - stacked[1]).norm(dim=-1) / (
        0.5 * (stacked[0].norm(dim=-1) + stacked[1].norm(dim=-1)) + 1e-9
    )

    print(f"{len(anchors)} sites, {int((multiplicity > 1).sum())} with replicas; "
          f"{args.n_viewpoints} viewpoints per site")
    print("an equivariant denoiser would score 0 everywhere; m == 1 is the null\n")
    print(f"{'orbit size m':>14}{'n sites':>9}{'rel. spread':>14}{'two-viewpoint rel. diff':>26}")
    print("-" * 63)
    for lo, hi, label in [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 10**6, "9+")]:
        sel = (multiplicity >= lo) & (multiplicity <= hi)
        if sel.sum() < 5:
            continue
        # the spread ratio is unstable where the mean score is near zero; the
        # pairwise column does not divide by a cancelling mean
        print(f"{label:>14}{int(sel.sum()):>9}{spread[sel].mean():>14.3f}"
              f"{pairwise[sel].mean():>26.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", default="runs/bench/mp20/inits.pkl")
    ap.add_argument("--ckpt", default=None,
                    help="asymmetric-unit checkpoint; omit to run the crystallography part only")
    ap.add_argument("--n_crystals", type=int, default=512)
    ap.add_argument("--n_viewpoints", type=int, default=4)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--timestep", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    entries = pickle.load(open(a.inits, "rb"))["entries"]
    part_a(entries)
    if a.ckpt:
        part_b(entries, a)


if __name__ == "__main__":
    main()
