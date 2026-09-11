"""Same weights, two graphs: where does the asymmetric unit stop matching the full cell?

Innovation 1 replaces the N-atom cell graph with K Wyckoff anchors. The anchor's
neighbour set is unchanged -- it still sees all N atoms, so every
symmetry-distinct contact is still an edge -- and at layer 0 every replica of a
site carries the same species embedding. The first message-passing step is
therefore identical in both graphs. What differs from the second step on is the
*source node state*: the full cell gives neighbour j a state built from j's own
viewpoint (`layers.py`, CSPLayer), while the collapsed graph hands it its
anchor's state (`layers.py`, WyckoffCSPLayer). Because the edge featurisation
`sin/cos((x_j - x_i) mod 1)` is not equivariant, those are not the same vector.

This script loads one vanilla checkpoint into both decoders -- the Wyckoff layers
reuse CSPLayer's parameter names, so the transfer is exact -- runs them on
identical inputs and reports how far the anchor's representation and its
predicted coordinate score have drifted apart after each layer, split by whether
the site's orbit has replicas.

See `docs/asymmetric-unit-deficit.md`.
"""
import argparse, pickle, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", default="runs/bench/mp20/inits.pkl")
    ap.add_argument("--ckpt", default="data/mp-20/test_ckpt.pt")
    ap.add_argument("--n_crystals", type=int, default=512)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--timestep", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from torch_geometric.loader import DataLoader

    from run_diffusion import DrawSet, load_vanilla
    from diffcsp.core.schedulers import SinusoidalTimeEmbeddings
    from diffcsp.models.cspnet import CSPNet
    from diffcsp.models.diffusion import CSPDiffusion
    from diffcsp.models.layers import generate_asymmetric_edges, generate_intra_crystal_edges
    from diffcsp.models.wyckoff_cspnet import WyckoffCSPNet
    from diffcsp.models.wyckoff_diffusion import WyckoffDiffusion

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    draws = pickle.load(open(a.inits, "rb"))["draws"][: a.n_crystals]
    kw = {"hidden_dim": a.hidden_dim, "num_layers": a.num_layers}
    full = CSPDiffusion(device=dev, decoder=CSPNet(**kw)).to(dev).eval()
    load_vanilla(full, a.ckpt)
    asym = WyckoffDiffusion(device=dev, decoder=WyckoffCSPNet(**kw)).to(dev).eval()
    asym.load_vanilla_checkpoint(a.ckpt)
    V, W = full.decoder, asym.decoder

    batch = next(iter(DataLoader(DrawSet(draws), batch_size=len(draws), shuffle=False))).to(dev)
    n_graphs = batch.num_graphs
    t_emb = SinusoidalTimeEmbeddings(256).to(dev)(
        torch.full((n_graphs,), a.timestep, device=dev, dtype=torch.long)
    )
    x = batch.frac_coords % 1.0
    lattices = torch.randn(n_graphs, 6, device=dev)

    anchors, inv = torch.unique(batch.anchor_index, return_inverse=True, sorted=True)
    site2graph = batch.batch[anchors]
    num_sites = torch.bincount(site2graph, minlength=n_graphs)
    multiplicity = torch.bincount(inv)

    with torch.no_grad():
        edges, frac_diff_full = generate_intra_crystal_edges(batch.num_atoms, x)
        h_full = V.atom_latent_emb(
            torch.cat(
                [V.node_embedding(batch.atom_types.clamp(0, 100)),
                 t_emb.repeat_interleave(batch.num_atoms, dim=0)], dim=-1
            )
        )
        targets, sources = generate_asymmetric_edges(num_sites, batch.num_atoms, dev)
        source_sites = inv[sources]
        frac_diff_asym = (x[sources] - x[anchors][targets]) % 1.0
        h_asym = W.atom_latent_emb(
            torch.cat(
                [W.node_embedding(batch.atom_types[anchors].clamp(0, 100)),
                 t_emb.repeat_interleave(num_sites, dim=0)], dim=-1
            )
        )

        print(f"{n_graphs} crystals, {int(batch.num_atoms.sum())} atoms, {len(anchors)} Wyckoff sites")
        print(f"layer-0 anchor states identical: "
              f"{torch.allclose(h_full[anchors], h_asym, atol=1e-5)}\n")
        print(f"{'after layer':>12}{'anchor state':>16}{'coord score':>14}"
              f"{'state, m==1':>14}{'state, m>1':>13}")
        print("-" * 69)
        for i in range(a.num_layers):
            h_full = V.csp_layers[i](
                h_full, x, lattices, edges, batch.batch[edges[0]], frac_diff=frac_diff_full
            )
            h_asym = W.csp_layers[i](
                h_asym, x[anchors], x, lattices, targets, source_sites, sources,
                site2graph[targets], frac_diff=frac_diff_asym,
            )
            ref = h_full[anchors]
            rel = (ref - h_asym).norm(dim=-1) / (ref.norm(dim=-1) + 1e-9)
            s_full, s_asym = V.coord_out(ref), W.coord_out(h_asym)
            rel_score = (s_full - s_asym).norm(dim=-1) / (s_full.norm(dim=-1) + 1e-9)
            single = multiplicity == 1
            print(f"{i + 1:>12}{rel.mean():>16.2e}{rel_score.mean():>14.2e}"
                  f"{rel[single].mean():>14.2e}{rel[~single].mean():>13.2e}")


if __name__ == "__main__":
    main()
