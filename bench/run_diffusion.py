"""Stage 3a/3b: generate structures with a diffusion model (orb-diffcsp or vanilla CSPNet)."""
import argparse, pickle, re, warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

from diffcsp.data.dataset import graph_arrays_to_pyg_data


class DrawSet(Dataset):
    def __init__(self, draws):
        self.draws = draws

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, i):
        return graph_arrays_to_pyg_data({"graph_arrays": self.draws[i]["graph_arrays"]})


def load_vanilla(model, ckpt_path: str):
    """Loads an original DiffCSP++ CSPNet checkpoint into this repo's CSPDiffusion.

    Three incompatibilities are handled: a torch.compile ``_orig_mod.`` prefix,
    per-layer attribute names (``csp_layer_N``) where this repo uses a
    ModuleList (``csp_layers.N``), and a node embedding of MAX_ATOMIC_NUM=100
    rows indexed by Z-1, where this repo allocates 101 rows indexed by Z.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Checkpoints written by this repo wrap the weights alongside optimizer state
    # and metadata; the original DiffCSP++ release is a bare state dict.
    if isinstance(raw, dict) and "model_state_dict" in raw:
        print(
            f"checkpoint epoch={raw.get('epoch')} train_loss={raw.get('train_loss')} "
            f"val_loss={raw.get('val_loss')}"
        )
        raw = raw["model_state_dict"]
    sd = {}
    for k, v in raw.items():
        k = re.sub(r"^_orig_mod\.", "", k)
        k = re.sub(r"^decoder\.csp_layer_(\d+)\.", lambda m: f"decoder.csp_layers.{m.group(1)}.", k)
        sd[k] = v
    ne = "decoder.node_embedding.weight"
    if ne in sd and sd[ne].shape[0] == 100:
        w = torch.zeros(101, sd[ne].shape[1], dtype=sd[ne].dtype)
        w[1:101] = sd[ne]  # checkpoint row i corresponds to atomic number i+1
        sd[ne] = w
    res = model.load_state_dict(sd, strict=False)
    learned = [
        k for k in res.missing_keys
        if not any(s in k for s in ("scheduler", "time_embedding", "crystal_family", "dis_emb"))
    ]
    if learned:
        raise RuntimeError(f"vanilla checkpoint is missing learned tensors: {learned}")
    print(f"loaded vanilla checkpoint ({len(sd)} tensors; {len(res.missing_keys)} regenerated buffers)")
    return model


def to_structures(out):
    from pymatgen.core import Structure

    frac = out["frac_coords"].detach().cpu()
    nat = out["num_atoms"].detach().cpu()
    typ = out["atom_types"].detach().cpu()
    lat = out["lattices"].detach().cpu()
    res, s = [], 0
    for i, n in enumerate(nat.tolist()):
        try:
            res.append(
                Structure(
                    lattice=lat[i].numpy(),
                    species=typ[s : s + n].numpy(),
                    coords=frac[s : s + n].numpy(),
                    coords_are_cartesian=False,
                )
            )
        except Exception:
            res.append(None)
        s += n
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["orb", "cspnet", "wyckoff", "asymm", "painn"], required=True)
    ap.add_argument("--inits", default="runs/bench/mp20/inits.pkl")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--num_layers", type=int, default=None)
    ap.add_argument("--orb_model", default="orb-v3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-orbit-average",
        action="store_true",
        help="cspnet regime only: use the anchor replica's score instead of the orbit mean "
             "(ablation for docs/asymmetric-unit-deficit.md)",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    with open(a.inits, "rb") as f:
        blob = pickle.load(f)
    draws = blob["draws"]
    print(f"{len(draws)} draws to generate ({a.regime})")

    if a.regime == "orb":
        from diffcsp.models.diffusion_orb import CSPDiffusionORB
        from diffcsp.cli.train import _load_adapter_state_dict

        h = a.hidden_dim if a.hidden_dim is not None else 128
        l = a.num_layers if a.num_layers is not None else 2
        model = CSPDiffusionORB(
            device=dev, orb_model_name=a.orb_model, hidden_dim=h, num_layers=l
        ).to(dev)
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        _load_adapter_state_dict(model.decoder, sd)
        print(f"checkpoint epoch={ck.get('epoch')} val_loss={ck.get('val_loss')}")
    elif a.regime == "painn":
        from diffcsp.models.wyckoff_diffusion import WyckoffDiffusion
        from diffcsp.models.wyckoff_painn import WyckoffPaiNN

        h = a.hidden_dim if a.hidden_dim is not None else 512
        l = a.num_layers if a.num_layers is not None else 6
        model = WyckoffDiffusion(
            device=dev,
            decoder=WyckoffPaiNN(hidden_dim=h, num_layers=l),
        ).to(dev)
        raw = torch.load(a.ckpt, map_location=dev, weights_only=False)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            print(f"checkpoint epoch={raw.get('epoch')} train_loss={raw.get('train_loss')} val_loss={raw.get('val_loss')}")
            model.load_state_dict(raw["model_state_dict"])
        elif isinstance(raw, dict):
            model.load_state_dict(raw)
    elif a.regime in ("wyckoff", "asymm"):
        from diffcsp.models.wyckoff_diffusion import WyckoffDiffusion
        from diffcsp.models.wyckoff_cspnet import WyckoffCSPNet

        h = a.hidden_dim if a.hidden_dim is not None else 512
        l = a.num_layers if a.num_layers is not None else 6
        model = WyckoffDiffusion(
            device=dev,
            decoder=WyckoffCSPNet(hidden_dim=h, num_layers=l),
        ).to(dev)
        raw = torch.load(a.ckpt, map_location=dev, weights_only=False)
        if isinstance(raw, dict) and raw.get("model_type") in ("wyckoff", "asymm"):
            print(f"checkpoint epoch={raw.get('epoch')} train_loss={raw.get('train_loss')} val_loss={raw.get('val_loss')}")
            model.load_state_dict(raw["model_state_dict"])
        else:
            load_vanilla(model, a.ckpt)
    else:
        from diffcsp.models.cspnet import CSPNet
        from diffcsp.models.diffusion import CSPDiffusion

        h = a.hidden_dim if a.hidden_dim is not None else 512
        l = a.num_layers if a.num_layers is not None else 6
        model = CSPDiffusion(
            device=dev,
            decoder=CSPNet(hidden_dim=h, num_layers=l),
        ).to(dev)
        load_vanilla(model, a.ckpt)
    model.eval()

    loader = DataLoader(DrawSet(draws), batch_size=a.batch_size, shuffle=False, num_workers=4)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"sampling ({a.regime})"):
            kw = {"orbit_average": False} if (a.no_orbit_average and a.regime == "cspnet") else {}
            out, _ = model.sample(batch.to(dev), disable_progress=True, **kw)
            preds.extend(to_structures(out))

    assert len(preds) == len(draws), f"{len(preds)} predictions for {len(draws)} draws"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(
            [{"entry": d["entry"], "trial": d["trial"], "pred": p} for d, p in zip(draws, preds)], f
        )
    print(f"wrote {len(preds)} structures ({sum(p is None for p in preds)} failed) -> {a.out}")


if __name__ == "__main__":
    main()
