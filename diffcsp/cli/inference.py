"""Inference CLI for DiffCSP++ structure generation from Wyckoff representations."""

import argparse
import gzip
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from diffcsp.data.dataset import WyckoffDataset
from diffcsp.models.diffusion import CSPDiffusion
from diffcsp.models.diffusion_orb import CSPDiffusionORB

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42) -> None:
    """Sets random seeds for deterministic generation."""
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_structures(
    wyckoff_file: str,
    ckpt_path: str = "diffcsp_ckpt.pt",
    model_type: str = "orb",
    orb_model: str = "orb-v2",
    mock_orb: bool = False,
    batch_size: int = 256,
    hidden_dim: int = 128,
    num_layers: int = 2,
    n_structures: int = 1100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: str | None = None,
) -> Path:
    """Generates crystal structures from a Wyckoff file using DiffCSP++."""
    set_random_seed(42)
    dev = torch.device(device)
    wyckoff_path = Path(wyckoff_file)

    testset = WyckoffDataset(wyckoff_path, mode="transformer", structure_count=n_structures)
    print(f"Loaded {len(testset)} Wyckoff structures for generation.")
    test_loader = DataLoader(testset, shuffle=False, batch_size=batch_size)

    if model_type == "orb":
        model = CSPDiffusionORB(
            device=dev,
            orb_model_name=orb_model,
            use_mock_orb=mock_orb,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(dev)
    else:
        model = CSPDiffusion(device=dev).to(dev)

    # Load checkpoint if available
    ckpt_file = Path(ckpt_path)
    if ckpt_file.exists():
        state_dict = torch.load(ckpt_file, map_location=dev, weights_only=True)
        if "decoder.coord_out.weight" in state_dict or "decoder.csp_layers.0.edge_mlp.0.weight" in state_dict:
            model.load_state_dict(state_dict, strict=False)
        elif hasattr(model, "decoder"):
            model.decoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {ckpt_file}")
    else:
        print(f"Warning: Checkpoint '{ckpt_file}' not found. Generating with initialized weights.")

    model.eval()

    frac_coords_list, num_atoms_list, atom_types_list, lattices_list = [], [], [], []
    for batch in tqdm(test_loader, desc="Generating structures"):
        batch = batch.to(dev)
        outputs, _ = model.sample(batch, disable_progress=True)
        frac_coords_list.append(outputs["frac_coords"].detach().cpu())
        num_atoms_list.append(outputs["num_atoms"].detach().cpu())
        atom_types_list.append(outputs["atom_types"].detach().cpu())
        lattices_list.append(outputs["lattices"].detach().cpu())

    frac_coords = torch.cat(frac_coords_list, dim=0)
    num_atoms = torch.cat(num_atoms_list, dim=0)
    atom_types = torch.cat(atom_types_list, dim=0)
    lattices = torch.cat(lattices_list, dim=0)

    preds_list = []
    start_idx = 0
    for n_atoms, lat in zip(num_atoms, lattices):
        cur_frac = frac_coords.narrow(0, start_idx, n_atoms)
        cur_types = atom_types.narrow(0, start_idx, n_atoms)
        preds_list.append(
            Structure(
                lattice=lat,
                species=cur_types,
                coords=cur_frac,
                coords_are_cartesian=False,
            )
        )
        start_idx += n_atoms

    out_file = (
        Path(output_path)
        if output_path
        else wyckoff_path.with_name(
            wyckoff_path.name.replace(".diffcsp-orb", "").replace(".diffcsp-pp", "").split(".")[0]
            + f".diffcsp-{model_type}.json.gz"
        )
    )

    pred_dicts = [s.as_dict() for s in preds_list]
    with gzip.open(out_file, "wt", encoding="ascii") as f:
        json.dump(pred_dicts, f)

    print(f"Successfully generated and wrote {len(preds_list)} structures to {out_file}")
    return out_file


def main() -> None:
    """CLI entrypoint for structure generation."""
    parser = argparse.ArgumentParser(description="DiffCSP++ Inference CLI")
    parser.add_argument("wyckoff_file", type=str, help="Path to input Wyckoff representation file")
    parser.add_argument("--ckpt_path", type=str, default="test_ckpt.pt", help="Path to checkpoint")
    parser.add_argument("--model", type=str, choices=["orb", "cspnet"], default="orb", help="Model backbone")
    parser.add_argument("--orb_model", type=str, default="orb-v2", help="ORB model variant")
    parser.add_argument("--mock_orb", action="store_true", help="Use lightweight mock ORB backbone")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Adapter hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of adapter layers")
    parser.add_argument("--n-structures", type=int, default=1100, help="Number of structures to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, default=None, help="Output file path (.json.gz)")
    args = parser.parse_args()

    generate_structures(
        wyckoff_file=args.wyckoff_file,
        ckpt_path=args.ckpt_path,
        model_type=args.model,
        orb_model=args.orb_model,
        mock_orb=args.mock_orb,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_structures=args.n_structures,
        device=args.device,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
