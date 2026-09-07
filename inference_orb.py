import argparse
import random
import gzip
from pathlib import Path
import json
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from pymatgen.core import Structure

from diffusion_orb import CSPDiffusionORB
from TransformerDataset import TransformerDataset


def set_random_seed(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Generate crystal structures using DiffCSP++ with Frozen ORB MLIP")
    parser.add_argument("wyckoff_file", type=Path, help="Path to the Wyckoff representation file")
    parser.add_argument("--ckpt_path", type=str, default="orb_diffcsp_ckpt.pt", help="Path to adapter checkpoint")
    parser.add_argument("--orb_model", type=str, default="orb-v2", help="ORB model variant")
    parser.add_argument("--mock_orb", action="store_true", help="Use mock ORB backbone for testing")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Adapter hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of adapter layers")
    parser.add_argument("--no_zero_force_condition", action="store_true", help="Disable by-design S=0 ==> F=0 condition")
    parser.add_argument("--gamma_min", type=float, default=1e-3, help="Minimum along-force scaling factor")
    parser.add_argument("--device", type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-structures", type=int, default=1100, help="Number of structures to produce")
    args = parser.parse_args()

    set_random_seed(args.seed)

    testset = TransformerDataset(args.wyckoff_file, 'transformer', structure_count=args.n_structures)
    print(f"Number of structures to generate: {len(testset)}")

    test_loader = DataLoader(testset, shuffle=False, batch_size=args.batch_size)

    # Initialize CSPDiffusionORB
    model = CSPDiffusionORB(
        device=args.device,
        orb_model_name=args.orb_model,
        use_mock_orb=args.mock_orb,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        enforce_zero_force_condition=not args.no_zero_force_condition,
        gamma_min=args.gamma_min
    ).to(args.device)

    # Load adapter checkpoint if available
    if Path(args.ckpt_path).exists():
        state_dict = torch.load(args.ckpt_path, map_location=args.device, weights_only=True)
        # Supports loading either full model or decoder state dict
        if "decoder.coord_out.weight" in state_dict:
            model.load_state_dict(state_dict, strict=False)
        else:
            model.decoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded adapter weights from {args.ckpt_path}")
    else:
        print(f"Warning: Checkpoint {args.ckpt_path} not found. Running with un-finetuned adapter.")

    model.eval()

    frac_coords, num_atoms, atom_types, lattices = [], [], [], []
    for batch in tqdm(test_loader, desc="Generating"):
        batch = batch.to(args.device)
        outputs, _ = model.sample(batch)

        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)

    preds_list = []
    start_idx = 0
    for this_num_atoms, this_lattice in zip(num_atoms, lattices):
        cur_frac_coords = frac_coords.narrow(0, start_idx, this_num_atoms)
        cur_atom_types = atom_types.narrow(0, start_idx, this_num_atoms)
        preds_list.append(
            Structure(
                lattice=this_lattice,
                species=cur_atom_types,
                coords=cur_frac_coords,
                coords_are_cartesian=False
            )
        )
        start_idx += this_num_atoms

    pred_list = [s.as_dict() for s in preds_list]
    output_file_name = args.wyckoff_file
    while output_file_name.suffix:
        output_file_name = output_file_name.with_suffix('')
    output_file_name = output_file_name.with_suffix('.diffcsp-orb.json.gz')
    with gzip.open(output_file_name, 'wt', encoding="ascii") as f:
        json.dump(pred_list, f)
    print(f"Wrote generated structures to {str(output_file_name)}")


if __name__ == '__main__':
    main()
