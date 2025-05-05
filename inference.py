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

from diffusion1 import CSPDiffusion
from TransformerDataset import TransformerDataset

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Produce structures from pyXtal Wyckoff representations")
    parser.add_argument("wyckoff_file", type=Path, help="Path to the Wyckoff file")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--device", type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-structures", type=int, default=1100, help="Number of structures to produce")
    args = parser.parse_args()

    set_random_seed(args.seed)

    testset = TransformerDataset(args.wyckoff_file, 'transformer', structure_count=args.n_structures)
    print(f"Number of structures: {len(testset)}")

    test_loader = DataLoader(testset, shuffle=False, batch_size=args.batch_size)

    model = CSPDiffusion(args.device).to(args.device)
    model = torch.compile(model, fullgraph=True)
    model.load_state_dict(torch.load('test_ckpt.pt', weights_only=True))
    model.eval()

    frac_coords, num_atoms, atom_types, lattices, input_data_list = [], [], [], [], []
    for batch in tqdm(test_loader):
        batch = batch.to(args.device)
        outputs, _ = model.sample(batch)

        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())
        input_data_list = input_data_list + batch.to_data_list()

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
            Structure(lattice=this_lattice, species=cur_atom_types,
                    coords=cur_frac_coords, coords_are_cartesian=False)
        )
        start_idx += this_num_atoms

    pred_list = [s.as_dict() for s in preds_list]
    output_file_name = args.wyckoff_file
    while output_file_name.suffix:
        output_file_name = output_file_name.with_suffix('')
    output_file_name = output_file_name.with_suffix('.diffcsp-pp.json.gz')
    with gzip.open(output_file_name, 'wt', encoding="ascii") as f:
        json.dump(pred_list, f)
    print(f"Wrote structures to {str(output_file_name)}")


if __name__ == '__main__':
    main()
