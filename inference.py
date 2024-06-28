import json
from TransformerDataset import TransformerDataset
from torch_geometric.data import DataLoader
from diffusion1 import CSPDiffusion
import torch
from tqdm import tqdm
import numpy as np
import random
from pymatgen.core import Structure


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(17)

testset = TransformerDataset('WyckoffTransformer_mp_20.json', 'transformer')

test_batch_size = 256

test_loader = DataLoader(testset, shuffle=False, batch_size=test_batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CSPDiffusion(device).to(device)
model.load_state_dict(torch.load('test_ckpt.pt'))

model.train(False)
with torch.no_grad():
    frac_coords, num_atoms, atom_types, lattices, input_data_list = [], [], [], [], []
    for idx, batch in tqdm(enumerate(test_loader)):
        batch = batch.to(device)
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
    for i in range(len(num_atoms)):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atoms[i])
        cur_atom_types = atom_types.narrow(0, start_idx, num_atoms[i])
        preds_list.append(
            Structure(lattice=lattices[i], species=cur_atom_types, coords=cur_frac_coords, coords_are_cartesian=False)
        )
        start_idx += num_atoms[i]

    with open('result.json') as f:
        pred_list = [s.as_dict() for s in preds_list]
        json.dump(pred_list, f)
