import torch
import pandas as pd
from torch_geometric.data import DataLoader
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from diffusion import CSPDiffusion
from dataset import CrystDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CSPDiffusion(device)
model.load_state_dict(torch.load('test_ckpt.pt'))
model.to(device)

with open('CaTiO3.cif') as f:
    struct_cif = f.read()

d = {'cif': [struct_cif for _ in range(10000)], 'material_id': ['mp-4019' for _ in range(10000)]}
df = pd.DataFrame(data=d)

df.to_csv('predict.csv')

testset = CrystDataset('predict.csv', 'predict')

test_batch_size = 200

test_loader = DataLoader(testset, shuffle=False, batch_size=test_batch_size)


model.train(False)
with torch.no_grad():
    frac_coords, num_atoms, atom_types, lattices, input_data_list = [], [], [], [], []
    for batch in tqdm(test_loader):
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

    for struct in tqdm(preds_list):
        CifWriter(struct).write_file('test_pred.cif', 'a')
