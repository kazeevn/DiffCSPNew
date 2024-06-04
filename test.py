from dataset import CrystDataset
from torch_geometric.data import DataLoader
from pymatgen.core import Structure, Lattice
from tqdm import tqdm
import torch
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher


matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

testset = CrystDataset('test.csv', 'test_sym')
test_batch_size = 256
test_loader = DataLoader(testset, shuffle=False, batch_size=test_batch_size)

input_data_list = []
for batch in tqdm(test_loader):
    input_data_list = input_data_list + batch.to_data_list()

input_list = []
for struct in input_data_list:
    input_list.append(
        Structure(
            lattice=Lattice.from_parameters(*(struct.lengths.tolist()[0] + struct.angles.tolist()[0])),
            species=struct.atom_types.to('cpu'),
            coords=struct.frac_coords.to('cpu'),
            coords_are_cartesian=False
        )
    )

data = torch.load("eval_diff.pt", map_location='cpu')

frac_coords = data['frac_coords'][0]
atom_types = data['atom_types'][0]
lengths = data['lengths'][0]
angles = data['angles'][0]
num_atoms = data['num_atoms'][0]

preds_list = []
start_idx = 0
for i in tqdm(range(len(num_atoms))):
    cur_frac_coords = frac_coords.narrow(0, start_idx, num_atoms[i])
    cur_atom_types = atom_types.narrow(0, start_idx, num_atoms[i])
    cur_lengths = lengths[i]
    cur_angles = angles[i]
    preds_list.append(
        Structure(
            lattice=Lattice.from_parameters(*(cur_lengths.tolist()[0] + cur_angles.tolist()[0])),
            species=cur_atom_types,
            coords=cur_frac_coords,
            coords_are_cartesian=False
        )
    )
    start_idx += num_atoms[i]

match_rate = np.array([
    d[0] if (d := matcher.get_rms_dist(s1, s2)) is not None else None for s1, s2 in tqdm(zip(input_list, preds_list))
])
print(np.sum(match_rate != None) / len(match_rate))
