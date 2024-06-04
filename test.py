from dataset import CrystDataset
from torch_geometric.data import DataLoader
from pymatgen.core import Structure, Lattice
from tqdm import tqdm
import torch
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


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
            lattice=Lattice.from_parameters(*(cur_lengths.tolist() + cur_angles.tolist())),
            species=cur_atom_types,
            coords=cur_frac_coords,
            coords_are_cartesian=False
        )
    )
    start_idx += num_atoms[i]

def process_pair(s1, s2):
    if not structure_validity(s1) or not structure_validity(s2):
        return None
    d = matcher.get_rms_dist(s1, s2)[0]
    if d is None:
        return None
    return d

match_rate = np.array([
    process_pair(s1, s2) for s1, s2 in tqdm(zip(input_list, preds_list))
])
print(np.sum(match_rate != None) / len(match_rate))
