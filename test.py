from dataset import CrystDataset
from torch_geometric.data import DataLoader
from pymatgen.core import Structure, Lattice
from tqdm import tqdm
import torch
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
import smact
import itertools
from smact.screening import pauling_test
from collections import Counter
from pymatgen.core.composition import Composition
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
CompFP = ElementProperty.from_preset('magpie')
CrystalNNFP = CrystalNNFingerprint.from_preset("ops")

import sys
sys.path.append('.')
from diffcsp.common.data_utils import chemical_symbols


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()


    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


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
        Crystal({
            'frac_coords': cur_frac_coords,
            'atom_types': cur_atom_types,
            'lengths': cur_lengths,
            'angles': cur_angles,
        })
    )
    start_idx += num_atoms[i]

# frac_coords = data['input_data_batch']['frac_coords']
# atom_types = data['input_data_batch']['atom_types']
# lengths = data['input_data_batch']['lengths']
# angles = data['input_data_batch']['angles']
# num_atoms = data['input_data_batch']['num_atoms']
#
# old_list = []
# start_idx = 0
# for i in tqdm(range(len(num_atoms))):
#     cur_frac_coords = frac_coords.narrow(0, start_idx, num_atoms[i])
#     cur_atom_types = atom_types.narrow(0, start_idx, num_atoms[i])
#     cur_lengths = lengths[i]
#     cur_angles = angles[i]
#     old_list.append(
#         Structure(
#             lattice=Lattice.from_parameters(*(cur_lengths.tolist() + cur_angles.tolist())),
#             species=cur_atom_types,
#             coords=cur_frac_coords,
#             coords_are_cartesian=False
#         )
#     )
#     start_idx += num_atoms[i]


def process_pair(s1, s2):
    if not structure_validity(s1) or not structure_validity(s2):
        return None
    d = matcher.get_rms_dist(s1, s2)
    if d is None:
        return None
    return d[0]


match_rate = np.array([
    process_pair(s1, s2.structure) for s1, s2 in tqdm(zip(input_list, preds_list))
])
print(np.sum(match_rate != None) / len(match_rate))

# match_rate = np.array([
#     process_pair(s1, s2) for s1, s2 in tqdm(zip(old_list, preds_list))
# ])
# print(np.sum(match_rate != None) / len(match_rate))
