import json
from pathlib import Path
import os
from torch.utils.data import Dataset
import gzip
import logging
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import trange
import torch
from pyxtal import pyxtal
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from scipy.linalg import pinv
import numpy as np

logger = logging.getLogger(__file__)

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False
)
crystalNN_tmp = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=10
)


def build_crystal_graph(crystal, graph_method='crystalnn'):
    c = pyxtal()
    c.from_random(**crystal, max_count=30)
    space_group = c.group.number
    crystal = c.to_pymatgen(resort=False)

    if graph_method == 'crystalnn':
        try:
            crystal_graph = StructureGraph.from_local_env_strategy(crystal, CrystalNN)
        except Exception as _:
            crystal_graph = StructureGraph.from_local_env_strategy(crystal, crystalNN_tmp)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    operation = []
    inv_rotation = []
    anchor_idxs = []
    for site in c.atom_sites:
        anchor_idxs.extend([len(operation) for _ in site.wp.ops])
        operation.extend([op.affine_matrix for op in site.wp.ops])
        inv_rotation.extend([pinv(op.rotation_matrix) for op in site.wp.ops])

    if len(operation) != len(crystal.frac_coords):
        print('rot', len(operation), 'frac', len(crystal.frac_coords))
        raise NotImplementedError

    operation = np.stack(operation)
    inv_rotation = np.stack(inv_rotation)
    anchor_idxs = np.array(anchor_idxs)

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, \
           num_atoms, operation, inv_rotation, anchor_idxs, space_group


def process_one(row, niggli, primitive, graph_method):
    result_dict = {}
    try:
        graph_arrays = build_crystal_graph(row, graph_method)
        result_dict.update({
            'graph_arrays': graph_arrays
        })
        return result_dict
    except (RuntimeError, TypeError) as e:
        logger.warning("Error processing %s: %s", row, e)
        return None


def preprocess(
    input_file: Path,
    niggli,
    primitive,
    graph_method,
    structure_count:int = 1000):
    if input_file.suffix == '.json':
        with open(input_file, 'rt', encoding="ascii") as f:
            data = json.load(f)
    elif input_file.suffixes == ['.json', '.gz']:
        with gzip.open(input_file, 'rt', encoding="ascii") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unknown file type: {input_file}")
    results = Parallel(n_jobs=-1)(delayed(process_one)(
        data[idx],
        niggli,
        primitive,
        graph_method,
    ) for idx in trange(structure_count))
    return [r for r in results if r is not None]


class TransformerDataset(Dataset):
    def __init__(self, path, mode, structure_count=1000):
        super().__init__()
        self.path = Path(path)
        self.cache_path = self.path.parent / 'cache' / self.path.stem / f"{mode}.pth"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.niggli = True
        self.primitive = False
        self.graph_method = 'crystalnn'
        self.structure_count = structure_count
        self.preprocess(self.cache_path)

    def preprocess(self, save_path):
        print(f'Preprocessing {self.path}')
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path, weights_only=False)
        else:
            cached_data = preprocess(
                self.path,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                structure_count=self.structure_count
            )
            torch.save(cached_data, save_path)
            self.cached_data = cached_data
        print('Done')

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms, operation, inv_rotation, anchor_idxs, space_group) = data_dict['graph_arrays']

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            ops=torch.Tensor(operation),
            ops_inv=torch.Tensor(inv_rotation),
            anchor_index=torch.LongTensor(anchor_idxs),
            spacegroup=space_group,
        )

        return data
