import torch
import pandas as pd
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data

from data_utils import preprocess


class CrystDataset(Dataset):
    def __init__(self, path, mode):
        super().__init__()
        self.path = path
        self.df = pd.read_csv(self.path)
        self.niggli = True
        self.primitive = False
        self.graph_method = 'crystalnn'

        self.preprocess(f"{mode}.pth")

    def preprocess(self, save_path):
        print(f'preprocessing {self.path}')
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
                self.path,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
            )
            torch.save(cached_data, save_path)
            self.cached_data = cached_data
        print('done')

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        if 'spacegroup' not in data_dict:
            (frac_coords, atom_types, lengths, angles, edge_indices,
             to_jimages, num_atoms, operation, inv_rotation, anchor_idxs, space_group) = data_dict['graph_arrays']
        else:
            (frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms) = data_dict['graph_arrays']
            space_group = torch.LongTensor([data_dict['spacegroup']])
            operation = torch.Tensor(data_dict['wyckoff_ops'])
            anchor_idxs = torch.LongTensor(data_dict['anchors'])
            inv_rotation = torch.linalg.pinv(operation[:, :3, :3])

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
