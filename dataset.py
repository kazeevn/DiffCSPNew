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

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms, rotation, inv_rotation, translation, wp_len, space_group) = data_dict['graph_arrays']

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
            rotation=torch.Tensor(rotation),
            inv_rotation=torch.Tensor(inv_rotation),
            translation=torch.Tensor(translation),
            wp_len=torch.LongTensor(wp_len),
            spacegroup=space_group,
        )

        return data
