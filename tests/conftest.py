"""Pytest configuration and shared test fixtures."""

import pytest
import torch
from torch_geometric.data import Batch, Data


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def synthetic_crystal_data():
    """Generates a synthetic crystal Data object compatible with DiffCSP++."""
    num_atoms = 4
    frac_coords = torch.rand(num_atoms, 3)
    atom_types = torch.tensor([14, 14, 8, 8], dtype=torch.long)  # e.g., SiO2-like
    lengths = torch.tensor([[4.91, 4.91, 5.40]], dtype=torch.float32)
    angles = torch.tensor([[90.0, 90.0, 120.0]], dtype=torch.float32)
    spacegroup = torch.tensor([152], dtype=torch.long)
    anchor_index = torch.arange(num_atoms, dtype=torch.long)

    # Identity affine operations
    ops = torch.eye(4).unsqueeze(0).repeat(num_atoms, 1, 1)
    ops_inv = torch.eye(3).unsqueeze(0).repeat(num_atoms, 1, 1)

    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    to_jimages = torch.zeros((6, 3), dtype=torch.long)

    return Data(
        frac_coords=frac_coords,
        atom_types=atom_types,
        lengths=lengths,
        angles=angles,
        spacegroup=spacegroup,
        anchor_index=anchor_index,
        ops=ops,
        ops_inv=ops_inv,
        edge_index=edge_index,
        to_jimages=to_jimages,
        num_atoms=num_atoms,
        num_bonds=6,
        num_nodes=num_atoms,
    )


@pytest.fixture
def synthetic_batch(synthetic_crystal_data):
    """Generates a synthetic PyG Batch containing 2 crystals."""
    d1 = synthetic_crystal_data.clone()
    d2 = synthetic_crystal_data.clone()
    batch = Batch.from_data_list([d1, d2])
    batch.batch_size = 2
    batch.num_atoms = torch.tensor([4, 4], dtype=torch.long)
    batch.spacegroup = torch.tensor([152, 152], dtype=torch.long)
    return batch
