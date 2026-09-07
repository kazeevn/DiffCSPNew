"""Tests for diffcsp.data.transforms."""

import torch

from diffcsp.data.transforms import (
    cart_forces_to_frac_forces,
    clamp_forces_and_stress,
    frac_to_cart_coords,
    lattice_params_to_matrix_torch,
)


def test_lattice_params_to_matrix_cubic():
    lengths = torch.tensor([[5.0, 5.0, 5.0]])
    angles = torch.tensor([[90.0, 90.0, 90.0]])
    m = lattice_params_to_matrix_torch(lengths, angles)
    assert m.shape == (1, 3, 3)
    # Diagonal elements should be 5.0
    diag = torch.diagonal(m[0])
    assert torch.allclose(diag.abs(), torch.tensor([5.0, 5.0, 5.0]), atol=1e-5)


def test_frac_cart_force_transforms():
    B = 2
    node2graph = torch.tensor([0, 0, 1, 1])
    frac_coords = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    lattices = torch.eye(3).unsqueeze(0).repeat(B, 1, 1) * 10.0  # 10A cubic

    cart_coords = frac_to_cart_coords(frac_coords, lattices, node2graph)
    assert torch.allclose(cart_coords, frac_coords * 10.0)

    cart_forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]])
    frac_forces = cart_forces_to_frac_forces(cart_forces, lattices, node2graph)
    assert torch.allclose(frac_forces, cart_forces * 10.0)


def test_clamp_forces_and_stress():
    forces = torch.randn(10, 3) * 100.0  # extreme forces
    stress = torch.randn(2, 3, 3) * 200.0
    max_f = 15.0
    max_s = 40.0
    c_forces, c_stress = clamp_forces_and_stress(forces, stress, max_force=max_f, max_stress=max_s)

    f_norms = torch.norm(c_forces, dim=-1)
    assert torch.all(f_norms <= max_f + 1e-4)

    s_norms = torch.norm(c_stress, dim=(-2, -1))
    assert torch.all(s_norms <= max_s + 1e-4)
