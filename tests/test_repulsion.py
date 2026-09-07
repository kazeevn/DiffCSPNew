"""Unit tests for smooth conservative repulsive potential."""

import pytest
import torch

from diffcsp.models.repulsion import SmoothRepulsivePotential, compute_crystal_repulsion


def test_repulsion_conservative_force():
    """Verify that repulsive forces are strictly conservative: F = -dE/dr."""
    # Cubic cell 5x5x5 Angstroms
    lattice = 5.0 * torch.eye(3, dtype=torch.float64)
    # Two Silicon atoms with small interatomic separation (0.8 Angstroms < r_c ~ 1.55 Angstroms)
    cart_coords = torch.tensor(
        [[2.0, 2.5, 2.5], [2.8, 2.5, 2.5]], dtype=torch.float64, requires_grad=True
    )
    atom_types = torch.tensor([14, 14], dtype=torch.long)

    forces, stress, energy = compute_crystal_repulsion(
        cart_coords=cart_coords,
        lattice=lattice,
        atom_types=atom_types,
        f_max=20.0,
        eta=0.70,
    )

    # Compute autograd forces: F_auto = - d(energy)/d(coords)
    grad_coords = torch.autograd.grad(energy, cart_coords)[0]
    expected_forces = -grad_coords

    assert torch.allclose(forces, expected_forces, atol=1e-5), (
        f"Analytic forces do not match autograd conservative forces: diff={torch.norm(forces - expected_forces)}"
    )
    # Action-reaction: sum of internal forces in periodic cell should be zero
    assert torch.allclose(forces.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-5)


def test_repulsion_smooth_cutoff():
    """Verify that potential, forces, and stress vanish smoothly when r >= r_c."""
    lattice = 10.0 * torch.eye(3, dtype=torch.float32)
    # Two Carbon atoms (covalent radius = 0.76 A, r_c = 0.70 * (0.76 + 0.76) = 1.064 A)
    # Separation = 1.5 A > r_c -> no overlap
    cart_coords = torch.tensor([[1.0, 5.0, 5.0], [2.5, 5.0, 5.0]], dtype=torch.float32)
    atom_types = torch.tensor([6, 6], dtype=torch.long)

    forces, stress, energy = compute_crystal_repulsion(
        cart_coords=cart_coords,
        lattice=lattice,
        atom_types=atom_types,
        f_max=20.0,
        eta=0.70,
    )

    assert torch.allclose(energy, torch.zeros(1, dtype=torch.float32))
    assert torch.allclose(forces, torch.zeros_like(cart_coords))
    assert torch.allclose(stress, torch.zeros(3, 3, dtype=torch.float32))


def test_repulsion_batch_module():
    """Verify SmoothRepulsivePotential batch wrapper."""
    module = SmoothRepulsivePotential(f_max=20.0, eta=0.70)

    # Batch of 2 crystals: 1st with 2 atoms overlapping, 2nd with 3 atoms separated
    lattices = torch.stack([5.0 * torch.eye(3), 6.0 * torch.eye(3)])
    cart_coords = torch.tensor(
        [
            # Crystal 1 (overlapping)
            [1.0, 2.0, 2.0],
            [1.5, 2.0, 2.0],
            # Crystal 2 (non-overlapping)
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    atom_types = torch.tensor([14, 14, 8, 8, 8], dtype=torch.long)
    num_atoms = torch.tensor([2, 3], dtype=torch.long)

    out = module(cart_coords, lattices, atom_types, num_atoms)

    assert out["forces"].shape == (5, 3)
    assert out["stress"].shape == (2, 3, 3)
    assert out["energy"].shape == (2, 1)

    # Crystal 1 has overlap -> non-zero force and stress
    assert torch.norm(out["forces"][:2]) > 0.1
    assert torch.norm(out["stress"][0]) > 0.01
    assert out["energy"][0].item() > 0.0

    # Crystal 2 has no overlap -> zero force and stress
    assert torch.allclose(out["forces"][2:], torch.zeros(3, 3))
    assert torch.allclose(out["stress"][1], torch.zeros(3, 3))
    assert torch.allclose(out["energy"][1], torch.zeros(1))
