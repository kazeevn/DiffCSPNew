"""Lattice, coordinate, and force transformations for crystal structures.

Handles conversions between lattice parameters (lengths, angles) and
3x3 lattice matrices, Cartesian and fractional coordinates, and Cartesian
and fractional force vectors with numerical safety clamping.
"""

import torch


def lattice_params_to_matrix_torch(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Batched computation of 3x3 lattice matrix from lattice parameters.

    Args:
        lengths: (N, 3) tensor of lattice vector lengths (a, b, c) in Angstroms.
        angles: (N, 3) tensor of lattice angles (alpha, beta, gamma) in degrees.

    Returns:
        (N, 3, 3) tensor of lattice row-vectors [a, b, c].
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1] + 1e-8)
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    zeros = torch.zeros(lengths.size(0), device=lengths.device, dtype=lengths.dtype)
    vector_a = torch.stack([lengths[:, 0] * sins[:, 1], zeros, lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack([zeros, zeros, lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def frac_to_cart_coords(
    frac_coords: torch.Tensor, lattices: torch.Tensor, node2graph: torch.Tensor
) -> torch.Tensor:
    """Converts fractional coordinates to Cartesian coordinates: R = x @ L.

    Args:
        frac_coords: (N, 3) fractional coordinates in [0, 1).
        lattices: (B, 3, 3) lattice matrices where rows are [a, b, c].
        node2graph: (N,) mapping each atom to its graph/batch index.

    Returns:
        (N, 3) Cartesian coordinates in Angstroms.
    """
    lattices_per_atom = lattices[node2graph]  # (N, 3, 3)
    return torch.einsum("ni, nij -> nj", frac_coords, lattices_per_atom)


def cart_forces_to_frac_forces(
    cart_forces: torch.Tensor, lattices: torch.Tensor, node2graph: torch.Tensor
) -> torch.Tensor:
    """Converts Cartesian forces to fractional forces: F_frac = F_cart @ L.

    Since R = x @ L, dR/dx = L, so F_frac = -dE/dx = -(dE/dR) @ (dR/dx) = F_cart @ L.

    Args:
        cart_forces: (N, 3) Cartesian forces (eV/A).
        lattices: (B, 3, 3) lattice matrices.
        node2graph: (N,) mapping each atom to its graph/batch index.

    Returns:
        (N, 3) forces in fractional coordinate space.
    """
    lattices_per_atom = lattices[node2graph]
    return torch.einsum("ni, nij -> nj", cart_forces, lattices_per_atom)


def clamp_forces_and_stress(
    cart_forces: torch.Tensor, stress: torch.Tensor, max_force: float = 20.0, max_stress: float = 50.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smoothly clamps forces and stresses to prevent divergence during high-noise steps.

    Uses smooth tanh saturation preserving vector orientation.

    Args:
        cart_forces: (N, 3) Cartesian forces.
        stress: (B, 3, 3) Cauchy stress tensors.
        max_force: force norm threshold.
        max_stress: stress norm threshold.

    Returns:
        Tuple of clamped forces (N, 3) and clamped stress (B, 3, 3).
    """
    f_norm = torch.norm(cart_forces, dim=-1, keepdim=True) + 1e-8
    clamped_forces = cart_forces * (max_force * torch.tanh(f_norm / max_force) / f_norm)

    s_norm = torch.norm(stress, dim=(-2, -1), keepdim=True) + 1e-8
    clamped_stress = stress * (max_stress * torch.tanh(s_norm / max_stress) / s_norm)

    return clamped_forces, clamped_stress
