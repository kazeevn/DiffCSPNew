"""Smooth, conservative pairwise repulsive potential for crystal structures.

Provides physical restoring forces F = -Grad(E) and virial Cauchy stress
sigma = (1/V) * sum(r (x) F) when crystal configurations have unphysical
atomic overlaps or collapsed cells (e.g. at high diffusion timesteps).

The potential uses a C^2 smooth polynomial cutoff:
    V(r) = (1/4) * F_max * r_c * (1 - r / r_c)^4   for r < r_c
    V(r) = 0                                       for r >= r_c
The force magnitude is:
    F(r) = -dV/dr = F_max * (1 - r / r_c)^3
which is bounded at r=0 to F_max and smoothly vanishes to 0 at r=r_c
with F(r_c) = 0 and F'(r_c) = 0, ensuring a C^2 smooth transition into
the MLIP regime where all atoms are at physical bonding distances.
"""

from typing import Any
import torch
import torch.nn as nn
from ase.data import covalent_radii

# Standard covalent radii lookup table (Z=0..118)
COVALENT_RADII_TENSOR = torch.tensor(covalent_radii, dtype=torch.float32)

# Periodic neighbor shift offsets (-1, 0, 1) in fractional coordinates
SHIFTS_3D = torch.tensor(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=torch.float32,
)


def compute_crystal_repulsion(
    cart_coords: torch.Tensor,
    lattice: torch.Tensor,
    atom_types: torch.Tensor,
    f_max: float = 2.0,
    eta: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes smooth conservative repulsive forces, virial stress, and energy for a crystal.

    Args:
        cart_coords: (N, 3) Cartesian coordinates of atoms in the unit cell.
        lattice: (3, 3) Lattice matrix (rows are lattice vectors a, b, c).
        atom_types: (N,) Atomic numbers (Z).
        f_max: Maximum repulsive force magnitude at r -> 0 in eV/Angstrom.
        eta: Fraction of sum of covalent radii defining the overlap cutoff r_c = eta * (R_i + R_j).

    Returns:
        tuple of:
            forces: (N, 3) Repulsive forces on each atom in eV/Angstrom (F = -Grad(E)).
            stress: (3, 3) Virial Cauchy stress tensor in GPa or model units.
            energy: (1,) Total repulsive potential energy in eV.
    """
    device = cart_coords.device
    dtype = cart_coords.dtype
    n_atoms = cart_coords.shape[0]

    if n_atoms == 0:
        return (
            torch.zeros_like(cart_coords),
            torch.zeros((3, 3), dtype=dtype, device=device),
            torch.zeros(1, dtype=dtype, device=device),
        )

    cov_radii = COVALENT_RADII_TENSOR.to(device=device, dtype=dtype)
    shifts = SHIFTS_3D.to(device=device, dtype=dtype)

    # Element-dependent cutoff matrix: r_c(i, j) = eta * (R_i + R_j)
    z = atom_types.clamp(1, len(covalent_radii) - 1)
    r_cov = cov_radii[z]
    r_cut_matrix = eta * (r_cov.unsqueeze(1) + r_cov.unsqueeze(0))  # (N, N)

    # Transform coordinates to fractional
    inv_lattice = torch.linalg.pinv(lattice)
    frac_coords = (cart_coords @ inv_lattice) % 1.0

    vol = torch.abs(torch.linalg.det(lattice)).clamp(min=10.0)
    forces = torch.zeros_like(cart_coords)
    stress = torch.zeros((3, 3), dtype=dtype, device=device)
    total_energy = torch.zeros(1, dtype=dtype, device=device)

    for shift in shifts:
        # Cartesian displacement vector r_ij = s_ij @ lattice
        s_diff = frac_coords.unsqueeze(1) - (frac_coords.unsqueeze(0) + shift)
        r_vec = s_diff @ lattice  # (N, N, 3)
        dist = torch.norm(r_vec, dim=-1)  # (N, N)

        mask = dist < r_cut_matrix
        # Ignore self-interaction at zero periodic shift
        if torch.all(shift == 0):
            mask = mask & ~torch.eye(n_atoms, dtype=torch.bool, device=device)

        if not mask.any():
            continue

        rc = r_cut_matrix[mask]
        d = dist[mask].clamp(min=1e-4)
        vec = r_vec[mask]

        # C^2 smooth polynomial potential and force:
        # V(r) = (1/4) * f_max * rc * (1 - d / rc)^4
        # F_mag(r) = f_max * (1 - d / rc)^3
        ratio = 1.0 - d / rc
        f_mag = f_max * (ratio**3)
        v_pair = 0.25 * f_max * rc * (ratio**4)

        unit_vec = vec / d.unsqueeze(-1)
        f_pair = unit_vec * f_mag.unsqueeze(-1)  # (M, 3)

        total_energy = total_energy + 0.5 * v_pair.sum()

        row_indices = torch.arange(n_atoms, device=device).unsqueeze(1).expand(n_atoms, n_atoms)[mask]
        forces.index_add_(0, row_indices, f_pair)

        # Virial stress: sigma_ab = (1/V) * sum_pairs (r_a * F_b)
        virial = f_pair.unsqueeze(2) @ vec.unsqueeze(1)
        stress = stress + 0.5 * virial.sum(dim=0) / vol

    return forces, stress, total_energy


class SmoothRepulsivePotential(nn.Module):
    """Module wrapper for smooth conservative repulsive potential."""

    def __init__(self, f_max: float = 2.0, eta: float = 0.70) -> None:
        super().__init__()
        self.f_max = f_max
        self.eta = eta

    def forward(
        self,
        cart_coords: torch.Tensor,
        lattices: torch.Tensor,
        atom_types: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluates repulsive forces and stress across a batch of crystal structures."""
        batch_size = lattices.shape[0]
        device = cart_coords.device
        dtype = cart_coords.dtype

        all_forces = []
        all_stresses = []
        all_energies = []
        start_idx = 0

        for b in range(batch_size):
            n_atoms = int(num_atoms[b].item())
            sub_coords = cart_coords[start_idx : start_idx + n_atoms]
            sub_lat = lattices[b]
            sub_types = atom_types[start_idx : start_idx + n_atoms]

            f, s, e = compute_crystal_repulsion(
                cart_coords=sub_coords,
                lattice=sub_lat,
                atom_types=sub_types,
                f_max=self.f_max,
                eta=self.eta,
            )
            all_forces.append(f)
            all_stresses.append(s.unsqueeze(0))
            all_energies.append(e.unsqueeze(0) if e.dim() == 0 else e.view(1, 1))
            start_idx += n_atoms

        return {
            "forces": torch.cat(all_forces, dim=0) if all_forces else torch.zeros_like(cart_coords),
            "stress": torch.cat(all_stresses, dim=0) if all_stresses else torch.zeros((batch_size, 3, 3), dtype=dtype, device=device),
            "energy": torch.cat(all_energies, dim=0) if all_energies else torch.zeros((batch_size, 1), dtype=dtype, device=device),
        }


def batched_crystal_repulsion(
    cart_coords: torch.Tensor,
    lattices: torch.Tensor,
    atom_types: torch.Tensor,
    num_atoms: torch.Tensor,
    f_max: float = 2.0,
    eta: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched equivalent of :func:`compute_crystal_repulsion` over a whole batch.

    Vectorized across structures and atom pairs; the only Python loop is over
    the 27 fixed periodic shifts, so the number of kernel launches is constant
    in the batch size. Structures are zero-padded to the largest atom count and
    masked, and the dense mask is applied multiplicatively rather than by
    boolean indexing so that no host synchronization is needed.

    Unlike the MLIP path this needs no neighbour list, so it stays valid for the
    collapsed cells that occur at high diffusion noise.

    Args:
        cart_coords: (N, 3) Cartesian coordinates, concatenated over the batch.
        lattices: (B, 3, 3) lattice matrices (rows are lattice vectors).
        atom_types: (N,) atomic numbers.
        num_atoms: (B,) atom count per structure, summing to N.
        f_max: Maximum repulsive force magnitude at r -> 0 in eV/Angstrom.
        eta: Fraction of the sum of covalent radii defining r_c = eta * (R_i + R_j).

    Returns:
        tuple of forces (N, 3), Cauchy stress (B, 3, 3) and energy (B, 1).
    """
    device = cart_coords.device
    dtype = cart_coords.dtype
    batch_size = lattices.shape[0]

    if batch_size == 0 or cart_coords.shape[0] == 0:
        return (
            torch.zeros_like(cart_coords),
            torch.zeros((batch_size, 3, 3), dtype=dtype, device=device),
            torch.zeros((batch_size, 1), dtype=dtype, device=device),
        )

    max_n = int(num_atoms.max().item())
    positions = torch.arange(max_n, device=device)
    valid = positions.unsqueeze(0) < num_atoms.unsqueeze(1)  # (B, max_n)

    padded_coords = torch.zeros((batch_size, max_n, 3), dtype=dtype, device=device)
    padded_coords[valid] = cart_coords
    padded_types = torch.zeros((batch_size, max_n), dtype=torch.long, device=device)
    padded_types[valid] = atom_types

    cov_radii = COVALENT_RADII_TENSOR.to(device=device, dtype=dtype)
    shifts = SHIFTS_3D.to(device=device, dtype=dtype)

    z = padded_types.clamp(1, len(covalent_radii) - 1)
    r_cov = cov_radii[z]
    r_cut = eta * (r_cov.unsqueeze(2) + r_cov.unsqueeze(1))  # (B, max_n, max_n)

    inv_lattice = torch.linalg.pinv(lattices)
    frac_coords = torch.einsum("bnk,bkl->bnl", padded_coords, inv_lattice) % 1.0

    vol = torch.abs(torch.linalg.det(lattices)).clamp(min=10.0)  # (B,)

    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    not_self = ~torch.eye(max_n, dtype=torch.bool, device=device).unsqueeze(0)

    forces = torch.zeros((batch_size, max_n, 3), dtype=dtype, device=device)
    stress = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    energy = torch.zeros(batch_size, dtype=dtype, device=device)

    for shift_idx in range(shifts.shape[0]):
        shift = shifts[shift_idx]
        s_diff = frac_coords.unsqueeze(2) - (frac_coords.unsqueeze(1) + shift)
        r_vec = torch.einsum("bijk,bkl->bijl", s_diff, lattices)  # (B, max_n, max_n, 3)
        dist = torch.norm(r_vec, dim=-1)

        mask = (dist < r_cut) & pair_valid
        if bool((shift == 0).all()):
            mask = mask & not_self
        weight = mask.to(dtype)

        d = dist.clamp(min=1e-4)
        ratio = 1.0 - d / r_cut
        f_mag = f_max * (ratio**3) * weight
        v_pair = 0.25 * f_max * r_cut * (ratio**4) * weight

        f_pair = (r_vec / d.unsqueeze(-1)) * f_mag.unsqueeze(-1)

        forces = forces + f_pair.sum(dim=2)
        energy = energy + 0.5 * v_pair.sum(dim=(1, 2))
        # Virial stress: sigma_ab = (1/V) * sum_pairs (F_a * r_b)
        stress = stress + 0.5 * torch.einsum("bija,bijc->bac", f_pair, r_vec) / vol.view(-1, 1, 1)

    return forces[valid], stress, energy.unsqueeze(-1)
