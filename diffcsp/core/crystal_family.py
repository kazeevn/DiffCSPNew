"""Symmetry constraints and Lie algebra representations for crystal families.

Projects 3x3 lattice matrices to 6-dimensional vectors on the Lie algebra,
applying exact crystal family constraints based on space group symmetries.
"""

import numpy as np
import torch
import torch.nn as nn

from diffcsp.core.matrix import expm, logm, sqrtm


class CrystalFamily(nn.Module):
    """Encodes and enforces crystal family constraints on lattice tensors."""

    basis: torch.Tensor
    masks: torch.Tensor
    biases: torch.Tensor
    family: torch.Tensor

    def __init__(self) -> None:
        super().__init__()

        basis = self.get_basis()
        masks, biases = self.get_spacegroup_constraints()
        family = self.get_family_idx()

        self.register_buffer("basis", basis)
        self.register_buffer("masks", masks)
        self.register_buffer("biases", biases)
        self.register_buffer("family", family)

    @staticmethod
    def get_basis() -> torch.Tensor:
        """Constructs and normalizes the 6 basis matrices spanning symmetric 3x3 lattices."""
        basis = torch.tensor(
            [
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )

        # Normalize basis tensors
        norms = basis.norm(dim=(-1, -2), keepdim=True)
        return basis / norms

    @staticmethod
    def get_spacegroup_constraint(spacegroup: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the mask and bias vector for a specific spacegroup (1 to 230)."""
        mask = torch.ones(6, dtype=torch.float32)
        bias = torch.zeros(6, dtype=torch.float32)

        if 195 <= spacegroup <= 230:
            # Cubic
            mask[[0, 1, 2, 3, 4]] = 0.0
        elif 143 <= spacegroup <= 194:
            # Hexagonal / Trigonal
            mask[[0, 1, 2, 3]] = 0.0
            bias[0] = -0.25 * float(np.log(3) * np.sqrt(2))
        elif 75 <= spacegroup <= 142:
            # Tetragonal
            mask[[0, 1, 2, 3]] = 0.0
        elif 16 <= spacegroup <= 74:
            # Orthorhombic
            mask[[0, 1, 2]] = 0.0
        elif 3 <= spacegroup <= 15:
            # Monoclinic
            mask[[0, 2]] = 0.0
        elif 0 <= spacegroup <= 2:
            # Triclinic (unconstrained)
            pass

        return mask, bias

    def get_spacegroup_constraints(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Precomputes constraints for all 230 spacegroups + index 0."""
        masks, biases = [], []
        for sg in range(231):
            m, b = self.get_spacegroup_constraint(sg)
            masks.append(m.unsqueeze(0))
            biases.append(b.unsqueeze(0))
        return torch.cat(masks, dim=0), torch.cat(biases, dim=0)

    @staticmethod
    def get_family_idx() -> torch.Tensor:
        """Maps each space group 0..230 to its crystal family integer code (1..6)."""
        family = []
        for sg in range(231):
            if 195 <= sg <= 230:
                family.append(6)
            elif 143 <= sg <= 194:
                family.append(5)
            elif 75 <= sg <= 142:
                family.append(4)
            elif 16 <= sg <= 74:
                family.append(3)
            elif 3 <= sg <= 15:
                family.append(2)
            else:
                family.append(1)
        return torch.tensor(family, dtype=torch.long)

    def de_so3(self, lattice_mat: torch.Tensor) -> torch.Tensor:
        """Removes global rotation by computing symmetric square root of L @ L^T.

        Args:
            lattice_mat: (B, 3, 3) batch of lattice matrices.

        Returns:
            (B, 3, 3) canonical symmetric lattice matrix.
        """
        metric = lattice_mat @ lattice_mat.transpose(-1, -2)
        return sqrtm(metric)

    def v2m(self, vec: torch.Tensor) -> torch.Tensor:
        """Transforms Lie algebra vector representation back to 3x3 lattice matrix.

        Args:
            vec: (B, 6) or (B, 5) vector.

        Returns:
            (B, 3, 3) lattice matrix.
        """
        dims = vec.shape[-1]
        basis = self.basis if dims == 6 else self.basis[:-1]
        log_mat = torch.einsum("bk, kij -> bij", vec, basis)
        return expm(log_mat)

    def m2v(self, mat: torch.Tensor) -> torch.Tensor:
        """Transforms 3x3 lattice matrix into 6-dimensional Lie algebra vector.

        Args:
            mat: (B, 3, 3) lattice matrix.

        Returns:
            (B, 6) vector.
        """
        log_mat = logm(mat)
        return torch.einsum("bij, kij -> bk", log_mat, self.basis)

    def proj_k_to_spacegroup(self, vec: torch.Tensor, spacegroup: torch.Tensor) -> torch.Tensor:
        """Projects lattice vector onto the linear subspace required by the spacegroup.

        Args:
            vec: (B, 6) or (B, 5) vector.
            spacegroup: (B,) spacegroup indices in 1..230.

        Returns:
            (B, dims) projected vector.
        """
        sg_clamped = torch.clamp(spacegroup.long(), min=0, max=230)
        dims = vec.shape[-1]
        if dims == 6:
            sg_mask = self.masks[sg_clamped]
            sg_bias = self.biases[sg_clamped]
        else:
            sg_mask = self.masks[sg_clamped, :-1]
            sg_bias = self.biases[sg_clamped, :-1]
        return vec * sg_mask + sg_bias
