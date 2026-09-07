"""Tests for diffcsp.core.crystal_family."""

import torch

from diffcsp.core.crystal_family import CrystalFamily


def test_basis_orthonormality():
    cf = CrystalFamily()
    assert cf.basis.shape == (6, 3, 3)
    norms = cf.basis.norm(dim=(-1, -2))
    assert torch.allclose(norms, torch.ones(6), atol=1e-5), "All basis matrices must have unit Frobenius norm"


def test_v2m_m2v_roundtrip():
    cf = CrystalFamily()
    # Random Lie algebra vector
    vec = torch.randn(4, 6) * 0.1
    mat = cf.v2m(vec)
    reconstructed_vec = cf.m2v(mat)
    assert torch.allclose(vec, reconstructed_vec, atol=1e-4)


def test_spacegroup_constraints_coverage():
    cf = CrystalFamily()
    # Check all spacegroups 1..230
    assert cf.masks.shape == (231, 6)
    assert cf.biases.shape == (231, 6)
    assert cf.family.shape == (231,)

    # Cubic Fm-3m (225) should constrain off-diagonal and unequal axes
    mask, _ = cf.get_spacegroup_constraint(225)
    assert mask[0] == 0.0 and mask[1] == 0.0 and mask[2] == 0.0 and mask[3] == 0.0 and mask[4] == 0.0
    assert mask[5] == 1.0  # only isotropic volume remains free

    # Triclinic P-1 (2) should be completely unconstrained
    mask_tri, _ = cf.get_spacegroup_constraint(2)
    assert torch.all(mask_tri == 1.0)


def test_de_so3():
    cf = CrystalFamily()
    # Random invertible lattice
    lat_mat = torch.randn(2, 3, 3) + torch.eye(3).unsqueeze(0) * 5.0
    l_sym = cf.de_so3(lat_mat)
    assert torch.allclose(l_sym, l_sym.transpose(-1, -2), atol=1e-5), "de_so3 output must be symmetric"
    # Metric tensor L @ L^T must be preserved
    metric_orig = lat_mat @ lat_mat.transpose(-1, -2)
    metric_sym = l_sym @ l_sym.transpose(-1, -2)
    assert torch.allclose(metric_orig, metric_sym, atol=1e-4), "Metric tensor L L^T must be preserved"
