"""Tests for diffcsp.models (CSPNet, CSPNetORB, MockOrbBackbone, CSPLayer)."""

import torch

from diffcsp.models.cspnet import CSPNet
from diffcsp.models.cspnet_orb import CSPNetORB
from diffcsp.models.layers import CSPLayer, SinusoidsEmbedding


def test_sinusoids_embedding():
    emb_module = SinusoidsEmbedding(n_frequencies=8, n_space=3)
    x = torch.rand(5, 3)
    emb = emb_module(x)
    assert emb.shape == (5, 8 * 2 * 3)


def test_csp_layer():
    layer = CSPLayer(hidden_dim=32, ln=True)
    node_feat = torch.randn(6, 32)
    frac_coords = torch.rand(6, 3)
    lattices = torch.randn(2, 6)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long)
    edge2graph = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    out = layer(node_feat, frac_coords, lattices, edge_index, edge2graph)
    assert out.shape == (6, 32)


def test_cspnet_forward():
    model = CSPNet(hidden_dim=64, num_layers=2)
    B, N = 2, 6
    t = torch.randn(B, 256)
    atom_types = torch.tensor([14, 14, 8, 14, 8, 8], dtype=torch.long)
    frac_coords = torch.rand(N, 3)
    crys_fam = torch.randn(B, 6)
    num_atoms = torch.tensor([3, 3], dtype=torch.long)
    node2graph = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    lat_out, coord_out = model(t, atom_types, frac_coords, crys_fam, num_atoms, node2graph)
    assert lat_out.shape == (B, 6)
    assert coord_out.shape == (N, 3)


def test_cspnet_orb_zero_force_condition():
    """Verify mathematical properties of the zero-force constraint:

    1. S = 0 ==> f_frac = 0 (contrapositive: f_frac != 0 ==> S != 0)
    2. f_frac = 0 allows S != 0 (to escape local minima)
    """
    model = CSPNetORB(
        hidden_dim=32,
        num_layers=1,
        enforce_zero_force_condition=True,
        gamma_min=1e-3,
        use_mock=True,
    )
    N = 4
    v_coord = torch.randn(N, 3)
    node_features = torch.randn(N, 32)
    t_per_atom = torch.randn(N, 256)

    # Case A: Non-zero forces -> S cannot be zero
    f_nonzero = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.1, -0.2, 0.3], [0.0, 0.0, 2.0]])
    score_a = model.apply_zero_force_constraint(v_coord, f_nonzero, t_per_atom, node_features)
    s_norm_a = torch.norm(score_a, dim=-1)
    assert torch.all(s_norm_a > 1e-4), "If F != 0, score S must be strictly non-zero"

    # Case B: Zero forces -> S equals v_perp = v_coord, allowing non-zero proposals
    f_zero = torch.zeros(N, 3)
    score_b = model.apply_zero_force_constraint(v_coord, f_zero, t_per_atom, node_features)
    assert torch.allclose(score_b, v_coord, atol=1e-5), (
        "If F = 0, score S can be non-zero to escape local minima"
    )


def test_mock_orb_backbone():
    from diffcsp.models.orb_wrapper import MockOrbBackbone

    backbone = MockOrbBackbone(node_dim=64, graph_dim=64)
    N, B = 6, 2
    atom_types = torch.tensor([1, 6, 8, 14, 1, 8], dtype=torch.long)
    cart_coords = torch.rand(N, 3) * 5.0
    lattices = torch.eye(3).unsqueeze(0).repeat(B, 1, 1) * 5.0
    num_atoms = torch.tensor([3, 3], dtype=torch.long)
    node2graph = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    out = backbone(atom_types, cart_coords, lattices, num_atoms, node2graph)
    assert out["node_emb"].shape == (N, 64)
    assert out["graph_emb"].shape == (B, 64)
    assert out["forces"].shape == (N, 3)
    assert out["stress"].shape == (B, 3, 3)


def test_cspnet_orb_forward():
    model = CSPNetORB(hidden_dim=32, num_layers=1, use_mock=True)
    B, N = 2, 6
    t = torch.randn(B, 256)
    atom_types = torch.tensor([1, 6, 8, 14, 1, 8], dtype=torch.long)
    frac_coords = torch.rand(N, 3)
    crys_fam = model.crystal_family.m2v(torch.eye(3).unsqueeze(0).repeat(B, 1, 1) * 5.0)
    num_atoms = torch.tensor([3, 3], dtype=torch.long)
    node2graph = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    lat_out, coord_out = model(t, atom_types, frac_coords, crys_fam, num_atoms, node2graph)
    assert lat_out.shape == (B, 6)
    assert coord_out.shape == (N, 3)
