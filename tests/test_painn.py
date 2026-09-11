"""Unit and equivariance tests for WyckoffPaiNN architecture."""

import torch
import pytest
from diffcsp.core.schedulers import SinusoidalTimeEmbeddings
from diffcsp.models.painn_layers import bessel_rbf, WyckoffPaiNNLayer
from diffcsp.models.wyckoff_painn import WyckoffPaiNN
from diffcsp.models.wyckoff_diffusion import WyckoffDiffusion
from diffcsp.models.layers import generate_asymmetric_edges
from tests.test_wyckoff import create_mock_wyckoff_batch


def test_bessel_rbf_properties():
    r_cut = 6.0
    num_rbf = 32
    d_inside = torch.tensor([1.0, 2.5, 5.5])
    d_at_cut = torch.tensor([6.0, 6.1, 10.0])

    rbf_in = bessel_rbf(d_inside, num_rbf=num_rbf, r_cut=r_cut)
    rbf_out = bessel_rbf(d_at_cut, num_rbf=num_rbf, r_cut=r_cut)

    assert rbf_in.shape == (3, num_rbf)
    assert not torch.isnan(rbf_in).any()
    assert (torch.abs(rbf_in) > 1e-4).any()

    # C^2 smooth envelope ensures vanishing at and beyond r_cut
    assert torch.allclose(rbf_out, torch.zeros_like(rbf_out), atol=1e-6)


def test_painn_layer_rotational_equivariance():
    torch.manual_seed(42)
    hidden_dim = 32
    layer = WyckoffPaiNNLayer(hidden_dim=hidden_dim, num_rbf=32, r_cut=6.0)
    layer.eval()

    K, N = 3, 6
    s = torch.randn(K, hidden_dim)
    v = torch.randn(K, hidden_dim, 3)
    site_coords = torch.rand(K, 3)
    full_coords = torch.rand(N, 3)
    lattice = torch.tensor([[[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]]])

    target_sites = torch.repeat_interleave(torch.arange(K), N)
    source_atoms = torch.arange(N).repeat(K)
    source_sites = torch.tensor([0, 0, 1, 1, 2, 2])[source_atoms]
    edge2graph = torch.zeros_like(target_sites)

    # Random 3D rotation Q in SO(3)
    rand_mat = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(rand_mat)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # Baseline forward pass
    s_orig, v_orig, _, _, _ = layer(
        s, v, site_coords, full_coords, lattice,
        target_sites, source_sites, source_atoms, edge2graph
    )

    # Rotated input: L' = L @ Q^T, v' = v @ Q^T
    lattice_rot = lattice @ Q.T
    v_rot = torch.einsum("kca, ab -> kcb", v, Q.T)

    s_rot, v_rot_out, _, _, _ = layer(
        s, v_rot, site_coords, full_coords, lattice_rot,
        target_sites, source_sites, source_atoms, edge2graph
    )

    v_expected = torch.einsum("kca, ab -> kcb", v_orig, Q.T)

    s_err = torch.max(torch.abs(s_orig - s_rot)).item()
    v_err = torch.max(torch.abs(v_expected - v_rot_out)).item()

    assert s_err < 1e-5, f"Scalar invariance broken: max error {s_err}"
    assert v_err < 1e-5, f"Vector equivariance broken: max error {v_err}"


def test_wyckoff_painn_forward_and_backward():
    batch = create_mock_wyckoff_batch()
    painn = WyckoffPaiNN(hidden_dim=64, num_layers=3, num_rbf=32)

    unique_anchors, inverse_site_map = torch.unique(batch.anchor_index, return_inverse=True, sorted=True)
    site2graph = batch.batch[unique_anchors]
    num_sites = torch.bincount(site2graph, minlength=batch.num_graphs)
    site_atom_types = batch.atom_types[unique_anchors]
    site_coords = batch.frac_coords[unique_anchors]
    site_ops = batch.ops[unique_anchors]
    site_P = site_ops[:, :3, :3]

    times = torch.tensor([50, 100], dtype=torch.long)
    time_emb = SinusoidalTimeEmbeddings(256)(times)
    crys_fam = torch.randn((batch.num_graphs, 6))

    lattice_out, coord_out = painn(
        time_emb,
        site_atom_types,
        site_coords,
        batch.frac_coords,
        crys_fam,
        num_sites,
        batch.num_atoms,
        site2graph,
        inverse_site_map,
        site_projectors=site_P,
        batch_ops=batch.ops,
    )

    assert lattice_out.shape == (batch.num_graphs, 6)
    assert coord_out.shape == (len(unique_anchors), 3)

    # Gradient flow check
    loss = lattice_out.sum() + coord_out.sum()
    loss.backward()
    for name, param in painn.named_parameters():
        assert param.grad is not None, f"Parameter {name} received no gradient!"


def test_wyckoff_diffusion_with_painn():
    batch = create_mock_wyckoff_batch()
    decoder = WyckoffPaiNN(hidden_dim=32, num_layers=2, num_rbf=16)
    model = WyckoffDiffusion(decoder=decoder, timesteps=10)

    # Forward loss computation
    out = model(batch)
    assert "loss" in out
    assert "loss_coord" in out
    assert "loss_lattice" in out
    assert not torch.isnan(out["loss"])

    loss = out["loss"]
    loss.backward()

    # Generative sampling test (3 steps)
    model.eval()
    with torch.no_grad():
        out_sample, _ = model.sample(batch, disable_progress=True)
    assert "frac_coords" in out_sample
    assert "lattices" in out_sample
    assert out_sample["frac_coords"].shape == batch.frac_coords.shape


def test_wyckoff_painn_full_so3_equivariance():
    torch.manual_seed(42)
    batch = create_mock_wyckoff_batch()
    painn = WyckoffPaiNN(hidden_dim=32, num_layers=2, num_rbf=16)
    painn.eval()

    unique_anchors, inverse_site_map = torch.unique(batch.anchor_index, return_inverse=True, sorted=True)
    site2graph = batch.batch[unique_anchors]
    num_sites = torch.bincount(site2graph, minlength=batch.num_graphs)
    site_atom_types = batch.atom_types[unique_anchors]
    site_coords = batch.frac_coords[unique_anchors]
    site_ops = batch.ops[unique_anchors]
    site_P = site_ops[:, :3, :3]

    times = torch.tensor([50, 100], dtype=torch.long)
    time_emb = SinusoidalTimeEmbeddings(256)(times)
    crys_fam = torch.randn((batch.num_graphs, 6))
    L_orig = painn.crystal_family.v2m(crys_fam)

    # Baseline forward pass with L_orig
    with torch.no_grad():
        lat_orig, coord_orig = painn(
            time_emb,
            site_atom_types,
            site_coords,
            batch.frac_coords,
            L_orig,
            num_sites,
            batch.num_atoms,
            site2graph,
            inverse_site_map,
            site_projectors=site_P,
            batch_ops=batch.ops,
        )

    # Apply random 3D rotation Q in SO(3)
    rand_mat = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(rand_mat)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    L_rot = L_orig @ Q.T

    with torch.no_grad():
        lat_rot, coord_rot = painn(
            time_emb,
            site_atom_types,
            site_coords,
            batch.frac_coords,
            L_rot,
            num_sites,
            batch.num_atoms,
            site2graph,
            inverse_site_map,
            site_projectors=site_P,
            batch_ops=batch.ops,
        )

    # 1. Lattice parameter score is SO(3) invariant
    lat_err = torch.max(torch.abs(lat_orig - lat_rot)).item()
    assert lat_err < 5e-5, f"Lattice score invariance violated: max err {lat_err}"

    # 2. Fractional coordinate update is invariant (meaning Cartesian displacement is strictly equivariant: Delta_r' = Delta_r @ Q.T)
    coord_err = torch.max(torch.abs(coord_orig - coord_rot)).item()
    assert coord_err < 5e-5, f"Fractional coordinate invariance violated: max err {coord_err}"

    # 3. Explicit check on Cartesian vectors
    L_orig_sites = L_orig[site2graph]
    L_rot_sites = L_rot[site2graph]
    cart_orig = torch.einsum("ki, kij -> kj", coord_orig, L_orig_sites)
    cart_rot = torch.einsum("ki, kij -> kj", coord_rot, L_rot_sites)
    cart_expected = torch.einsum("ki, ij -> kj", cart_orig, Q.T)
    cart_err = torch.max(torch.abs(cart_expected - cart_rot)).item()
    assert cart_err < 5e-5, f"Cartesian coordinate equivariance violated: max err {cart_err}"
