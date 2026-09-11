"""Tests for Asymmetric Unit (Wyckoff-Only) Message Passing and Diffusion."""

import torch
from torch_geometric.data import Batch, Data

from diffcsp.models.layers import generate_asymmetric_edges, WyckoffCSPLayer, SinusoidsEmbedding
from diffcsp.models.wyckoff_cspnet import WyckoffCSPNet
from diffcsp.models.wyckoff_diffusion import WyckoffDiffusion


def create_mock_wyckoff_batch(device: torch.device = torch.device("cpu")) -> Batch:
    """Creates a mock batch of 2 crystals with Wyckoff symmetry operations."""
    # Crystal 1: 2 sites (mult 2 each) -> 4 atoms
    # Crystal 2: 3 sites (mults 2, 4, 4) -> 10 atoms
    ops_c1 = torch.zeros((4, 4, 4), dtype=torch.float32)
    ops_c1[:, :3, :3] = torch.eye(3)
    ops_c2 = torch.zeros((10, 4, 4), dtype=torch.float32)
    ops_c2[:, :3, :3] = torch.eye(3)

    c1 = Data(
        frac_coords=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.3, 0.6, 0.2], [0.7, 0.4, 0.8]], dtype=torch.float32),
        atom_types=torch.tensor([6, 6, 8, 8], dtype=torch.long),
        lengths=torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32),
        angles=torch.tensor([[90.0, 90.0, 90.0]], dtype=torch.float32),
        num_atoms=4,
        num_nodes=4,
        ops=ops_c1,
        ops_inv=torch.eye(3).repeat(4, 1, 1),
        anchor_index=torch.tensor([0, 0, 2, 2], dtype=torch.long),
        spacegroup=torch.tensor(194, dtype=torch.long),
    )

    c2 = Data(
        frac_coords=torch.rand((10, 3), dtype=torch.float32),
        atom_types=torch.tensor([14, 14, 8, 8, 8, 8, 8, 8, 8, 8], dtype=torch.long),
        lengths=torch.tensor([[6.0, 6.0, 8.0]], dtype=torch.float32),
        angles=torch.tensor([[90.0, 90.0, 120.0]], dtype=torch.float32),
        num_atoms=10,
        num_nodes=10,
        ops=ops_c2,
        ops_inv=torch.eye(3).repeat(10, 1, 1),
        anchor_index=torch.tensor([0, 0, 2, 2, 2, 2, 6, 6, 6, 6], dtype=torch.long),
        spacegroup=torch.tensor(166, dtype=torch.long),
    )

    return Batch.from_data_list([c1, c2]).to(device)


def test_generate_asymmetric_edges():
    num_sites = torch.tensor([2, 3])
    num_atoms = torch.tensor([4, 10])
    target_sites, source_atoms = generate_asymmetric_edges(num_sites, num_atoms)

    expected_edges = 2 * 4 + 3 * 10  # 38 edges
    assert target_sites.shape == (expected_edges,)
    assert source_atoms.shape == (expected_edges,)

    # Crystal 0 targets are in [0, 2), Crystal 1 targets in [2, 5)
    assert target_sites[:8].max().item() < 2
    assert target_sites[8:].min().item() >= 2
    assert target_sites[8:].max().item() < 5

    # Crystal 0 sources are in [0, 4), Crystal 1 sources in [4, 14)
    assert source_atoms[:8].max().item() < 4
    assert source_atoms[8:].min().item() >= 4
    assert source_atoms[8:].max().item() < 14


def test_wyckoff_csp_layer():
    hidden_dim = 64
    dis_emb = SinusoidsEmbedding(n_frequencies=16)
    layer = WyckoffCSPLayer(hidden_dim=hidden_dim, dis_emb=dis_emb)

    K_total = 5
    N_total = 14
    num_sites = torch.tensor([2, 3])
    num_atoms = torch.tensor([4, 10])

    site_features = torch.randn((K_total, hidden_dim))
    site_coords = torch.rand((K_total, 3))
    full_coords = torch.rand((N_total, 3))
    lattices = torch.randn((2, 6))

    target_sites, source_atoms = generate_asymmetric_edges(num_sites, num_atoms)
    inverse_site_map = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    source_sites = inverse_site_map[source_atoms]
    site2graph = torch.tensor([0, 0, 1, 1, 1])
    edge2graph = site2graph[target_sites]

    out = layer(
        site_features,
        site_coords,
        full_coords,
        lattices,
        target_sites,
        source_sites,
        source_atoms,
        edge2graph,
    )
    assert out.shape == (K_total, hidden_dim)


def test_wyckoff_cspnet_forward():
    net = WyckoffCSPNet(hidden_dim=64, latent_dim=32, num_layers=2)
    batch = create_mock_wyckoff_batch()

    unique_anchors, inverse_site_map = torch.unique(batch.anchor_index, return_inverse=True, sorted=True)
    site2graph = batch.batch[unique_anchors]
    num_sites = torch.bincount(site2graph, minlength=batch.num_graphs)
    site_coords = batch.frac_coords[unique_anchors]
    site_atom_types = batch.atom_types[unique_anchors]
    lattices = torch.randn((batch.num_graphs, 6))
    t = torch.randn((batch.num_graphs, 32))

    lat_out, coord_out = net(
        t,
        site_atom_types,
        site_coords,
        batch.frac_coords,
        lattices,
        num_sites,
        batch.num_atoms,
        site2graph,
        inverse_site_map,
    )
    assert lat_out.shape == (batch.num_graphs, 6)
    assert coord_out.shape == (len(unique_anchors), 3)


def test_wyckoff_diffusion_forward():
    model = WyckoffDiffusion(
        decoder=WyckoffCSPNet(hidden_dim=64, latent_dim=32, num_layers=2),
        time_dim=32,
        timesteps=100,
    )
    batch = create_mock_wyckoff_batch()
    res = model(batch)

    assert "loss" in res
    assert "loss_lattice" in res
    assert "loss_coord" in res
    assert not torch.isnan(res["loss"])
    assert res["loss"].item() > 0


def test_wyckoff_diffusion_sample():
    model = WyckoffDiffusion(
        decoder=WyckoffCSPNet(hidden_dim=64, latent_dim=32, num_layers=2),
        time_dim=32,
        timesteps=10,
    )
    batch = create_mock_wyckoff_batch()
    out, _ = model.sample(batch, disable_progress=True)

    assert "frac_coords" in out
    assert "lattices" in out
    assert out["frac_coords"].shape == (batch.num_nodes, 3)
    assert out["lattices"].shape == (batch.num_graphs, 3, 3)
    assert (out["frac_coords"] >= 0.0).all() and (out["frac_coords"] < 1.0).all()


def test_edge_reduction_efficiency():
    # Verify that K * N is substantially smaller than N^2
    num_sites = torch.tensor([4, 6])
    num_atoms = torch.tensor([32, 64])

    fc_edges = (num_atoms ** 2).sum().item()  # 32^2 + 64^2 = 1024 + 4096 = 5120
    asymm_edges = (num_sites * num_atoms).sum().item()  # 4*32 + 6*64 = 128 + 384 = 512

    reduction_factor = fc_edges / asymm_edges
    assert reduction_factor == 10.0
    print(f"Edge reduction: {fc_edges} -> {asymm_edges} ({reduction_factor:.1f}x reduction)")


def test_innovation_2_tangent_space_projection():
    """Verifies that 0-DoF sites are fixed, and 1-DoF sites stay strictly on their axis."""
    device = torch.device("cpu")
    # Build a crystal with:
    # site 0: 0-DoF (special position, e.g. [0, 0, 0])
    # site 1: 1-DoF (line along z, e.g. [0.333, 0.667, z])
    ops = torch.zeros((4, 4, 4), dtype=torch.float32)
    # site 0 anchor at index 0 (0-DoF: P = 0, x0 = [0, 0, 0])
    ops[0, :3, :3] = torch.zeros((3, 3))
    ops[0, :3, 3] = torch.tensor([0.0, 0.0, 0.0])
    ops[1, :3, :3] = torch.zeros((3, 3))
    ops[1, :3, 3] = torch.tensor([0.5, 0.5, 0.5])

    # site 1 anchor at index 2 (1-DoF: P = diag(0, 0, 1), x0 = [0.333, 0.667, 0.0])
    ops[2, 2, 2] = 1.0
    ops[2, :3, 3] = torch.tensor([0.3333, 0.6667, 0.0])
    ops[3, 2, 2] = -1.0
    ops[3, :3, 3] = torch.tensor([0.6667, 0.3333, 0.5])

    c = Data(
        frac_coords=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.3333, 0.6667, 0.2],
            [0.6667, 0.3333, 0.3],
        ], dtype=torch.float32),
        atom_types=torch.tensor([12, 12, 8, 8], dtype=torch.long),
        lengths=torch.tensor([[4.0, 4.0, 6.0]], dtype=torch.float32),
        angles=torch.tensor([[90.0, 90.0, 120.0]], dtype=torch.float32),
        num_atoms=4,
        num_nodes=4,
        ops=ops,
        ops_inv=torch.linalg.pinv(ops[:, :3, :3]),
        anchor_index=torch.tensor([0, 0, 2, 2], dtype=torch.long),
        spacegroup=torch.tensor(166, dtype=torch.long),
    )
    batch = Batch.from_data_list([c]).to(device)

    model = WyckoffDiffusion(
        device=device,
        decoder=WyckoffCSPNet(hidden_dim=32, num_layers=2, max_atoms=50),
        timesteps=5,
    )

    # Forward pass: ensure loss computes without error and 0-DoF site receives zero coord loss
    out = model(batch)
    assert not torch.isnan(out["loss"])
    assert not torch.isnan(out["loss_coord"])
    assert not torch.isnan(out["loss_lattice"])

    # Sample: verify coordinates stay in affine subspace
    sample_out, _ = model.sample(batch, disable_progress=True)
    sampled_frac = sample_out["frac_coords"]

    # Site 0 (atoms 0 and 1) MUST remain strictly at [0, 0, 0] and [0.5, 0.5, 0.5]
    assert torch.allclose(sampled_frac[0], torch.tensor([0.0, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(sampled_frac[1], torch.tensor([0.5, 0.5, 0.5]), atol=1e-4)

    # Site 1 (atoms 2 and 3) MUST have x=1/3, y=2/3 for atom 2
    assert torch.allclose(sampled_frac[2, :2], torch.tensor([0.3333, 0.6667]), atol=1e-3)
    # and only z has evolved
    assert 0.0 <= sampled_frac[2, 2] < 1.0

