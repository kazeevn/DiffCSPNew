"""Comprehensive test suite for DiffCSP-GeoV2 (GeoNet) architecture and diffusion.

Verifies:
1. Exact parameter budget matching: strictly <= 12,283,784 parameters and within 0.05% of 12.28M.
2. Distance-Gated Attention Aggregation in GeoV2CSPLayer.
3. Directional Bond Projection Readout and 2-layer MLP coordinate head in GeoV2CSPNet.
4. Forward pass shape and loss dictionary.
5. Invariance and exact locking of 0-DoF Wyckoff special positions.
6. Gradient flow across all model heads (node head, bond head, attention, gating).
7. Annealed predictor-corrector sampler execution.
8. EMA (Exponential Moving Average) update, shadow application, and checkpoint serialization.
"""

from pathlib import Path
import pytest
import torch
from torch_geometric.data import Batch, Data

from diffcsp.data.dataset import graph_arrays_to_pyg_data
from diffcsp.models.geo_cspnet import GeoCSPNet
from diffcsp.models.geo_v2_cspnet import GeoV2CSPLayer, GeoV2CSPNet
from diffcsp.models.geo_v2_diffusion import GeoV2Diffusion, EMAModel


TEST_DATA_PATH = Path("data/mp-20/test_sym_128.pth")


@pytest.fixture
def mp20_data():
    """Loads crystals from test_sym_128.pth if present."""
    if not TEST_DATA_PATH.exists():
        pytest.skip(f"Data file {TEST_DATA_PATH} not found")
    return torch.load(TEST_DATA_PATH, weights_only=False)


@pytest.fixture
def synthetic_batch():
    """Creates a synthetic batch of crystals with both 0-DoF and free-DoF sites."""
    d1 = Data(
        frac_coords=torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=torch.float32),
        atom_types=torch.tensor([14, 14], dtype=torch.long),
        lengths=torch.tensor([[5.43, 5.43, 5.43]], dtype=torch.float32),
        angles=torch.tensor([[90.0, 90.0, 90.0]], dtype=torch.float32),
        ops=torch.eye(4).unsqueeze(0).repeat(2, 1, 1),
        ops_inv=torch.eye(3).unsqueeze(0).repeat(2, 1, 1),
        anchor_index=torch.tensor([0, 1], dtype=torch.long),
        dofs=torch.tensor([0, 3], dtype=torch.long),
        multiplicities=torch.tensor([1, 1], dtype=torch.long),
        spacegroup=torch.tensor([227], dtype=torch.long),
        num_atoms=torch.tensor([2]),
        num_nodes=2,
    )
    # Give site 0 an exact 0-DoF zero projector
    d1.ops[0, :3, :3] = 0.0

    d2 = Data(
        frac_coords=torch.tensor(
            [[0.5, 0.5, 0.5], [0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=torch.float32
        ),
        atom_types=torch.tensor([8, 22, 22], dtype=torch.long),
        lengths=torch.tensor([[4.59, 4.59, 2.96]], dtype=torch.float32),
        angles=torch.tensor([[90.0, 90.0, 90.0]], dtype=torch.float32),
        ops=torch.eye(4).unsqueeze(0).repeat(3, 1, 1),
        ops_inv=torch.eye(3).unsqueeze(0).repeat(3, 1, 1),
        anchor_index=torch.tensor([0, 1, 1], dtype=torch.long),
        dofs=torch.tensor([0, 1, 1], dtype=torch.long),
        multiplicities=torch.tensor([1, 2, 2], dtype=torch.long),
        spacegroup=torch.tensor([136], dtype=torch.long),
        num_atoms=torch.tensor([3]),
        num_nodes=3,
    )
    d2.ops[0, :3, :3] = 0.0

    return Batch.from_data_list([d1, d2])


def test_exact_parameter_matching():
    """Test 1: Parameter count strictly <= 12,283,784 and within 0.05% of 12.28M."""
    geov1_net = GeoCSPNet(hidden_dim=512, num_layers=6)
    geov2_net = GeoV2CSPNet(hidden_dim=512, num_layers=6, edge_dim=482)

    geov1_params = sum(p.numel() for p in geov1_net.parameters() if p.requires_grad)
    geov2_params = sum(p.numel() for p in geov2_net.parameters() if p.requires_grad)

    # Baseline DiffCSP-Geo has 12,283,784 parameters
    assert geov1_params == 12283784, f"Baseline GeoCSPNet params mismatch: {geov1_params}"

    # Strict constraint 1: <= 12,283,784
    assert geov2_params <= 12283784, (
        f"GeoV2CSPNet exceeded parameter budget: {geov2_params:,} > 12,283,784 "
        f"(excess: +{geov2_params - 12283784})"
    )

    # Strict constraint 2: within 0.05% tolerance of 12.28M
    diff_pct = (12283784 - geov2_params) / 12283784
    assert diff_pct <= 0.0005, (
        f"GeoV2CSPNet parameter delta too large: {geov2_params:,} vs 12,283,784 "
        f"(delta={diff_pct:.4%}, max allowed 0.05%)"
    )

    # Absolute parameter count check
    assert geov2_params == 12282724, f"Expected 12,282,724 params, got {geov2_params:,}"


def test_distance_gated_attention_layer():
    """Test 2: GeoV2CSPLayer distance-gated attention aggregation mechanics."""
    hidden_dim = 64
    edge_dim = 64
    num_nodes = 5
    num_edges = 12

    layer = GeoV2CSPLayer(hidden_dim=hidden_dim, edge_dim=edge_dim)

    node_features = torch.randn(num_nodes, hidden_dim)
    frac_coords = torch.rand(num_nodes, 3)
    lattice_mat = torch.eye(3).unsqueeze(0) * 5.0
    lattice_rep = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge2graph = torch.zeros(num_edges, dtype=torch.long)

    node_out, edge_feat, unit_r = layer(
        node_features=node_features,
        frac_coords=frac_coords,
        lattice_mat=lattice_mat,
        lattice_rep=lattice_rep,
        edge_index=edge_index,
        edge2graph=edge2graph,
    )

    assert node_out.shape == (num_nodes, hidden_dim)
    assert edge_feat.shape == (num_edges, hidden_dim)
    assert unit_r.shape == (num_edges, 3)

    # Check attention gating outputs are bounded and non-NaN
    attn_logits = layer.attn_proj(edge_feat)
    gate = torch.sigmoid(layer.gate_proj(edge_feat))
    assert not torch.isnan(attn_logits).any()
    assert not torch.isnan(gate).any()
    assert torch.all((gate >= 0.0) & (gate <= 1.0))


def test_forward_pass_shape_and_loss(synthetic_batch):
    """Test 3: Forward pass loss computation and tensor shapes on batch."""
    model = GeoV2Diffusion(timesteps=10)
    out = model(synthetic_batch)

    assert "loss" in out
    assert "loss_lattice" in out
    assert "loss_coord" in out

    assert out["loss"].dim() == 0
    assert not torch.isnan(out["loss"])
    assert not torch.isinf(out["loss"])
    assert out["loss"].item() > 0.0

    # Test underlying decoder forward
    B = synthetic_batch.num_graphs
    N = synthetic_batch.num_nodes
    time_emb = model.time_embedding(torch.tensor([5, 5]))
    Pa, Pj, dofs, mult, sg = model.derive_wyckoff_symmetries(synthetic_batch)
    crys_fam = model.crystal_family.m2v(synthetic_batch.lengths.unsqueeze(-1).repeat(1, 1, 3))

    pred_lat, pred_x = model.decoder(
        t=time_emb,
        atom_types=synthetic_batch.atom_types,
        frac_coords=synthetic_batch.frac_coords,
        lattices=crys_fam,
        num_atoms=synthetic_batch.num_atoms,
        node2graph=synthetic_batch.batch,
        spacegroups=sg,
        multiplicities=mult,
        dofs=dofs,
        site_projectors=Pj,
    )
    assert pred_lat.shape == (B, 6)
    assert pred_x.shape == (N, 3)


def test_zero_dof_locking_and_projection(synthetic_batch):
    """Test 4: Invariance and exact locking of 0-DoF sites under tangent projections and sampling."""
    model = GeoV2Diffusion(timesteps=3)
    model.eval()

    Pa, Pj, dofs, multiplicities, spacegroups = model.derive_wyckoff_symmetries(synthetic_batch)
    is_zero_dof = (dofs == 0)
    assert is_zero_dof.any(), "Synthetic batch must include 0-DoF sites"

    # 1. Projectors on 0-DoF sites must be zero matrices
    assert torch.all(Pa[is_zero_dof] == 0.0), "Anchor projector on 0-DoF sites must be 0"
    assert torch.all(Pj[is_zero_dof] == 0.0), "Atom projector on 0-DoF sites must be 0"

    # 2. Predicted coordinates through decoder on 0-DoF sites must be identically zero
    time_emb = model.time_embedding(torch.zeros(synthetic_batch.num_graphs, dtype=torch.long))
    crys_fam = model.crystal_family.proj_k_to_spacegroup(
        torch.randn(synthetic_batch.num_graphs, 6), spacegroups
    )
    _, pred_x = model.decoder(
        t=time_emb,
        atom_types=synthetic_batch.atom_types,
        frac_coords=synthetic_batch.frac_coords,
        lattices=crys_fam,
        num_atoms=synthetic_batch.num_atoms,
        node2graph=synthetic_batch.batch,
        spacegroups=spacegroups,
        multiplicities=multiplicities,
        dofs=dofs,
        site_projectors=Pj,
    )
    assert torch.all(pred_x[is_zero_dof] == 0.0), "Decoder outputs on 0-DoF sites must be strictly 0"

    # 3. Tangent space noise on 0-DoF sites must be strictly zero
    rand_x_raw = torch.randn_like(synthetic_batch.frac_coords)
    free_mask = (dofs > 0)
    rand_x_anchor = (Pa @ rand_x_raw[synthetic_batch.anchor_index].unsqueeze(-1)).squeeze(-1)
    rand_x_anchor = torch.where(
        free_mask.unsqueeze(-1), rand_x_anchor, torch.zeros_like(rand_x_anchor)
    )
    R = synthetic_batch.ops[:, :3, :3]
    rand_x = (R @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
    assert torch.all(rand_x[is_zero_dof] == 0.0), "Noise on 0-DoF sites must be strictly 0"

    # 4. Sampling preserves exact initial coordinates for 0-DoF sites
    init_coords = synthetic_batch.frac_coords.clone()
    sampled, _ = model.sample(synthetic_batch, disable_progress=True)
    sampled_coords = sampled["frac_coords"]

    diff_0dof = (
        ((sampled_coords[is_zero_dof] - init_coords[is_zero_dof] + 0.5) % 1.0 - 0.5)
        .abs()
        .max()
        .item()
    )
    assert diff_0dof < 1e-5, f"0-DoF coordinates drifted during sampling: max diff = {diff_0dof}"


def test_gradient_flow(synthetic_batch):
    """Test 5: Backward pass verifies gradient flow to all heads and layers."""
    model = GeoV2Diffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    model.train()
    optimizer.zero_grad()

    out = model(synthetic_batch)
    loss = out["loss"]
    assert not torch.isnan(loss)

    loss.backward()

    # Check key layer gradients
    assert model.decoder.coord_node_mlp[0].weight.grad is not None
    assert model.decoder.coord_node_mlp[2].weight.grad is not None
    assert model.decoder.bond_mlp[0].weight.grad is not None
    assert model.decoder.bond_mlp[2].weight.grad is not None
    assert model.decoder.csp_layers[0].attn_proj.weight.grad is not None
    assert model.decoder.csp_layers[0].gate_proj.weight.grad is not None
    assert model.decoder.csp_layers[0].edge_mlp[0].weight.grad is not None
    assert model.decoder.csp_layers[0].node_mlp[0].weight.grad is not None
    assert model.decoder.element_emb.weight.grad is not None
    assert model.decoder.lattice_out[0].weight.grad is not None

    # Verify no NaN or Inf gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    optimizer.step()


def test_annealed_sampler_execution(synthetic_batch):
    """Test 6: Annealed predictor-corrector sampler runs cleanly."""
    model = GeoV2Diffusion(timesteps=2)
    model.eval()

    # Test with annealing (default step_lr=5e-6)
    out, traj_stack = model.sample(
        synthetic_batch, step_lr=5e-6, anneal_corrector=True, disable_progress=True
    )
    assert "frac_coords" in out
    assert "lattices" in out
    assert "crys_fam" in out

    assert out["frac_coords"].shape == synthetic_batch.frac_coords.shape
    assert out["lattices"].shape == (synthetic_batch.num_graphs, 3, 3)
    assert not torch.isnan(out["frac_coords"]).any()
    assert not torch.isnan(out["lattices"]).any()

    assert "all_frac_coords" in traj_stack
    assert "all_lattices" in traj_stack
    assert traj_stack["all_frac_coords"].shape[0] == 2
    assert traj_stack["all_lattices"].shape[0] == 2

    # Test without annealing
    out_no_anneal, _ = model.sample(
        synthetic_batch, step_lr=5e-6, anneal_corrector=False, disable_progress=True
    )
    assert not torch.isnan(out_no_anneal["frac_coords"]).any()


def test_ema_functionality(synthetic_batch):
    """Test 7: EMA initialization, parameter shadowing, and checkpoint round-trip."""
    model = GeoV2Diffusion(timesteps=2)
    model.init_ema(decay=0.9999)
    assert model.ema is not None

    orig_weight = model.decoder.coord_node_mlp[0].weight.clone()

    # Perturb model weights to simulate training step
    with torch.no_grad():
        model.decoder.coord_node_mlp[0].weight.add_(torch.ones_like(orig_weight) * 0.5)

    # Shadow before update should match orig_weight
    shadow_before = model.ema.shadow["coord_node_mlp.0.weight"].clone()
    assert torch.allclose(shadow_before, orig_weight)

    # Update EMA: shadow = 0.9999 * shadow + 0.0001 * current
    model.update_ema()
    shadow_after = model.ema.shadow["coord_node_mlp.0.weight"]
    expected = 0.9999 * shadow_before + 0.0001 * model.decoder.coord_node_mlp[0].weight
    assert torch.allclose(shadow_after, expected, atol=1e-6)

    # Test context manager ema_scope
    current_weight = model.decoder.coord_node_mlp[0].weight.clone()
    with model.ema_scope():
        in_scope_weight = model.decoder.coord_node_mlp[0].weight
        assert torch.allclose(in_scope_weight, shadow_after)
    restored_weight = model.decoder.coord_node_mlp[0].weight
    assert torch.allclose(restored_weight, current_weight)

    # Test state dict round-trip
    ema_sd = model.ema_state_dict()
    assert ema_sd is not None
    assert "coord_node_mlp.0.weight" in ema_sd

    new_model = GeoV2Diffusion(timesteps=2)
    new_model.load_ema_state_dict(ema_sd)
    assert new_model.ema is not None
    assert torch.allclose(new_model.ema.shadow["coord_node_mlp.0.weight"], shadow_after)

    # Test sampling with use_ema=True
    out_ema, _ = model.sample(synthetic_batch, disable_progress=True, use_ema=True)
    assert not torch.isnan(out_ema["frac_coords"]).any()


def test_mp20_data_if_available(mp20_data):
    """Test 8: Runs forward pass and 2 sampling steps on actual MP-20 dataset crystals."""
    batch_crystals = [graph_arrays_to_pyg_data(mp20_data[i]) for i in range(4)]
    batch = Batch.from_data_list(batch_crystals)

    model = GeoV2Diffusion(timesteps=2)
    model.init_ema()
    model.eval()

    # Forward loss
    out = model(batch)
    assert "loss" in out
    assert not torch.isnan(out["loss"])
    assert out["loss"].item() > 0.0

    # Sample 2 steps with EMA and annealed corrector
    sampled, traj = model.sample(batch, step_lr=5e-6, disable_progress=True, use_ema=True)
    assert sampled["frac_coords"].shape == batch.frac_coords.shape
    assert sampled["lattices"].shape == (batch.num_graphs, 3, 3)
    assert not torch.isnan(sampled["frac_coords"]).any()
    assert not torch.isnan(sampled["lattices"]).any()
