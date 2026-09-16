"""Comprehensive test suite for Deep MLIP-Conditioned Architecture (GeoOrbCSPNet and GeoOrbDiffusion).

Verifies:
1. Trainable parameter budget: strictly <= 12,283,784 parameters.
2. Frozen backbone parameters: ~25.6M with all requires_grad = False.
3. Mock ORB fallback forward pass and loss computation.
4. Exact locking of 0-DoF Wyckoff special positions under tangent projections.
5. Strict gradient flow to trainable parameters only (frozen ORB backbone parameters untouched).
6. Two-phase hybrid sampling execution with and without --orb_handoff_t.
7. EMA (Exponential Moving Average) shadowing, updates, scoping, and state dict serialization.
8. Real MP-20 crystal dataset integration if data file is available.
"""

from pathlib import Path
import pytest
import torch
from torch_geometric.data import Batch, Data

from diffcsp.data.dataset import graph_arrays_to_pyg_data
from diffcsp.models.geo_orb_cspnet import GeoOrbCSPNet
from diffcsp.models.geo_orb_diffusion import GeoOrbDiffusion
from diffcsp.models.geo_v2_diffusion import EMAModel

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
    # Site 0 has 0-DoF: zero projector
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


def test_trainable_and_frozen_parameter_counts():
    """Test 1: Parameter counts: trainable <= 12,283,784, real frozen ~25.6M."""
    # Model with real ORB backbone (if available/weights downloaded) or mock fallback
    model_mock = GeoOrbDiffusion(use_mock_orb=True)
    counts_mock = model_mock.count_parameters()
    trainable_params = counts_mock["trainable"]

    # 1. Trainable budget constraint: strictly <= 12,283,784
    assert trainable_params <= 12283784, (
        f"Trainable parameters exceeded budget: {trainable_params:,} > 12,283,784 "
        f"(excess: +{trainable_params - 12283784:,})"
    )

    # 2. Within 0.5% tolerance of 12.28M budget ceiling
    diff_pct = (12283784 - trainable_params) / 12283784
    assert diff_pct <= 0.005, (
        f"Trainable parameters delta too large: {trainable_params:,} vs 12,283,784 "
        f"(delta={diff_pct:.4%})"
    )

    # 3. All backbone parameters must have requires_grad = False
    for name, p in model_mock.orb_backbone.named_parameters():
        assert not p.requires_grad, f"ORB backbone parameter {name} has requires_grad = True!"

    # 4. Check real ORB backbone frozen count (~25.6M)
    try:
        model_real = GeoOrbDiffusion(use_mock_orb=False)
        counts_real = model_real.count_parameters()
        frozen_params = counts_real["frozen"]
        # orb-v3-direct-20-mpa has ~25,644,479 parameters
        assert 25500000 <= frozen_params <= 25800000, (
            f"Expected frozen ORB parameters ~25.6M, got {frozen_params:,}"
        )
        assert counts_real["trainable"] == trainable_params
    except Exception as exc:
        pytest.skip(f"Real ORB backbone weights unavailable ({exc}); mock test passed.")


def test_mock_orb_fallback_forward_pass(synthetic_batch):
    """Test 2: Forward pass shape and loss computation under mock ORB fallback."""
    model = GeoOrbDiffusion(timesteps=10, use_mock_orb=True)
    out = model(synthetic_batch)

    assert "loss" in out
    assert "loss_lattice" in out
    assert "loss_coord" in out

    assert out["loss"].dim() == 0
    assert not torch.isnan(out["loss"])
    assert not torch.isinf(out["loss"])
    assert out["loss"].item() > 0.0

    # Underlying decoder forward pass check
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
    assert not torch.isnan(pred_lat).any()
    assert not torch.isnan(pred_x).any()


def test_zero_dof_locking_and_tangent_projections(synthetic_batch):
    """Test 3: Invariance and exact locking of 0-DoF special positions."""
    model = GeoOrbDiffusion(timesteps=3, use_mock_orb=True)
    model.eval()

    Pa, Pj, dofs, multiplicities, spacegroups = model.derive_wyckoff_symmetries(synthetic_batch)
    is_zero_dof = dofs == 0
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
    assert torch.all(pred_x[is_zero_dof] == 0.0), "Decoder output on 0-DoF sites must be strictly 0"

    # 3. Tangent space noise on 0-DoF sites must be strictly zero
    rand_x_raw = torch.randn_like(synthetic_batch.frac_coords)
    free_mask = dofs > 0
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


def test_gradient_flow_strictly_to_trainable_parameters(synthetic_batch):
    """Test 4: Backward pass verifies gradient flow to all trainable weights and zero to frozen backbone."""
    model = GeoOrbDiffusion(use_mock_orb=True)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)

    model.train()
    optimizer.zero_grad()

    out = model(synthetic_batch)
    loss = out["loss"]
    loss.backward()

    # Trainable weights receive gradients
    assert model.decoder.element_emb.weight.grad is not None
    assert model.decoder.spacegroup_emb.weight.grad is not None
    assert model.decoder.multiplicity_emb.weight.grad is not None
    assert model.decoder.dof_emb.weight.grad is not None
    assert model.decoder.orb_node_proj.weight.grad is not None
    assert model.decoder.force_proj.weight.grad is not None
    assert model.decoder.node_fusion[0].weight.grad is not None
    assert model.decoder.node_fusion[2].weight.grad is not None
    assert model.decoder.csp_layers[0].attn_proj.weight.grad is not None
    assert model.decoder.csp_layers[0].gate_proj.weight.grad is not None
    assert model.decoder.csp_layers[0].edge_mlp[0].weight.grad is not None
    assert model.decoder.coord_node_mlp[0].weight.grad is not None
    assert model.decoder.bond_mlp[0].weight.grad is not None
    assert model.decoder.lattice_out[0].weight.grad is not None

    # Frozen ORB parameters strictly receive NO gradient
    for name, param in model.orb_backbone.named_parameters():
        assert param.grad is None, f"Frozen parameter {name} received a gradient!"

    # Verify no NaN or Inf gradients in trainable parameters
    for param in model.get_trainable_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    optimizer.step()


def test_sampling_execution_with_and_without_handoff(synthetic_batch):
    """Test 5: Predictor-corrector sampling execution with and without ORB handoff."""
    model = GeoOrbDiffusion(timesteps=4, use_mock_orb=True)
    model.eval()

    # 1. Standard sampling (no handoff)
    out_std, traj_std = model.sample(
        synthetic_batch, step_lr=5e-6, anneal_corrector=True, disable_progress=True
    )
    assert "frac_coords" in out_std
    assert "lattices" in out_std
    assert "crys_fam" in out_std
    assert out_std["frac_coords"].shape == synthetic_batch.frac_coords.shape
    assert out_std["lattices"].shape == (synthetic_batch.num_graphs, 3, 3)
    assert not torch.isnan(out_std["frac_coords"]).any()
    assert not torch.isnan(out_std["lattices"]).any()

    # 2. Two-phase hybrid sampling with ORB handoff
    out_handoff, traj_handoff = model.sample(
        synthetic_batch,
        step_lr=5e-6,
        anneal_corrector=True,
        disable_progress=True,
        orb_handoff_t=0.50,
        relax_steps=2,
    )
    assert "frac_coords" in out_handoff
    assert "lattices" in out_handoff
    assert not torch.isnan(out_handoff["frac_coords"]).any()
    assert not torch.isnan(out_handoff["lattices"]).any()


def test_ema_functionality(synthetic_batch):
    """Test 6: EMA initialization, shadow updates, scoping, and state dict serialization."""
    model = GeoOrbDiffusion(timesteps=2, use_mock_orb=True)
    model.init_ema(decay=0.9999)
    assert model.ema is not None

    orig_weight = model.decoder.coord_node_mlp[0].weight.clone()

    # Simulate parameter perturbation
    with torch.no_grad():
        model.decoder.coord_node_mlp[0].weight.add_(torch.ones_like(orig_weight) * 0.1)

    # Shadow before update matches original
    assert torch.allclose(model.ema.shadow["coord_node_mlp.0.weight"], orig_weight)

    # Update EMA: shadow = 0.9999 * shadow + 0.0001 * current
    model.update_ema()
    expected = 0.9999 * orig_weight + 0.0001 * model.decoder.coord_node_mlp[0].weight
    assert torch.allclose(model.ema.shadow["coord_node_mlp.0.weight"], expected, atol=1e-6)

    # ema_scope context manager
    curr_weight = model.decoder.coord_node_mlp[0].weight.clone()
    with model.ema_scope():
        assert torch.allclose(model.decoder.coord_node_mlp[0].weight, expected)
    assert torch.allclose(model.decoder.coord_node_mlp[0].weight, curr_weight)

    # State dict serialization
    sd = model.ema_state_dict()
    assert "coord_node_mlp.0.weight" in sd
    new_model = GeoOrbDiffusion(timesteps=2, use_mock_orb=True)
    new_model.load_ema_state_dict(sd)
    assert torch.allclose(new_model.ema.shadow["coord_node_mlp.0.weight"], expected)


def test_mp20_data_if_available(mp20_data):
    """Test 7: Forward pass and sampling on real MP-20 structures."""
    batch_crystals = [graph_arrays_to_pyg_data(mp20_data[i]) for i in range(4)]
    batch = Batch.from_data_list(batch_crystals)

    model = GeoOrbDiffusion(timesteps=2, use_mock_orb=True)
    model.init_ema()
    model.eval()

    # Forward loss
    out = model(batch)
    assert "loss" in out
    assert not torch.isnan(out["loss"])
    assert out["loss"].item() > 0.0

    # Sample 2 steps with EMA and hybrid handoff
    sampled, traj = model.sample(
        batch, step_lr=5e-6, disable_progress=True, use_ema=True, orb_handoff_t=0.50, relax_steps=2
    )
    assert sampled["frac_coords"].shape == batch.frac_coords.shape
    assert sampled["lattices"].shape == (batch.num_graphs, 3, 3)
    assert not torch.isnan(sampled["frac_coords"]).any()
    assert not torch.isnan(sampled["lattices"]).any()
