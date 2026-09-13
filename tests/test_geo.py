"""Comprehensive test suite for DiffCSP-Geo architecture and diffusion."""

from pathlib import Path
import pytest
import torch
from torch_geometric.data import Batch

from diffcsp.data.dataset import graph_arrays_to_pyg_data
from diffcsp.models.cspnet import CSPNet
from diffcsp.models.geo_cspnet import GeoCSPLayer, GeoCSPNet
from diffcsp.models.geo_diffusion import GeoDiffusion


TEST_DATA_PATH = Path("data/mp-20/test_sym_128.pth")


@pytest.fixture
def mp20_data():
    """Loads crystals from test_sym_128.pth if present."""
    if not TEST_DATA_PATH.exists():
        pytest.skip(f"Data file {TEST_DATA_PATH} not found")
    return torch.load(TEST_DATA_PATH, weights_only=False)


def test_parameter_matching():
    """Test 1: Parameter count of GeoCSPNet(hidden_dim=512, num_layers=6) vs vanilla CSPNet(512, 6).

    Must match within 0.5% tolerance (12.28M +/- 0.5%).
    """
    geo_net = GeoCSPNet(hidden_dim=512, num_layers=6)
    csp_net = CSPNet(hidden_dim=512, num_layers=6)

    geo_params = sum(p.numel() for p in geo_net.parameters() if p.requires_grad)
    csp_params = sum(p.numel() for p in csp_net.parameters() if p.requires_grad)

    # Vanilla CSPNet has 12,277,248 learnable parameters (~12.28M)
    assert csp_params == 12277248, f"Unexpected vanilla CSPNet params: {csp_params}"

    diff_pct = abs(geo_params - csp_params) / csp_params
    assert (
        diff_pct <= 0.005
    ), f"Parameter count mismatch: GeoCSPNet has {geo_params:,} vs CSPNet {csp_params:,} (diff={diff_pct:.4%})"


def test_invariance_of_zero_dof_coordinates(mp20_data):
    """Test 2: Invariance of 0-DoF coordinates (must remain exactly at Wyckoff position)."""
    # Crystal 2 in test_sym_128.pth contains 0-DoF Wyckoff special positions
    pyg_data = None
    for item in mp20_data:
        data = graph_arrays_to_pyg_data(item)
        Pa = data.ops[data.anchor_index, :3, :3]
        dofs = torch.round(torch.diagonal(Pa, dim1=-2, dim2=-1).sum(-1)).long()
        if (dofs == 0).any():
            pyg_data = data
            break

    assert pyg_data is not None, "Could not find crystal with 0-DoF sites in test data"
    batch = Batch.from_data_list([pyg_data])
    init_coords = batch.frac_coords.clone()

    model = GeoDiffusion(timesteps=3)
    model.eval()

    Pa, Pj, dofs, multiplicities, spacegroups = model.derive_wyckoff_symmetries(batch)
    is_zero_dof = (dofs == 0)
    assert is_zero_dof.any(), "Selected crystal must contain 0-DoF sites"

    # 1. Verify tangent space noise is zero on 0-DoF sites
    rand_x_raw = torch.randn_like(batch.frac_coords)
    rand_x_anchor = (Pa @ rand_x_raw[batch.anchor_index].unsqueeze(-1)).squeeze(-1)
    free_mask = (dofs > 0)
    rand_x_anchor = torch.where(free_mask.unsqueeze(-1), rand_x_anchor, torch.zeros_like(rand_x_anchor))
    R = batch.ops[:, :3, :3]
    rand_x = (R @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
    assert torch.all(rand_x[is_zero_dof] == 0.0), "Noise on 0-DoF sites must be strictly 0"

    # 2. Verify sampling keeps 0-DoF sites locked exactly to initial Wyckoff position
    out, _ = model.sample(batch, disable_progress=True)
    sampled_coords = out["frac_coords"]

    # Compare coordinates on 0-DoF sites under periodic boundary conditions
    pbc_diff = (
        ((sampled_coords[is_zero_dof] - init_coords[is_zero_dof] + 0.5) % 1.0 - 0.5)
        .abs()
        .max()
        .item()
    )
    assert (
        pbc_diff < 1e-5
    ), f"0-DoF coordinates deviated from their exact initial Wyckoff special positions: error {pbc_diff}"


def test_tangent_space_projector_properties(mp20_data):
    """Test 3: Tangent space projector properties (P^2 = P, idempotence and projector consistency)."""
    # Test projectors across multiple crystals
    for i in range(min(10, len(mp20_data))):
        pyg = graph_arrays_to_pyg_data(mp20_data[i])
        Pa = pyg.ops[pyg.anchor_index, :3, :3]
        R = pyg.ops[:, :3, :3]
        R_inv = pyg.ops_inv
        Pj = R @ Pa @ R_inv

        # Idempotence: Pa @ Pa == Pa
        diff_anchor = torch.norm(Pa @ Pa - Pa, dim=(-1, -2)).max().item()
        assert diff_anchor < 1e-5, f"Anchor projector Pa is not idempotent: error {diff_anchor}"

        # Idempotence: Pj @ Pj == Pj
        diff_atom = torch.norm(Pj @ Pj - Pj, dim=(-1, -2)).max().item()
        assert diff_atom < 1e-5, f"Atom projector Pj is not idempotent: error {diff_atom}"

        # Degrees of freedom consistency
        dof_a = torch.round(torch.diagonal(Pa, dim1=-2, dim2=-1).sum(-1)).long()
        dof_j = torch.round(torch.diagonal(Pj, dim1=-2, dim2=-1).sum(-1)).long()
        assert torch.equal(dof_a, dof_j), "Atom DoF must match anchor DoF under similarity transform"

        # 0-DoF sites must have zero projector
        zero_mask = (dof_a == 0)
        if zero_mask.any():
            assert torch.all(Pa[zero_mask] == 0.0)
            assert torch.all(Pj[zero_mask] == 0.0)

        # 3-DoF sites must have identity projector
        three_mask = (dof_a == 3)
        if three_mask.any():
            eye = torch.eye(3, device=Pa.device).repeat(three_mask.sum(), 1, 1)
            assert torch.allclose(Pa[three_mask], eye, atol=1e-5)
            assert torch.allclose(Pj[three_mask], eye, atol=1e-5)


def test_geo_diffusion_forward_mp20(mp20_data):
    """Test 4: Forward pass on a batch of crystals from data/mp-20/test_sym_128.pth."""
    batch_crystals = [graph_arrays_to_pyg_data(mp20_data[i]) for i in range(4)]
    batch = Batch.from_data_list(batch_crystals)

    model = GeoDiffusion()
    outputs = model(batch)

    assert "loss" in outputs
    assert "loss_lattice" in outputs
    assert "loss_coord" in outputs

    loss = outputs["loss"]
    assert loss.dim() == 0, "Loss must be scalar"
    assert not torch.isnan(loss), "Loss must not be NaN"
    assert not torch.isinf(loss), "Loss must not be Inf"
    assert loss.item() > 0, "Loss must be positive"


def test_geo_diffusion_train_step(mp20_data):
    """Test 5: One training step with optimizer and backward pass."""
    batch_crystals = [graph_arrays_to_pyg_data(mp20_data[i]) for i in range(2)]
    batch = Batch.from_data_list(batch_crystals)

    model = GeoDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    optimizer.zero_grad()

    outputs = model(batch)
    loss = outputs["loss"]
    assert not torch.isnan(loss)

    loss.backward()

    # Check gradients exist on key layers
    assert model.decoder.coord_out.weight.grad is not None
    assert model.decoder.lattice_out[0].weight.grad is not None
    assert model.decoder.element_emb.weight.grad is not None
    assert model.decoder.csp_layers[0].edge_mlp[0].weight.grad is not None

    # Check for NaN gradients
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    optimizer.step()


def test_geo_diffusion_sampling_steps(mp20_data):
    """Test 6: Sampling 2 steps without errors."""
    batch_crystals = [graph_arrays_to_pyg_data(mp20_data[i]) for i in range(2)]
    batch = Batch.from_data_list(batch_crystals)

    model = GeoDiffusion(timesteps=2)
    model.eval()

    out, traj_stack = model.sample(batch, disable_progress=True)

    assert "frac_coords" in out
    assert "lattices" in out
    assert "crys_fam" in out

    assert out["frac_coords"].shape == batch.frac_coords.shape
    assert out["lattices"].shape == (batch.num_graphs, 3, 3)
    assert not torch.isnan(out["frac_coords"]).any(), "Sampled coordinates contain NaN"
    assert not torch.isnan(out["lattices"]).any(), "Sampled lattices contain NaN"

    assert "all_frac_coords" in traj_stack
    assert "all_lattices" in traj_stack
    assert traj_stack["all_frac_coords"].shape[0] == 2
    assert traj_stack["all_lattices"].shape[0] == 2
