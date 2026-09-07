"""Tests for diffcsp.models.diffusion and diffcsp.models.diffusion_orb."""

import torch

from diffcsp.models.cspnet import CSPNet
from diffcsp.models.diffusion import CSPDiffusion
from diffcsp.models.diffusion_orb import CSPDiffusionORB


def test_csp_diffusion_forward(synthetic_batch):
    decoder = CSPNet(hidden_dim=32, num_layers=1)
    diff = CSPDiffusion(device="cpu", decoder=decoder, timesteps=10)
    out = diff(synthetic_batch)
    assert "loss" in out
    assert "loss_lattice" in out
    assert "loss_coord" in out
    assert not torch.isnan(out["loss"])


def test_csp_diffusion_orb_forward(synthetic_batch):
    diff_orb = CSPDiffusionORB(
        device="cpu",
        use_mock_orb=True,
        hidden_dim=32,
        num_layers=1,
        timesteps=10,
    )
    out = diff_orb(synthetic_batch)
    assert "loss" in out
    assert not torch.isnan(out["loss"])

    # Test parameter counts
    counts = diff_orb.count_parameters()
    assert counts["trainable"] > 0
    assert counts["frozen"] > 0
    assert counts["total"] == counts["trainable"] + counts["frozen"]


def test_csp_diffusion_sample_step(synthetic_batch):
    decoder = CSPNet(hidden_dim=32, num_layers=1)
    diff = CSPDiffusion(device="cpu", decoder=decoder, timesteps=3)
    outputs, traj_stack = diff.sample(synthetic_batch, disable_progress=True)
    assert "frac_coords" in outputs
    assert "lattices" in outputs
    assert outputs["frac_coords"].shape == (8, 3)
    assert outputs["lattices"].shape == (2, 3, 3)
