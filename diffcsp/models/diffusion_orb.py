"""DiffCSP++ diffusion model powered by a frozen ORB MLIP backbone."""

from typing import Any

import torch
import torch.nn as nn

from diffcsp.models.cspnet_orb import CSPNetORB
from diffcsp.models.diffusion import CSPDiffusion
from diffcsp.models.orb_wrapper import build_orb_backbone


class CSPDiffusionORB(CSPDiffusion):
    """DiffCSP++ crystal diffusion model with a frozen ORB MLIP potential backbone."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        orb_model_name: str = "orb-v3",
        use_mock_orb: bool = False,
        use_orb_node_features: bool = True,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_force_residual: bool = False,
        enforce_zero_force_condition: bool = False,
        enforce_zero_stress_condition: bool = False,
        gamma_min: float = 1e-3,
        orb_node_dim: int = 256,
        orb_graph_dim: int = 256,
        timesteps: int = 1000,
        beta_scheduler_mode: str = "cosine",
        sigma_begin: float = 0.005,
        sigma_end: float = 0.5,
    ) -> None:
        device_obj = torch.device(device)
        orb_backbone = build_orb_backbone(
            model_name=orb_model_name,
            device=device_obj,
            use_mock=use_mock_orb,
            node_dim=orb_node_dim,
            graph_dim=orb_graph_dim,
            use_node_features=use_orb_node_features,
        )

        decoder = CSPNetORB(
            orb_backbone=orb_backbone,
            hidden_dim=hidden_dim,
            latent_dim=256,
            num_layers=num_layers,
            use_force_residual=use_force_residual,
            enforce_zero_force_condition=enforce_zero_force_condition,
            enforce_zero_stress_condition=enforce_zero_stress_condition,
            gamma_min=gamma_min,
            orb_node_dim=orb_node_dim,
            orb_graph_dim=orb_graph_dim,
            device=device_obj,
        )

        super().__init__(
            device=device_obj,
            decoder=decoder,
            time_dim=256,
            timesteps=timesteps,
            beta_scheduler_mode=beta_scheduler_mode,
            sigma_begin=sigma_begin,
            sigma_end=sigma_end,
        )
        self.orb_backbone = orb_backbone

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Returns only the trainable adapter parameters (excluding frozen ORB backbone)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> dict[str, Any]:
        """Returns parameter counts distinguishing trainable adapter vs frozen backbone."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_pct": (trainable / total * 100.0) if total > 0 else 0.0,
        }
