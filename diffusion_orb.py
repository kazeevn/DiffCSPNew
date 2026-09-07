import torch
import torch.nn as nn
from diffusion1 import CSPDiffusion, SinusoidalTimeEmbeddings
from utils import BetaScheduler, SigmaScheduler
from crystal_family import CrystalFamily
from cspnet_orb import CSPNetORB
from orb_wrapper import build_orb_backbone, OrbBackboneWrapper


class CSPDiffusionORB(CSPDiffusion):
    """
    DiffCSP++ diffusion model powered by a frozen ORB MLIP backbone
    and a lightweight adapter network.
    
    Inherits the exact same training loss, noise schedule, Wyckoff symmetry
    projections, and predictor-corrector sampling algorithm from DiffCSP++,
    while replacing the heavy full-GNN CSPNet with CSPNetORB.
    
    Incorporates the by-design constraint that the denoising step is zero
    ONLY IF the ORB forces are zero (S = 0 ==> F = 0), while allowing non-zero
    steps when F = 0 to escape local minima.
    """

    def __init__(
        self,
        device: str = 'cpu',
        orb_model_name: str = "orb-v2",
        use_mock_orb: bool = False,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_force_residual: bool = True,
        enforce_zero_force_condition: bool = True,
        enforce_zero_stress_condition: bool = False,
        gamma_min: float = 1e-3,
        orb_node_dim: int = 256,
        orb_graph_dim: int = 256
    ) -> None:
        # Initialize base diffusion components (schedulers, time embeddings, crystal family)
        nn.Module.__init__(self)

        self.device = device
        self.beta_scheduler = BetaScheduler(1000, 'cosine')
        self.sigma_scheduler = SigmaScheduler(1000, 0.005, 0.5)
        self.time_dim = 256
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.crystal_family = CrystalFamily()

        # Build frozen ORB backbone wrapper
        self.orb_backbone = build_orb_backbone(
            model_name=orb_model_name,
            device=device,
            use_mock=use_mock_orb,
            node_dim=orb_node_dim,
            graph_dim=orb_graph_dim
        )

        # Build lightweight adapter decoder
        self.decoder = CSPNetORB(
            orb_backbone=self.orb_backbone,
            hidden_dim=hidden_dim,
            latent_dim=self.time_dim,
            num_layers=num_layers,
            use_force_residual=use_force_residual,
            enforce_zero_force_condition=enforce_zero_force_condition,
            enforce_zero_stress_condition=enforce_zero_stress_condition,
            gamma_min=gamma_min,
            orb_node_dim=orb_node_dim,
            orb_graph_dim=orb_graph_dim,
            device=device
        )

    def get_trainable_parameters(self):
        """Returns list of parameters with requires_grad=True (adapter only)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        """Returns a dict with count of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        return {
            'trainable': trainable,
            'frozen': frozen,
            'total': total,
            'trainable_pct': (trainable / total * 100) if total > 0 else 0.0
        }
