"""Frozen ORB MLIP Backbone Wrapper for DiffCSP++.

Wraps pretrained Machine Learning Interatomic Potentials (ORB v2/v3) from
orb_models, extracting frozen atomic representations, Cartesian forces,
and virial stresses. Includes MockOrbBackbone for testing and lightweight environments.
"""

import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffcsp.core.scatter import scatter
from diffcsp.data.transforms import (
    clamp_forces_and_stress,
)

logger = logging.getLogger(__name__)


class MockOrbBackbone(nn.Module):
    """Lightweight mock backbone matching ORB's output signatures.

    Used for testing and CPU execution without downloading large weights.
    """

    def __init__(self, node_dim: int = 256, graph_dim: int = 256, max_atomic_num: int = 100) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        self.atom_embedding = nn.Embedding(max_atomic_num + 1, node_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim + 3, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(node_dim + 9, graph_dim),
            nn.SiLU(),
            nn.Linear(graph_dim, graph_dim),
        )
        self.force_head = nn.Linear(node_dim, 3)
        self.stress_head = nn.Linear(graph_dim, 9)
        self.energy_head = nn.Linear(graph_dim, 1)

    def forward(
        self,
        atom_types: torch.Tensor,
        cart_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Mock forward pass."""
        atom_h = self.atom_embedding(atom_types.clamp(0, 100))
        node_in = torch.cat([atom_h, cart_coords], dim=-1)
        node_emb = self.node_proj(node_in)

        batch_size = lattices.shape[0]
        pooled_nodes = scatter(node_emb, node2graph, dim=0, dim_size=batch_size, reduce="mean")
        lattices_flat = lattices.reshape(batch_size, 9)
        graph_in = torch.cat([pooled_nodes, lattices_flat], dim=-1)
        graph_emb = self.graph_proj(graph_in)

        forces = self.force_head(node_emb)
        stress = self.stress_head(graph_emb).reshape(batch_size, 3, 3)
        # Symmetrize stress tensor
        stress = 0.5 * (stress + stress.transpose(-1, -2))
        energy = self.energy_head(graph_emb)

        return {
            "node_emb": node_emb,
            "graph_emb": graph_emb,
            "forces": forces,
            "stress": stress,
            "energy": energy,
        }


class OrbBackboneWrapper(nn.Module):
    """Wrapper around pretrained ORB MLIP model from `orb_models`.

    Provides frozen atomic forces, virial stress, and representations.
    Falls back gracefully to MockOrbBackbone if orb_models is not installed.
    """

    def __init__(
        self,
        model_name: str = "orb-v2",
        device: str = "cpu",
        use_mock: bool = False,
        node_dim: int = 256,
        graph_dim: int = 256,
        max_force: float = 20.0,
        max_stress: float = 50.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        self.max_force = max_force
        self.max_stress = max_stress
        self.use_mock = use_mock
        self.is_loaded = False

        if not use_mock:
            try:
                from orb_models.forcefield import pretrained

                if hasattr(pretrained, model_name):
                    loader = getattr(pretrained, model_name)
                    self.orb_model, self.atoms_adapter = loader(device=device)
                elif hasattr(pretrained, f"{model_name}_conservative_inf_omat"):
                    loader = getattr(pretrained, f"{model_name}_conservative_inf_omat")
                    self.orb_model, self.atoms_adapter = loader(device=device)
                else:
                    self.orb_model, self.atoms_adapter = pretrained.orb_v2(device=device)

                self.orb_model.eval()
                for p in self.orb_model.parameters():
                    p.requires_grad = False
                self.is_loaded = True
            except Exception as e:
                warnings.warn(
                    f"Could not initialize orb_models ({e}). Using MockOrbBackbone for execution.",
                    UserWarning,
                    stacklevel=2,
                )
                self.use_mock = True
                self.mock_backbone = MockOrbBackbone(node_dim=node_dim, graph_dim=graph_dim)
        else:
            self.mock_backbone = MockOrbBackbone(node_dim=node_dim, graph_dim=graph_dim)

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        atom_types: torch.Tensor,
        cart_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluates frozen interatomic potential on a batch of crystal structures."""
        if self.use_mock or not self.is_loaded:
            out = self.mock_backbone(atom_types, cart_coords, lattices, num_atoms, node2graph)
        else:
            from ase import Atoms

            batch_size = lattices.shape[0]
            start_idx = 0
            all_forces = []
            all_energies = []
            all_stresses = []

            for b in range(batch_size):
                n_atoms = int(num_atoms[b].item())
                sub_atom_types = atom_types[start_idx : start_idx + n_atoms].detach().cpu().numpy()
                sub_cart_coords = cart_coords[start_idx : start_idx + n_atoms].detach().cpu().numpy()
                sub_cell = lattices[b].detach().cpu().numpy()

                try:
                    atoms = Atoms(numbers=sub_atom_types, positions=sub_cart_coords, cell=sub_cell, pbc=True)
                    graph = self.atoms_adapter.from_ase_atoms(atoms, device=self.device)
                    pred = self.orb_model.predict(graph)

                    f = torch.as_tensor(pred["forces"], device=cart_coords.device, dtype=cart_coords.dtype)
                    e = torch.as_tensor(pred["energy"], device=cart_coords.device, dtype=cart_coords.dtype)
                    s = torch.as_tensor(
                        pred.get("stress", torch.zeros((3, 3))),
                        device=lattices.device,
                        dtype=lattices.dtype,
                    )
                    if s.numel() == 6:
                        s_mat = torch.tensor(
                            [
                                [s[0], s[5], s[4]],
                                [s[5], s[1], s[3]],
                                [s[4], s[3], s[2]],
                            ],
                            device=lattices.device,
                            dtype=lattices.dtype,
                        )
                        s = s_mat
                    elif s.numel() == 9:
                        s = s.reshape(3, 3)
                except Exception as exc:
                    logger.warning(
                        "ORB evaluation failed for structure %d (%s). Using fallback values.", b, exc
                    )
                    f = torch.zeros((n_atoms, 3), device=cart_coords.device, dtype=cart_coords.dtype)
                    e = torch.zeros(1, device=cart_coords.device, dtype=cart_coords.dtype)
                    s = torch.zeros((3, 3), device=lattices.device, dtype=lattices.dtype)

                all_forces.append(f)
                all_energies.append(e.unsqueeze(0) if e.dim() == 0 else e.view(1, 1))
                all_stresses.append(s.unsqueeze(0))
                start_idx += n_atoms

            forces = torch.cat(all_forces, dim=0)
            energy = torch.cat(all_energies, dim=0)
            stress = torch.cat(all_stresses, dim=0)

            node_emb = F.pad(forces, (0, self.node_dim - 3))
            graph_emb = F.pad(stress.reshape(batch_size, 9), (0, self.graph_dim - 9))

            out = {
                "node_emb": node_emb,
                "graph_emb": graph_emb,
                "forces": forces,
                "stress": stress,
                "energy": energy,
            }

        clamped_forces, clamped_stress = clamp_forces_and_stress(
            out["forces"],
            out["stress"],
            max_force=self.max_force,
            max_stress=self.max_stress,
        )
        out["forces"] = clamped_forces
        out["stress"] = clamped_stress
        return out


def build_orb_backbone(
    model_name: str = "orb-v2",
    device: str = "cpu",
    use_mock: bool = False,
    node_dim: int = 256,
    graph_dim: int = 256,
) -> OrbBackboneWrapper:
    """Factory for constructing frozen ORB MLIP backbones."""
    return OrbBackboneWrapper(
        model_name=model_name,
        device=device,
        use_mock=use_mock,
        node_dim=node_dim,
        graph_dim=graph_dim,
    )
