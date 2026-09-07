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
from diffcsp.models.repulsion import compute_crystal_repulsion

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
        max_force: float = 2.0,
        max_stress: float = 0.05,
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
            from orb_models.common.atoms.batch.graph_batch import AtomGraphs

            batch_size = lattices.shape[0]
            start_idx = 0

            rep_forces = []
            rep_energies = []
            rep_stresses = []
            valid_graphs = []
            valid_batch_indices = []
            valid_atom_slices = []
            current_valid_atom_offset = 0

            for b in range(batch_size):
                n_atoms = int(num_atoms[b].item())
                sub_coords_t = cart_coords[start_idx : start_idx + n_atoms]
                sub_lat_t = lattices[b]
                sub_types_t = atom_types[start_idx : start_idx + n_atoms]

                # Compute smooth conservative repulsive potential
                f_rep, s_rep, e_rep = compute_crystal_repulsion(
                    cart_coords=sub_coords_t,
                    lattice=sub_lat_t,
                    atom_types=sub_types_t,
                    f_max=self.max_force,
                    eta=0.70,
                )
                rep_forces.append(f_rep)
                rep_stresses.append(s_rep)
                rep_energies.append(e_rep.view(1, 1))

                sub_atom_types = sub_types_t.detach().cpu().numpy()
                sub_cart_coords = sub_coords_t.detach().cpu().numpy()
                sub_cell = sub_lat_t.detach().cpu().numpy()

                try:
                    atoms = Atoms(numbers=sub_atom_types, positions=sub_cart_coords, cell=sub_cell, pbc=True)
                    # Use CPU for neighbor search to avoid unsupported CUDA warp / triton kernels on Kepler
                    graph = self.atoms_adapter.from_ase_atoms(atoms, device="cpu")
                    valid_graphs.append(graph)
                    valid_batch_indices.append(b)
                    valid_atom_slices.append((current_valid_atom_offset, current_valid_atom_offset + n_atoms))
                    current_valid_atom_offset += n_atoms
                except Exception as exc:
                    logger.debug(
                        "ORB graph construction failed for structure %d (%s). Using smooth conservative repulsive fallback.", b, exc
                    )

                start_idx += n_atoms

            final_forces = list(rep_forces)
            final_stresses = list(rep_stresses)
            final_energies = list(rep_energies)

            if len(valid_graphs) > 0:
                try:
                    batched_graph = AtomGraphs.batch(valid_graphs)
                    if str(self.device) != "cpu":
                        batched_graph = batched_graph.to(self.device)
                    pred = self.orb_model.predict(batched_graph)

                    b_forces = torch.as_tensor(pred["forces"], device=cart_coords.device, dtype=cart_coords.dtype)
                    b_energies = torch.as_tensor(pred["energy"], device=cart_coords.device, dtype=cart_coords.dtype)
                    b_stresses = torch.as_tensor(
                        pred.get("stress", torch.zeros((len(valid_graphs), 6))),
                        device=lattices.device,
                        dtype=lattices.dtype,
                    )

                    # Convert Voigt stress (N, 6) [xx, yy, zz, yz, xz, xy] to (N, 3, 3) Cauchy tensor
                    if b_stresses.dim() == 2 and b_stresses.shape[-1] == 6:
                        idx_map = torch.tensor(
                            [[0, 5, 4],
                             [5, 1, 3],
                             [4, 3, 2]],
                            device=b_stresses.device,
                            dtype=torch.long,
                        )
                        b_stresses = b_stresses[:, idx_map]
                    elif b_stresses.dim() == 1 and b_stresses.shape[0] == 6:
                        idx_map = torch.tensor(
                            [[0, 5, 4],
                             [5, 1, 3],
                             [4, 3, 2]],
                            device=b_stresses.device,
                            dtype=torch.long,
                        )
                        b_stresses = b_stresses[idx_map].unsqueeze(0)
                    elif b_stresses.dim() == 2 and b_stresses.shape[-1] == 9:
                        b_stresses = b_stresses.reshape(-1, 3, 3)

                    # Augment valid predictions with repulsive core
                    for v_idx, b in enumerate(valid_batch_indices):
                        a_start, a_end = valid_atom_slices[v_idx]
                        f_orb = b_forces[a_start:a_end]
                        s_orb = b_stresses[v_idx]
                        e_orb = b_energies[v_idx]

                        final_forces[b] = f_orb + rep_forces[b]
                        final_stresses[b] = s_orb + rep_stresses[b]
                        final_energies[b] = e_orb.view(1, 1) + rep_energies[b]
                except Exception as exc:
                    logger.warning(
                        "Batched ORB prediction failed (%s). Falling back to repulsive potential.", exc
                    )

            forces = torch.cat(final_forces, dim=0)
            stress = torch.stack(final_stresses, dim=0)
            energy = torch.cat(final_energies, dim=0)

            clamped_forces, clamped_stress = clamp_forces_and_stress(
                forces,
                stress,
                max_force=self.max_force,
                max_stress=self.max_stress,
            )

            node_emb = F.pad(clamped_forces, (0, self.node_dim - 3))
            graph_emb = F.pad(clamped_stress.reshape(batch_size, 9), (0, self.graph_dim - 9))

            out = {
                "node_emb": node_emb,
                "graph_emb": graph_emb,
                "forces": clamped_forces,
                "stress": clamped_stress,
                "energy": energy,
            }
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
