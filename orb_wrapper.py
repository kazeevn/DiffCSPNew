import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


def frac_to_cart_coords(frac_coords: torch.Tensor, lattices: torch.Tensor, node2graph: torch.Tensor) -> torch.Tensor:
    """
    Convert fractional coordinates to Cartesian coordinates.
    
    Args:
        frac_coords: (N, 3) fractional coordinates in [0, 1)
        lattices: (B, 3, 3) lattice matrices
        node2graph: (N,) mapping from atom index to graph/crystal index
        
    Returns:
        cart_coords: (N, 3) Cartesian coordinates in Angstroms
    """
    lattices_per_atom = lattices[node2graph]  # (N, 3, 3)
    cart_coords = torch.einsum('ni, nij -> nj', frac_coords, lattices_per_atom)
    return cart_coords


def cart_forces_to_frac_forces(cart_forces: torch.Tensor, lattices: torch.Tensor, node2graph: torch.Tensor) -> torch.Tensor:
    """
    Convert Cartesian forces to fractional coordinates forces.
    Since R = x @ L, dR/dx = L, so dE/dx = (dE/dR) @ L = -F_cart @ L.
    Hence F_frac = -dE/dx = F_cart @ L.
    
    Args:
        cart_forces: (N, 3) Cartesian forces (eV/A)
        lattices: (B, 3, 3) lattice matrices
        node2graph: (N,) mapping from atom index to graph/crystal index
        
    Returns:
        frac_forces: (N, 3) forces in fractional coordinate space
    """
    lattices_per_atom = lattices[node2graph]  # (N, 3, 3)
    frac_forces = torch.einsum('ni, nij -> nj', cart_forces, lattices_per_atom)
    return frac_forces


def clamp_forces_and_stress(
    cart_forces: torch.Tensor,
    stress: torch.Tensor,
    max_force: float = 20.0,
    max_stress: float = 50.0
):
    """
    Smoothly clamp forces and stresses to prevent numerical divergence
    during high-noise diffusion steps where atoms may artificially overlap.
    Uses soft tanh scaling to preserve vector direction while bounding magnitude.
    
    Args:
        cart_forces: (N, 3) Cartesian forces
        stress: (B, 3, 3) Cauchy stress tensor
        max_force: maximum force norm threshold (eV/A)
        max_stress: maximum stress norm threshold (GPa or eV/A^3)
        
    Returns:
        clamped_forces: (N, 3)
        clamped_stress: (B, 3, 3)
    """
    f_norm = torch.norm(cart_forces, dim=-1, keepdim=True) + 1e-8
    clamped_forces = cart_forces * (max_force * torch.tanh(f_norm / max_force) / f_norm)
    
    s_norm = torch.norm(stress, dim=(-2, -1), keepdim=True) + 1e-8
    clamped_stress = stress * (max_stress * torch.tanh(s_norm / max_stress) / s_norm)
    
    return clamped_forces, clamped_stress


class MockOrbBackbone(nn.Module):
    """
    A lightweight mock backbone matching ORB's output signatures.
    Used for testing, local development on machines without GPU / large weights,
    and verifying the end-to-end diffusion pipeline.
    """
    def __init__(self, node_dim: int = 256, graph_dim: int = 256, max_atomic_num: int = 100):
        super().__init__()
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        self.atom_embedding = nn.Embedding(max_atomic_num + 1, node_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim + 3, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(node_dim + 9, graph_dim),
            nn.SiLU(),
            nn.Linear(graph_dim, graph_dim)
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
        node2graph: torch.Tensor
    ):
        """
        Mock forward pass returning synthetic features and outputs.
        """
        # Node embeddings
        atom_h = self.atom_embedding(atom_types.clamp(0, 100))
        node_in = torch.cat([atom_h, cart_coords], dim=-1)
        node_emb = self.node_proj(node_in)
        
        # Graph pooling
        from torch_scatter import scatter
        pooled_nodes = scatter(node_emb, node2graph, dim=0, reduce='mean', dim_size=lattices.shape[0])
        lattices_flat = lattices.reshape(lattices.shape[0], 9)
        graph_in = torch.cat([pooled_nodes, lattices_flat], dim=-1)
        graph_emb = self.graph_proj(graph_in)
        
        # Predicted forces, stress, energy
        forces = self.force_head(node_emb)
        stress = self.stress_head(graph_emb).reshape(-1, 3, 3)
        # Symmetrize stress tensor
        stress = 0.5 * (stress + stress.transpose(-1, -2))
        energy = self.energy_head(graph_emb)
        
        return {
            'node_emb': node_emb,
            'graph_emb': graph_emb,
            'forces': forces,
            'stress': stress,
            'energy': energy
        }


class OrbBackboneWrapper(nn.Module):
    """
    Wrapper around the pretrained ORB MLIP model from `orb_models`.
    Extracts frozen node embeddings, graph features, Cartesian forces, and stress.
    Automatically falls back to MockOrbBackbone if orb_models is not installed.
    """
    def __init__(
        self,
        model_name: str = "orb-v2",
        device: str = "cpu",
        use_mock: bool = False,
        node_dim: int = 256,
        graph_dim: int = 256,
        max_force: float = 20.0,
        max_stress: float = 50.0
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        self.max_force = max_force
        self.max_stress = max_stress
        self.use_mock = use_mock
        
        if not use_mock:
            try:
                # Attempt to load orb_models
                import orb_models
                from orb_models.forcefield import pretrained
                
                # Check for available pretrained constructor
                if hasattr(pretrained, model_name):
                    loader = getattr(pretrained, model_name)
                    self.orb_model, self.atoms_adapter = loader(device=device)
                elif hasattr(pretrained, f"{model_name}_conservative_inf_omat"):
                    loader = getattr(pretrained, f"{model_name}_conservative_inf_omat")
                    self.orb_model, self.atoms_adapter = loader(device=device)
                else:
                    # Generic fallback in pretrained
                    self.orb_model, self.atoms_adapter = pretrained.orb_v2(device=device)
                    
                self.orb_model.eval()
                for p in self.orb_model.parameters():
                    p.requires_grad = False
                self.is_loaded = True
            except Exception as e:
                warnings.warn(
                    f"orb_models could not be loaded ({e}). "
                    f"Falling back to MockOrbBackbone. This enables testing and development "
                    f"without large model weights."
                )
                self.use_mock = True
                self.mock_backbone = MockOrbBackbone(node_dim=node_dim, graph_dim=graph_dim)
                self.is_loaded = False
        else:
            self.mock_backbone = MockOrbBackbone(node_dim=node_dim, graph_dim=graph_dim)
            self.is_loaded = False
            
        # Ensure all parameters in the backbone wrapper are frozen
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        atom_types: torch.Tensor,
        cart_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor
    ):
        """
        Runs frozen ORB MLIP forward pass.
        
        Returns:
            dict containing:
                node_emb: (N, node_dim)
                graph_emb: (B, graph_dim)
                forces: (N, 3) Cartesian forces (clamped)
                stress: (B, 3, 3) Cauchy stress tensor (clamped)
                energy: (B, 1) Total energy
        """
        if self.use_mock or not self.is_loaded:
            out = self.mock_backbone(atom_types, cart_coords, lattices, num_atoms, node2graph)
        else:
            # Construct input structure for ORB and run forward pass
            from ase import Atoms
            from torch_scatter import scatter
            
            batch_size = lattices.shape[0]
            start_idx = 0
            
            all_forces = []
            all_energies = []
            all_stresses = []
            
            # Forward batch through ORB
            for b in range(batch_size):
                n_atoms = num_atoms[b].item()
                sub_atom_types = atom_types[start_idx:start_idx + n_atoms].detach().cpu().numpy()
                sub_cart_coords = cart_coords[start_idx:start_idx + n_atoms].detach().cpu().numpy()
                sub_cell = lattices[b].detach().cpu().numpy()
                
                atoms = Atoms(numbers=sub_atom_types, positions=sub_cart_coords, cell=sub_cell, pbc=True)
                graph = self.atoms_adapter.from_ase_atoms(atoms, device=self.device)
                pred = self.orb_model.predict(graph)
                
                f = torch.as_tensor(pred["forces"], device=cart_coords.device, dtype=cart_coords.dtype)
                e = torch.as_tensor(pred["energy"], device=cart_coords.device, dtype=cart_coords.dtype)
                s = torch.as_tensor(pred.get("stress", torch.zeros((3, 3))), device=lattices.device, dtype=lattices.dtype)
                if s.numel() == 6:
                    # Convert 6-voigt to 3x3
                    s_mat = torch.tensor([
                        [s[0], s[5], s[4]],
                        [s[5], s[1], s[3]],
                        [s[4], s[3], s[2]]
                    ], device=lattices.device, dtype=lattices.dtype)
                    s = s_mat
                elif s.numel() == 9:
                    s = s.reshape(3, 3)
                    
                all_forces.append(f)
                all_energies.append(e.unsqueeze(0))
                all_stresses.append(s.unsqueeze(0))
                
                start_idx += n_atoms
                
            forces = torch.cat(all_forces, dim=0)
            energy = torch.cat(all_energies, dim=0)
            stress = torch.cat(all_stresses, dim=0)
            
            # Project forces and stress to representations
            node_emb = F.pad(forces, (0, self.node_dim - 3))
            graph_emb = F.pad(stress.reshape(batch_size, 9), (0, self.graph_dim - 9))
            
            out = {
                'node_emb': node_emb,
                'graph_emb': graph_emb,
                'forces': forces,
                'stress': stress,
                'energy': energy
            }

        # Apply smooth force and stress clamping
        clamped_forces, clamped_stress = clamp_forces_and_stress(
            out['forces'],
            out['stress'],
            max_force=self.max_force,
            max_stress=self.max_stress
        )
        out['forces'] = clamped_forces
        out['stress'] = clamped_stress
        return out


def build_orb_backbone(
    model_name: str = "orb-v2",
    device: str = "cpu",
    use_mock: bool = False,
    node_dim: int = 256,
    graph_dim: int = 256
) -> OrbBackboneWrapper:
    """Factory function for constructing the frozen ORB backbone wrapper."""
    return OrbBackboneWrapper(
        model_name=model_name,
        device=device,
        use_mock=use_mock,
        node_dim=node_dim,
        graph_dim=graph_dim
    )
