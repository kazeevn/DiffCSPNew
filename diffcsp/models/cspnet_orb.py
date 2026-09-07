"""CSPNetORB: Lightweight adapter head on top of frozen ORB MLIP potential."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffcsp.core.crystal_family import CrystalFamily
from diffcsp.core.scatter import scatter
from diffcsp.data.transforms import cart_forces_to_frac_forces, frac_to_cart_coords
from diffcsp.models.layers import CSPLayer, SinusoidsEmbedding, generate_intra_crystal_edges
from diffcsp.models.orb_wrapper import OrbBackboneWrapper, build_orb_backbone


class CSPNetORB(nn.Module):
    """Lightweight Denoising Adapter on top of a frozen ORB MLIP Backbone.

    Enforces by design that the coordinate denoising step can be zero
    ONLY IF the ORB forces are zero: S = 0 ==> F = 0.
    """

    def __init__(
        self,
        orb_backbone: OrbBackboneWrapper | None = None,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_layers: int = 2,
        max_atoms: int = 100,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        num_freqs: int = 64,
        pooling: str = "mean",
        use_force_residual: bool = True,
        enforce_zero_force_condition: bool = True,
        enforce_zero_stress_condition: bool = False,
        gamma_min: float = 1e-3,
        orb_node_dim: int = 256,
        orb_graph_dim: int = 256,
        device: str = "cpu",
        use_mock: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.pooling = pooling
        self.use_force_residual = use_force_residual
        self.enforce_zero_force_condition = enforce_zero_force_condition
        self.enforce_zero_stress_condition = enforce_zero_stress_condition
        self.gamma_min = gamma_min
        self.crystal_family = CrystalFamily()

        # 1. Frozen ORB Backbone
        if orb_backbone is None:
            self.orb_backbone = build_orb_backbone(
                model_name="orb-v2",
                device=device,
                use_mock=use_mock,
                node_dim=orb_node_dim,
                graph_dim=orb_graph_dim,
            )
        else:
            self.orb_backbone = orb_backbone

        self.orb_backbone.eval()
        for p in self.orb_backbone.parameters():
            p.requires_grad = False

        # 2. Embedding & Projections
        self.node_embedding = nn.Embedding(max_atoms + 1, hidden_dim)
        self.act_fn = nn.SiLU() if act_fn == "silu" else nn.ReLU()
        self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs) if dis_emb == "sin" else None

        self.orb_node_proj = nn.Linear(orb_node_dim, hidden_dim)
        self.orb_graph_proj = nn.Linear(orb_graph_dim, hidden_dim)
        self.force_proj = nn.Linear(6, hidden_dim)
        self.stress_proj = nn.Linear(9, hidden_dim)

        combined_dim = hidden_dim * 3 + latent_dim
        self.input_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            self.act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. Lightweight Message Passing Adapter
        self.csp_layers = nn.ModuleList(
            [
                CSPLayer(
                    hidden_dim=hidden_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    ln=True,
                    ip=True,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output Heads
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)

        if self.enforce_zero_force_condition:
            self.gamma_head = nn.Sequential(
                nn.Linear(latent_dim + hidden_dim, 64),
                self.act_fn,
                nn.Linear(64, 1),
            )
        elif self.use_force_residual:
            self.time_gate = nn.Sequential(
                nn.Linear(latent_dim, 64),
                self.act_fn,
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        lattice_in_dim = hidden_dim * 3 + 6 + latent_dim
        self.lattice_mlp = nn.Sequential(
            nn.Linear(lattice_in_dim, hidden_dim),
            self.act_fn,
            nn.Linear(hidden_dim, 6, bias=False),
        )

        if self.enforce_zero_stress_condition:
            self.gamma_stress_head = nn.Sequential(
                nn.Linear(latent_dim + hidden_dim, 64),
                self.act_fn,
                nn.Linear(64, 1),
            )

    def apply_zero_force_constraint(
        self,
        v_coord: torch.Tensor,
        f_frac: torch.Tensor,
        t_per_atom: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """Enforces S = 0 ==> f_frac = 0 by decomposing into parallel and perpendicular components."""
        f_norm = torch.norm(f_frac, dim=-1, keepdim=True)
        has_force = (f_norm > 1e-7).float()
        f_unit = f_frac / (f_norm + 1e-8)

        v_parallel = torch.sum(v_coord * f_unit, dim=-1, keepdim=True) * f_unit * has_force
        v_perp = v_coord - v_parallel

        gamma_raw = self.gamma_head(torch.cat([t_per_atom, node_features], dim=-1))
        gamma = self.gamma_min + F.softplus(gamma_raw)

        return gamma * f_frac + v_perp

    def apply_zero_stress_constraint(
        self,
        v_lattice: torch.Tensor,
        stress: torch.Tensor,
        t: torch.Tensor,
        pooled_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Analogous zero-stress condition for the lattice update."""
        s_vec = torch.stack(
            [
                stress[:, 0, 0],
                stress[:, 1, 1],
                stress[:, 2, 2],
                stress[:, 1, 2],
                stress[:, 0, 2],
                stress[:, 0, 1],
            ],
            dim=-1,
        )

        s_norm = torch.norm(s_vec, dim=-1, keepdim=True)
        has_stress = (s_norm > 1e-7).float()
        s_unit = s_vec / (s_norm + 1e-8)

        v_parallel = torch.sum(v_lattice * s_unit, dim=-1, keepdim=True) * s_unit * has_stress
        v_perp = v_lattice - v_parallel

        gamma_raw = self.gamma_stress_head(torch.cat([t, pooled_nodes], dim=-1))
        gamma = self.gamma_min + F.softplus(gamma_raw)

        return gamma * s_vec + v_perp

    def forward(
        self,
        t: torch.Tensor,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        crys_fam: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapter forward pass."""
        batch_size = crys_fam.shape[0]

        lattices = self.crystal_family.v2m(crys_fam)
        cart_coords = frac_to_cart_coords(frac_coords, lattices, node2graph)

        self.orb_backbone.eval()
        with torch.no_grad():
            orb_out = self.orb_backbone(
                atom_types=atom_types,
                cart_coords=cart_coords,
                lattices=lattices,
                num_atoms=num_atoms,
                node2graph=node2graph,
            )

        f_cart = orb_out["forces"]
        f_frac = cart_forces_to_frac_forces(f_cart, lattices, node2graph)
        stress = orb_out["stress"]
        orb_node_emb = orb_out["node_emb"]
        orb_graph_emb = orb_out["graph_emb"]

        atom_feat = self.node_embedding(atom_types.clamp(0, 100))
        t_per_atom = t.repeat_interleave(num_atoms, dim=0)

        orb_node_feat = self.act_fn(self.orb_node_proj(orb_node_emb))
        force_feat = self.act_fn(self.force_proj(torch.cat([f_cart, f_frac], dim=-1)))

        fused_in = torch.cat([atom_feat, t_per_atom, orb_node_feat, force_feat], dim=-1)
        node_features = self.input_fusion(fused_in)

        edges, frac_diff = generate_intra_crystal_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]

        for layer in self.csp_layers:
            node_features = layer(
                node_features=node_features,
                frac_coords=frac_coords,
                lattices=crys_fam,
                edge_index=edges,
                edge2graph=edge2graph,
                frac_diff=frac_diff,
            )

        v_coord = self.coord_out(node_features)

        if self.enforce_zero_force_condition:
            coord_score = self.apply_zero_force_constraint(
                v_coord=v_coord,
                f_frac=f_frac,
                t_per_atom=t_per_atom,
                node_features=node_features,
            )
        elif self.use_force_residual:
            gate = self.time_gate(t).repeat_interleave(num_atoms, dim=0)
            coord_score = v_coord + gate * f_frac
        else:
            coord_score = v_coord

        pooled_nodes = scatter(node_features, node2graph, dim=0, dim_size=batch_size, reduce=self.pooling)
        orb_graph_feat = self.act_fn(self.orb_graph_proj(orb_graph_emb))
        stress_flat = stress.reshape(batch_size, 9)
        stress_feat = self.act_fn(self.stress_proj(stress_flat))

        lattice_in = torch.cat([pooled_nodes, orb_graph_feat, stress_feat, crys_fam, t], dim=-1)
        v_lattice = self.lattice_mlp(lattice_in)

        if self.enforce_zero_stress_condition:
            lattice_noise = self.apply_zero_stress_constraint(
                v_lattice=v_lattice,
                stress=stress,
                t=t,
                pooled_nodes=pooled_nodes,
            )
        else:
            lattice_noise = v_lattice

        return lattice_noise, coord_score
