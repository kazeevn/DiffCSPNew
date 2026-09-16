"""GeoOrbCSPNet: Deep MLIP-Conditioned Geometry-Aware CSPNet.

Integrates frozen ORB MLIP representations, Cartesian forces, and virial stresses
with full-cell geometric message passing, distance-gated attention, directional
bond force readout, and Lie-algebra stabilizer tangent space projections.

Trainable parameter budget: Strictly <= 12,283,784 parameters.
"""

from typing import Any
import torch
import torch.nn as nn

from diffcsp.core.crystal_family import CrystalFamily
from diffcsp.core.scatter import scatter
from diffcsp.data.transforms import cart_forces_to_frac_forces, frac_to_cart_coords
from diffcsp.models.geo_v2_cspnet import GeoV2CSPLayer
from diffcsp.models.layers import generate_intra_crystal_edges
from diffcsp.models.orb_wrapper import OrbBackboneWrapper, build_orb_backbone

MAX_ATOMIC_NUM = 100


class GeoOrbCSPNet(nn.Module):
    """Deep MLIP-Conditioned Geometry-Aware CSPNet.

    Architecture highlights:
    1. Frozen ORB Backbone: evaluates interatomic potential (orb-v3), providing
       frozen atomic representations (256-d), Cartesian forces (3-d), and virial stresses (3x3).
       All backbone parameters are frozen (requires_grad = False).
    2. Rich Crystallographic & MLIP Node Conditioning:
       Concatenates element embeddings (384-d), space group (64-d), multiplicity (32-d),
       site DoF (32-d), ORB node features (projected from 256 to 512), and ORB forces
       (projected from 6 to 512), fused into hidden_dim=512 alongside timestep embeddings.
    3. Geometric Message Passing: 6 GeoV2CSPLayer layers with distance-gated attention,
       Bessel radial basis functions (r_cut=8.0 A, 32 freqs), unit bond vectors, and Fourier sinusoids.
    4. Coordinate Readout: Directional bond force projection along r_hat @ L^{-1} combined
       with a 2-layer MLP coordinate head and site stabilizer tangent space projection P_j.
    5. Lattice Readout: Dual graph pooling (mean + max) combined with current lattice
       representation, timestep embedding, and ORB graph representation / stress, projected
       to 6 Lie algebra parameters and projected to the crystal family space group.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        latent_dim: int = 256,
        max_atoms: int = 100,
        act_fn: str = "silu",
        num_freqs: int = 120,
        num_rbf: int = 32,
        r_cut: float = 8.0,
        edge_dim: int | None = None,
        ln: bool = False,
        dense: bool = False,
        force_dim: int = 6,
        orb_backbone: OrbBackboneWrapper | None = None,
        orb_model_name: str = "orb-v3",
        orb_node_dim: int = 256,
        orb_graph_dim: int = 256,
        use_mock_orb: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms
        self.force_dim = force_dim
        self.ln = ln
        self.dense = dense
        self.act_fn = nn.SiLU() if act_fn == "silu" else nn.ReLU()
        self.crystal_family = CrystalFamily()

        # 1. Frozen ORB Backbone (all parameters requires_grad = False)
        if orb_backbone is None:
            self.orb_backbone = build_orb_backbone(
                model_name=orb_model_name,
                device=device,
                use_mock=use_mock_orb,
                node_dim=orb_node_dim,
                graph_dim=orb_graph_dim,
            )
        else:
            self.orb_backbone = orb_backbone

        self.orb_backbone.eval()
        for p in self.orb_backbone.parameters():
            p.requires_grad = False

        # 2. Crystallographic Embeddings
        # element (384) + spacegroup (64) + multiplicity (32) + DoF (32) = 512
        self.element_emb = nn.Embedding(max_atoms + 1, 384)
        self.spacegroup_emb = nn.Embedding(231, 64)
        self.multiplicity_emb = nn.Embedding(193, 32)
        self.dof_emb = nn.Embedding(4, 32)

        # 3. ORB Conditioning Projections
        # Projects frozen ORB representations (256) and forces (6) to hidden_dim (512)
        self.orb_node_proj = nn.Linear(orb_node_dim, hidden_dim)
        self.force_proj = nn.Linear(force_dim, hidden_dim)

        # 4. Node Conditioning Fusion
        # Fuses [crys_feat (512), orb_node_feat (512), force_feat (512), t_per_atom (256)] -> hidden_dim (512)
        node_in_dim = 384 + 64 + 32 + 32 + hidden_dim + hidden_dim + latent_dim
        self.node_fusion = nn.Sequential(
            nn.Linear(node_in_dim, 256),
            self.act_fn,
            nn.Linear(256, hidden_dim),
        )

        # 5. Message Passing Layers
        # Default edge_dim=478 matches parameter budget <= 12,283,784 strictly (within 0.2% of 12.28M)
        if edge_dim is None:
            edge_dim = 478 if hidden_dim == 512 else hidden_dim
        self.edge_dim = edge_dim

        self.csp_layers = nn.ModuleList(
            [
                GeoV2CSPLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    act_fn=act_fn,
                    num_freqs=num_freqs,
                    num_rbf=num_rbf,
                    r_cut=r_cut,
                    ln=ln,
                )
                for _ in range(num_layers)
            ]
        )

        hidden_dim_out = hidden_dim * (num_layers + 1) if self.dense else hidden_dim

        # 6. Coordinate Readout Heads
        # 2-layer MLP coordinate head on node features
        self.coord_node_mlp = nn.Sequential(
            nn.Linear(hidden_dim_out, 64),
            self.act_fn,
            nn.Linear(64, 3, bias=False),
        )

        # Directional bond projection readout from final edge features
        self.bond_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            self.act_fn,
            nn.Linear(32, 1, bias=False),
        )

        # 7. Lattice Readout Head
        # Graph pooling (mean 512 + max 512) + lattice_rep (6) + time (256) + ORB graph (256) + stress (9) = 1551
        lat_in_dim = hidden_dim * 2 + 6 + latent_dim + orb_graph_dim + 9
        self.lattice_out = nn.Sequential(
            nn.Linear(lat_in_dim, 64),
            self.act_fn,
            nn.Linear(64, 6, bias=False),
        )

    def forward(
        self,
        t: torch.Tensor,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor,
        spacegroups: torch.Tensor | None = None,
        multiplicities: torch.Tensor | None = None,
        dofs: torch.Tensor | None = None,
        site_projectors: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward denoising step conditioned on frozen ORB MLIP potential.

        Args:
            t: (B, latent_dim) timestep embeddings.
            atom_types: (N,) atomic numbers.
            frac_coords: (N, 3) fractional atomic coordinates in [0, 1).
            lattices: (B, 6) crystal family representation or (B, 3, 3) lattice matrices.
            num_atoms: (B,) atom counts per structure.
            node2graph: (N,) graph membership index.
            spacegroups: (B,) or (N,) space group numbers (1..230).
            multiplicities: (N,) Wyckoff orbit multiplicities (1..192).
            dofs: (N,) site degrees of freedom (0..3).
            site_projectors: Optional (N, 3, 3) site stabilizer tangent space projectors P_j.

        Returns:
            Tuple of (lattice_out, coord_out) where lattice_out is (B, 6) and coord_out is (N, 3).
        """
        B = num_atoms.shape[0]
        N = atom_types.shape[0]

        if lattices.dim() == 2 and lattices.shape[-1] == 6:
            lattice_rep = lattices
            lattice_mat = self.crystal_family.v2m(lattices)
        elif lattices.dim() == 3 and lattices.shape[-2:] == (3, 3):
            lattice_mat = lattices
            lattice_rep = self.crystal_family.m2v(lattices)
        else:
            raise ValueError(f"Expected lattices to be (B, 6) or (B, 3, 3), got {lattices.shape}")

        if spacegroups is None:
            spacegroups = torch.ones(B, dtype=torch.long, device=atom_types.device)
        if multiplicities is None:
            multiplicities = torch.ones(N, dtype=torch.long, device=atom_types.device)
        if dofs is None:
            dofs = torch.full((N,), 3, dtype=torch.long, device=atom_types.device)

        if spacegroups.shape[0] == B:
            sg_per_atom = spacegroups[node2graph]
        else:
            sg_per_atom = spacegroups

        # 1. Evaluate frozen ORB backbone in Cartesian space
        cart_coords = frac_to_cart_coords(frac_coords, lattice_mat, node2graph)
        self.orb_backbone.eval()
        with torch.no_grad():
            orb_out = self.orb_backbone(
                atom_types=atom_types,
                cart_coords=cart_coords,
                lattices=lattice_mat,
                num_atoms=num_atoms,
                node2graph=node2graph,
            )

        f_cart = torch.nan_to_num(orb_out["forces"])
        f_frac = torch.nan_to_num(cart_forces_to_frac_forces(f_cart, lattice_mat, node2graph))
        stress = torch.nan_to_num(orb_out["stress"])
        orb_node_emb = torch.nan_to_num(orb_out["node_emb"])
        orb_graph_emb = torch.nan_to_num(orb_out["graph_emb"])

        # 2. Crystallographic embeddings
        z_feat = self.element_emb(atom_types.clamp(0, self.max_atoms))
        sg_feat = self.spacegroup_emb(sg_per_atom.clamp(0, 230))
        mult_feat = self.multiplicity_emb(multiplicities.clamp(0, 192))
        dof_feat = self.dof_emb(dofs.clamp(0, 3))

        # 3. Projected ORB node features and forces
        orb_node_feat = self.act_fn(self.orb_node_proj(orb_node_emb))
        if self.force_dim == 6:
            force_in = torch.cat([f_cart, f_frac], dim=-1)
        else:
            force_in = f_cart
        force_feat = self.act_fn(self.force_proj(force_in))

        # 4. Fused node conditioning with timestep embedding
        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_in = torch.cat(
            [z_feat, sg_feat, mult_feat, dof_feat, orb_node_feat, force_feat, t_per_atom],
            dim=-1,
        )
        node_features = self.node_fusion(node_in)

        # 5. Build intra-crystal edges and message passing
        edges, frac_diff = generate_intra_crystal_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]

        h_list = [node_features]
        final_edge_feat = None
        final_unit_r = None

        for i, layer in enumerate(self.csp_layers):
            node_features, edge_feat, unit_r = layer(
                node_features,
                frac_coords,
                lattice_mat,
                lattice_rep,
                edges,
                edge2graph,
                frac_diff=frac_diff,
            )
            if i != self.num_layers - 1:
                h_list.append(node_features)
            else:
                final_edge_feat = edge_feat
                final_unit_r = unit_r

        h_list.append(node_features)
        if self.dense:
            node_features = torch.cat(h_list, dim=-1)

        # 6. Directional bond projection readout + 2-layer MLP coordinate head
        u_node = self.coord_node_mlp(node_features)  # (N, 3)

        f_ij = self.bond_mlp(final_edge_feat)  # (E, 1)
        L_inv = torch.linalg.pinv(lattice_mat)  # (B, 3, 3)
        L_inv_edges = L_inv[edge2graph]  # (E, 3, 3)
        u_bond_edges = torch.einsum("ei, eij -> ej", f_ij * final_unit_r, L_inv_edges)  # (E, 3)
        u_bond = scatter(u_bond_edges, edges[0], dim=0, dim_size=N, reduce="sum")  # (N, 3)

        u_coord = u_node + u_bond

        # Project onto Lie-algebra stabilizer tangent space P_j
        if site_projectors is not None:
            coord_out = (site_projectors @ u_coord.unsqueeze(-1)).squeeze(-1)
        else:
            coord_out = u_coord

        # 7. Lattice prediction head with dual graph pooling and spacegroup projection
        pooled_mean = scatter(node_features, node2graph, dim=0, dim_size=B, reduce="mean")
        pooled_max = scatter(node_features, node2graph, dim=0, dim_size=B, reduce="max")
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)

        stress_flat = stress.reshape(B, 9)
        lattice_in = torch.cat([pooled, lattice_rep, t, orb_graph_emb, stress_flat], dim=-1)
        v_lattice = self.lattice_out(lattice_in)
        lattice_out = self.crystal_family.proj_k_to_spacegroup(v_lattice, spacegroups)

        return lattice_out, coord_out
