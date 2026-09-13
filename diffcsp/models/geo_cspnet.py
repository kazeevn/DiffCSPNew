"""GeoCSPNet: Geometry-aware, symmetry-projected Graph Neural Network for crystals.

DiffCSP-Geo retains the full-cell graph (N atoms) to preserve hidden state
diversity and all-to-all Fourier torus coverage, while resolving DiffCSP++
limitations through:
1. Physical Euclidean distances, Bessel RBFs, and Cartesian direction vectors.
2. Lie-algebra Wyckoff tangent-space projection on coordinate heads and updates.
3. Rich crystallographic conditioning: element Z, spacegroup G, multiplicity m, DoF.
4. Direct conditioning of the lattice prediction head on graph, lattice, and time.
"""

from typing import Any
import math
import torch
import torch.nn as nn

from diffcsp.core.crystal_family import CrystalFamily
from diffcsp.core.scatter import scatter
from diffcsp.models.layers import SinusoidsEmbedding, generate_intra_crystal_edges
from diffcsp.models.painn_layers import bessel_rbf

MAX_ATOMIC_NUM = 100


class GeoCSPLayer(nn.Module):
    """Geometry-aware message passing layer incorporating Euclidean and torus representations."""

    def __init__(
        self,
        hidden_dim: int = 512,
        edge_dim: int | None = None,
        act_fn: str = "silu",
        num_freqs: int = 120,
        num_rbf: int = 32,
        r_cut: float = 8.0,
        ln: bool = False,
        ip: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim if edge_dim is not None else hidden_dim
        self.num_freqs = num_freqs
        self.num_rbf = num_rbf
        self.r_cut = r_cut
        self.ln = ln
        self.ip = ip

        act = nn.SiLU() if act_fn == "silu" else nn.ReLU()
        self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs, n_space=3)

        # Edge features:
        # h_i (hidden_dim) + h_j (hidden_dim) + lattice_rep (6) +
        # frac_diff_sin (720) + bessel_rbf (32) + unit_r (3) = 2 * hidden_dim + 761
        in_edge_dim = hidden_dim * 2 + 6 + self.dis_emb.dim + num_rbf + 3

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_dim, self.edge_dim),
            act,
            nn.Linear(self.edge_dim, hidden_dim),
            act,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
        )
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice_mat: torch.Tensor,
        lattice_rep: torch.Tensor,
        edge_index: torch.Tensor,
        edge2graph: torch.Tensor,
        frac_diff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Residual geometric message passing step.

        Args:
            node_features: (N, hidden_dim) node embeddings.
            frac_coords: (N, 3) fractional coordinates in [0, 1).
            lattice_mat: (B, 3, 3) Cartesian lattice matrices.
            lattice_rep: (B, 6) Lie-algebra crystal family lattice representation.
            edge_index: (2, E) directed graph edges (source i -> target j).
            edge2graph: (E,) mapping each edge to its crystal index in batch.
            frac_diff: Optional precomputed periodic fractional difference (xj - xi) % 1.0.

        Returns:
            Updated (N, hidden_dim) node feature tensor.
        """
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)

        hi = node_features[edge_index[0]]
        hj = node_features[edge_index[1]]

        if frac_diff is None:
            xi = frac_coords[edge_index[0]]
            xj = frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0

        # Minimum image convention displacement on periodic torus
        frac_diff_mic = (frac_diff + 0.5) % 1.0 - 0.5

        # Real Cartesian displacement vector r_ij = Delta x_mic @ L
        L_edges = lattice_mat[edge2graph]  # (E, 3, 3)
        r_ij = torch.einsum("ei, eij -> ej", frac_diff_mic, L_edges)  # (E, 3)

        # Distance and unit direction vector
        d_ij = torch.norm(r_ij, dim=-1, keepdim=True)  # (E, 1)
        unit_r = r_ij / torch.clamp(d_ij, min=1e-4)  # (E, 3)

        # Radial basis expansion and fractional sinusoids
        rbf_feat = bessel_rbf(d_ij, num_rbf=self.num_rbf, r_cut=self.r_cut)  # (E, 32)
        sin_feat = self.dis_emb(frac_diff)  # (E, 720)
        lat_edges = lattice_rep[edge2graph]  # (E, 6)

        edge_in = torch.cat([hi, hj, lat_edges, sin_feat, rbf_feat, unit_r], dim=-1)
        edge_feat = self.edge_mlp(edge_in)

        agg = scatter(edge_feat, edge_index[0], dim=0, dim_size=node_features.shape[0], reduce="mean")
        node_in = torch.cat([node_features, agg], dim=-1)
        node_out = self.node_mlp(node_in)

        return node_input + node_out


class GeoCSPNet(nn.Module):
    """Geometry-aware CSPNet with full-cell representation and crystallographic conditioning."""

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
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.ln = ln
        self.dense = dense
        self.pooling = pooling

        # Rich Crystallographic Conditioning:
        # Atomic number Z: 384
        # Space group G: 64
        # Multiplicity m: 32
        # Site DoF: 32
        # Total = 384 + 64 + 32 + 32 = 512
        self.element_emb = nn.Embedding(max_atoms + 1, 384)
        self.spacegroup_emb = nn.Embedding(231, 64)
        self.multiplicity_emb = nn.Embedding(193, 32)
        self.dof_emb = nn.Embedding(4, 32)

        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        # Default edge_dim=486 for hidden_dim=512 matches CSPNet's 12.28M parameters within 0.05%
        if edge_dim is None:
            edge_dim = 486 if hidden_dim == 512 else hidden_dim

        self.csp_layers = nn.ModuleList(
            [
                GeoCSPLayer(
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

        self.coord_out = nn.Linear(hidden_dim_out, 3, bias=False)
        self.lattice_out = nn.Sequential(
            nn.Linear(hidden_dim_out + 6 + latent_dim, hidden_dim),
            nn.SiLU() if act_fn == "silu" else nn.ReLU(),
            nn.Linear(hidden_dim, 6, bias=False),
        )

        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.crystal_family = CrystalFamily()

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
        """Forward denoising step.

        Args:
            t: (B, latent_dim) timestep embeddings.
            atom_types: (N,) atomic numbers.
            frac_coords: (N, 3) fractional atomic coordinates.
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

        # Crystallographic feature embedding
        z_feat = self.element_emb(atom_types.clamp(0, MAX_ATOMIC_NUM))
        sg_feat = self.spacegroup_emb(sg_per_atom.clamp(0, 230))
        mult_feat = self.multiplicity_emb(multiplicities.clamp(0, 192))
        dof_feat = self.dof_emb(dofs.clamp(0, 3))

        node_features = torch.cat([z_feat, sg_feat, mult_feat, dof_feat], dim=-1)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=-1)
        node_features = self.atom_latent_emb(node_features)

        edges, frac_diff = generate_intra_crystal_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]

        h_list = [node_features]
        for i, layer in enumerate(self.csp_layers):
            node_features = layer(
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

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        h_list.append(node_features)
        if self.dense:
            node_features = torch.cat(h_list, dim=-1)

        graph_features = scatter(
            node_features, node2graph, dim=0, dim_size=B, reduce=self.pooling
        )

        coord_out = self.coord_out(node_features)
        if site_projectors is not None:
            coord_out = (site_projectors @ coord_out.unsqueeze(-1)).squeeze(-1)

        lattice_in = torch.cat([graph_features, lattice_rep, t], dim=-1)
        lattice_out = self.lattice_out(lattice_in)

        return lattice_out, coord_out
