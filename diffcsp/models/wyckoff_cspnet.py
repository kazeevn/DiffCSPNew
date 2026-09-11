"""WyckoffCSPNet: Graph Neural Network denoiser operating strictly on the Asymmetric Unit."""

import torch
import torch.nn as nn

from diffcsp.core.scatter import scatter
from diffcsp.models.layers import SinusoidsEmbedding, WyckoffCSPLayer, generate_asymmetric_edges

MAX_ATOMIC_NUM = 100


class WyckoffCSPNet(nn.Module):
    """Deep GNN denoiser operating on K Wyckoff site anchors rather than all N atoms.

    Instead of expanding the Wyckoff gene into all N conventional unit cell atoms
    and computing an O(N^2) fully-connected intra-crystal graph, WyckoffCSPNet
    maintains graph node embeddings strictly for the K unique Wyckoff sites (the
    asymmetric unit, K in [1, 8]).

    Interactions with symmetry replicas across the unit cell are computed via
    Wyckoff-to-Wyckoff multi-edges, reducing graph size from O(N^2) to O(K * N).

    Reference:
        Innovation 1, docs/architectural-innovations.md
        SymmCD (NeurIPS 2024), SGEquiDiff (Princeton/UIUC 2024)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_layers: int = 6,
        max_atoms: int = 100,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        num_freqs: int = 128,
        ln: bool = False,
        dense: bool = False,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dense = dense
        self.ln = ln
        self.pooling = pooling

        self.node_embedding = nn.Embedding(max_atoms + 1, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        self.act_fn = nn.SiLU() if act_fn == "silu" else nn.ReLU()
        self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs) if dis_emb == "sin" else None

        self.csp_layers = nn.ModuleList(
            [WyckoffCSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln) for _ in range(num_layers)]
        )

        hidden_dim_out = hidden_dim * (num_layers + 1) if self.dense else hidden_dim
        self.coord_out = nn.Linear(hidden_dim_out, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim_out, 6, bias=False)

        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        t: torch.Tensor,
        site_atom_types: torch.Tensor,
        site_coords: torch.Tensor,
        full_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_sites: torch.Tensor,
        num_atoms: torch.Tensor,
        site2graph: torch.Tensor,
        inverse_site_map: torch.Tensor,
        target_sites: torch.Tensor | None = None,
        source_atoms: torch.Tensor | None = None,
        source_sites: torch.Tensor | None = None,
        edge2graph: torch.Tensor | None = None,
        site_projectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward denoising step on the asymmetric unit.

        Args:
            t: (B, latent_dim) timestep embeddings.
            site_atom_types: (K_total,) atomic numbers of Wyckoff site anchors.
            site_coords: (K_total, 3) fractional coordinates of Wyckoff site anchors.
            full_coords: (N_total, 3) fractional coordinates of all unit cell atoms.
            lattices: (B, 6) crystal family lattice parameter vectors.
            num_sites: (B,) number of Wyckoff site anchors per crystal.
            num_atoms: (B,) number of atoms in unit cell per crystal.
            site2graph: (K_total,) crystal index for each Wyckoff site anchor.
            inverse_site_map: (N_total,) mapping each unit cell atom to its Wyckoff site index.
            target_sites: Optional precomputed (E,) target site indices.
            source_atoms: Optional precomputed (E,) source atom indices.
            source_sites: Optional precomputed (E,) source site indices.
            edge2graph: Optional precomputed (E,) edge to graph indices.
            site_projectors: Optional (K_total, 3, 3) tangent space projectors P_k.

        Returns:
            lattice_out: (B, 6) predicted lattice updates.
            coord_out: (K_total, 3) predicted coordinate score vectors strictly in tangent space T_k.
        """
        if target_sites is None or source_atoms is None:
            target_sites, source_atoms = generate_asymmetric_edges(
                num_sites, num_atoms, site_coords.device
            )
        if source_sites is None:
            source_sites = inverse_site_map[source_atoms]
        if edge2graph is None:
            edge2graph = site2graph[target_sites]

        xi = site_coords[target_sites]
        xj = full_coords[source_atoms]
        frac_diff = (xj - xi) % 1.0

        node_features = self.node_embedding(site_atom_types.clamp(0, MAX_ATOMIC_NUM))
        t_per_site = t.repeat_interleave(num_sites, dim=0)
        node_features = torch.cat([node_features, t_per_site], dim=-1)
        node_features = self.atom_latent_emb(node_features)

        h_list = [node_features]
        for i, layer in enumerate(self.csp_layers):
            node_features = layer(
                node_features,
                site_coords,
                full_coords,
                lattices,
                target_sites,
                source_sites,
                source_atoms,
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
            node_features, site2graph, dim=0, dim_size=lattices.shape[0], reduce=self.pooling
        )

        coord_out = self.coord_out(node_features)
        if site_projectors is not None:
            coord_out = (site_projectors @ coord_out.unsqueeze(-1)).squeeze(-1)
        lattice_out = self.lattice_out(graph_features)

        return lattice_out, coord_out
