"""CSPNet: Graph Neural Network denoiser for crystal structures."""

import torch
import torch.nn as nn

from diffcsp.core.scatter import scatter
from diffcsp.models.layers import CSPLayer, SinusoidsEmbedding, generate_intra_crystal_edges

MAX_ATOMIC_NUM = 100


class CSPNet(nn.Module):
    """Deep GNN denoiser predicting coordinate scores and lattice updates."""

    def __init__(
        self,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_layers: int = 6,
        max_atoms: int = 100,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        num_freqs: int = 128,
        edge_style: str = "fc",
        coord_style: str = "node",
        ln: bool = False,
        dense: bool = False,
        smooth: bool = False,
        ip: bool = True,
        pred_type: bool = False,
        pred_scalar: bool = False,
        pooling: str = "mean",
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.smooth = smooth
        self.ip = ip
        self.dense = dense
        self.ln = ln
        self.pooling = pooling
        self.pred_type = pred_type
        self.pred_scalar = pred_scalar

        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms + 1, hidden_dim)

        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        self.act_fn = nn.SiLU() if act_fn == "silu" else nn.ReLU()
        self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs) if dis_emb == "sin" else None

        self.csp_layers = nn.ModuleList(
            [CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip) for _ in range(num_layers)]
        )

        hidden_dim_out = hidden_dim * (num_layers + 1) if self.dense else hidden_dim

        self.coord_out = nn.Linear(hidden_dim_out, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim_out, 6, bias=False)

        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

        if self.pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim_out, 1)

    def forward(
        self,
        t: torch.Tensor,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward denoising step.

        Args:
            t: (B, latent_dim) timestep embeddings.
            atom_types: (N,) atomic numbers.
            frac_coords: (N, 3) fractional atomic coordinates.
            lattices: (B, 6) crystal family lattice vectors.
            num_atoms: (B,) atom counts per structure.
            node2graph: (N,) graph membership index.

        Returns:
            Tuple of (lattice_out, coord_out) or extended outputs if type/scalar requested.
        """
        edges, frac_diff = generate_intra_crystal_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]

        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types.clamp(0, MAX_ATOMIC_NUM))

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=-1)
        node_features = self.atom_latent_emb(node_features)

        h_list = [node_features]
        for i, layer in enumerate(self.csp_layers):
            node_features = layer(
                node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff
            )
            if i != self.num_layers - 1:
                h_list.append(node_features)

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        h_list.append(node_features)
        if self.dense:
            node_features = torch.cat(h_list, dim=-1)

        graph_features = scatter(
            node_features, node2graph, dim=0, dim_size=lattices.shape[0], reduce=self.pooling
        )

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        coord_out = self.coord_out(node_features)
        lattice_out = self.lattice_out(graph_features)

        if self.pred_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out

        return lattice_out, coord_out
