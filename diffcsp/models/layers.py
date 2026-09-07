"""Neural network layers and message passing primitives for crystal denoising."""

import math

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from diffcsp.core.scatter import scatter


class SinusoidsEmbedding(nn.Module):
    """Sinusoidal frequency embedding of fractional coordinates or spatial distances."""

    def __init__(self, n_frequencies: int = 10, n_space: int = 3) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        frequencies = 2.0 * math.pi * torch.arange(self.n_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds coordinates (N, n_space) -> (N, dim)."""
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :]
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def generate_intra_crystal_edges(
    num_atoms: torch.Tensor, frac_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs intra-crystal fully-connected edge index and periodic fractional offsets.

    Args:
        num_atoms: (B,) number of atoms per crystal in the batch.
        frac_coords: (N, 3) fractional atomic coordinates.

    Returns:
        edges: (2, num_edges) edge index tensor.
        frac_diff: (num_edges, 3) fractional displacement vectors on torus in [0, 1).
    """
    block_list = [
        torch.ones((n.item(), n.item()), device=num_atoms.device, dtype=torch.bool) for n in num_atoms
    ]
    fc_graph = torch.block_diag(*block_list)
    fc_edges, _ = dense_to_sparse(fc_graph)
    frac_diff = (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
    return fc_edges, frac_diff


class CSPLayer(nn.Module):
    """Message passing graph layer for CSPNet."""

    def __init__(
        self,
        hidden_dim: int = 128,
        act_fn: nn.Module | None = None,
        dis_emb: nn.Module | None = None,
        ln: bool = False,
        ip: bool = True,
    ) -> None:
        super().__init__()
        self.dis_emb = dis_emb
        self.dis_dim = dis_emb.dim if dis_emb is not None else 3
        self.ln = ln
        self.ip = ip

        act = act_fn if act_fn is not None else nn.SiLU()

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6 + self.dis_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
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

    def edge_model(
        self,
        node_features: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice_rep: torch.Tensor,
        edge_index: torch.Tensor,
        edge2graph: torch.Tensor,
        frac_diff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes message features along edges."""
        hi = node_features[edge_index[0]]
        hj = node_features[edge_index[1]]

        if frac_diff is None:
            xi = frac_coords[edge_index[0]]
            xj = frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0

        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)

        lattice_edges = lattice_rep[edge2graph]
        edge_in = torch.cat([hi, hj, lattice_edges, frac_diff], dim=-1)
        return self.edge_mlp(edge_in)

    def node_model(
        self, node_features: torch.Tensor, edge_features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Aggregates edge messages into updated node states."""
        agg = scatter(edge_features, edge_index[0], dim=0, dim_size=node_features.shape[0], reduce="mean")
        node_in = torch.cat([node_features, agg], dim=-1)
        return self.node_mlp(node_in)

    def forward(
        self,
        node_features: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        edge_index: torch.Tensor,
        edge2graph: torch.Tensor,
        frac_diff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Residual message passing step."""
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_feat = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        node_out = self.node_model(node_features, edge_feat, edge_index)
        return node_input + node_out
