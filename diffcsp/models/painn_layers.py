"""Cartesian vector-equivariant PaiNN layers operating on the Asymmetric Unit.

Implements exact E(3)-equivariant directional message passing and intra-node
scalar-vector mixing for Wyckoff site anchors and their symmetry replicas.

Reference:
    PaiNN: Polarizable Atom Interaction Neural Network (Schütt et al., ICML 2021)
    Innovation 1 & 3, docs/architectural-innovations.md
"""

import math
import torch
import torch.nn as nn

from diffcsp.core.scatter import scatter


def bessel_rbf(d: torch.Tensor, num_rbf: int = 64, r_cut: float = 6.0) -> torch.Tensor:
    """Bessel radial basis functions with a C^2 smooth polynomial cutoff envelope.

    Args:
        d: (..., 1) or (...) distance tensor in Angstroms.
        num_rbf: Number of radial basis frequencies.
        r_cut: Interaction cutoff radius in Angstroms.

    Returns:
        (..., num_rbf) basis function activations smoothly vanishing at r_cut.
    """
    if d.dim() == 1:
        d = d.unsqueeze(-1)
    n = torch.arange(1, num_rbf + 1, dtype=d.dtype, device=d.device)
    c = math.pi / r_cut
    rbf = torch.sin(c * n * d) / d.clamp(min=1e-6)

    # C^2 smooth polynomial cutoff envelope
    s = d / r_cut
    env = 1.0 - 6.0 * s**5 + 15.0 * s**4 - 10.0 * s**3
    env = torch.where(d < r_cut, env, torch.zeros_like(env))
    return rbf * env


class WyckoffPaiNNLayer(nn.Module):
    """Cartesian vector-equivariant message passing layer for the Asymmetric Unit.

    Each Wyckoff site anchor possesses a dual feature state:
      - Invariant scalar channels: s_i in R^C
      - Equivariant Cartesian vector channels: v_i in R^(C x 3)

    Under space group operations (R_j, t_j), Wyckoff symmetry replicas j of
    anchor s(j) transform their vector features via the exact Cartesian isometry:
        v_j = v_s(j) @ M_j
    where M_j = L^(-1) @ R_j^T @ L in O(3).

    Message passing computes true Euclidean bond vectors r_ij in Angstroms under
    periodic boundary conditions, expands distances d_ij using Bessel RBFs,
    and updates both scalar and vector features equivariantly.
    """

    def __init__(self, hidden_dim: int = 128, num_rbf: int = 64, r_cut: float = 6.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.r_cut = r_cut

        # Block A: Inter-node message filter generators from invariant distances
        self.filter_mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
        self.scalar_message_lin = nn.Linear(hidden_dim, hidden_dim)
        self.scalar_norm = nn.LayerNorm(hidden_dim)

        # Block B: Intra-node scalar-vector mixing
        self.w_u = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_norm = nn.LayerNorm(hidden_dim)

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        # Zero-initialization of residual update branches for deep GNN stability
        nn.init.zeros_(self.update_mlp[-1].weight)
        nn.init.zeros_(self.update_mlp[-1].bias)
        nn.init.zeros_(self.w_p.weight)

    def forward(
        self,
        s_sites: torch.Tensor,
        v_sites: torch.Tensor,
        site_coords: torch.Tensor,
        full_coords: torch.Tensor,
        lattices: torch.Tensor,
        target_sites: torch.Tensor,
        source_sites: torch.Tensor,
        source_atoms: torch.Tensor,
        edge2graph: torch.Tensor,
        replica_rotations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward message passing and intra-node update.

        Args:
            s_sites: (K, C) invariant scalar states of Wyckoff site anchors.
            v_sites: (K, C, 3) equivariant Cartesian vector states of anchors.
            site_coords: (K, 3) fractional coordinates of anchors.
            full_coords: (N, 3) fractional coordinates of all unit cell atoms.
            lattices: (B, 3, 3) Cartesian lattice matrices where r = x @ L.
            target_sites: (E,) target site indices in [0, K).
            source_sites: (E,) source anchor indices in [0, K).
            source_atoms: (E,) source atom indices in [0, N).
            edge2graph: (E,) crystal graph index for each edge.
            replica_rotations: Optional (E, 3, 3) Cartesian rotation matrices M_j.

        Returns:
            s_out: (K, C) updated scalar states.
            v_out: (K, C, 3) updated Cartesian vector states.
            e_ij: (E, num_rbf) radial basis features.
            r_cart: (E, 3) Cartesian edge displacement vectors.
            r_unit: (E, 3) normalized unit direction vectors.
        """
        L = lattices[edge2graph]  # (E, 3, 3)
        xi = site_coords[target_sites]  # (E, 3)
        xj = full_coords[source_atoms]  # (E, 3)

        # Minimum image convention on fractional coordinates under PBC
        frac_diff = (xj - xi + 0.5) % 1.0 - 0.5
        r_cart = torch.einsum("ei, eij -> ej", frac_diff, L)  # (E, 3)
        d_ij = torch.sqrt(torch.sum(r_cart ** 2, dim=-1, keepdim=True) + 1e-8)  # (E, 1)
        r_unit = r_cart / d_ij  # (E, 3)

        # Invariant Bessel radial expansion
        e_ij = bessel_rbf(d_ij, num_rbf=self.num_rbf, r_cut=self.r_cut)  # (E, num_rbf)
        w_filters = self.filter_mlp(e_ij)  # (E, 3 * C)
        w_s, w_v1, w_v2 = torch.chunk(w_filters, 3, dim=-1)

        # Wyckoff Replica Vector Transformation:
        # Rotate source anchor's vector state by the space group operation M_j:
        # v_replica = v_source @ M_j
        v_source = v_sites[source_sites]  # (E, C, 3)
        if replica_rotations is not None:
            v_replica = torch.einsum("eca, eab -> ecb", v_source, replica_rotations)
        else:
            v_replica = v_source
        s_replica = s_sites[source_sites]  # (E, C) - scalar is invariant

        # Message computation with normalized scalar input
        s_norm = self.scalar_norm(s_sites)
        s_replica = s_norm[source_sites]
        h_s = self.scalar_message_lin(s_replica)  # (E, C)

        # Vector message: combines rotated source vector with directional bond vector
        m_v = (h_s * w_v1).unsqueeze(-1) * v_replica + (h_s * w_v2).unsqueeze(-1) * r_unit.unsqueeze(1)
        m_s = h_s * w_s

        # Aggregate messages at target anchor sites
        K = s_sites.shape[0]
        agg_v = scatter(m_v, target_sites, dim=0, dim_size=K, reduce="mean")
        agg_s = scatter(m_s, target_sites, dim=0, dim_size=K, reduce="mean")

        s_sites = s_sites + agg_s
        v_sites = v_sites + agg_v

        # Block B: Intra-node mixing
        # Vector linear combinations across channels
        U = self.w_u(v_sites.transpose(1, 2)).transpose(1, 2)  # (K, C, 3)
        V = self.w_v(v_sites.transpose(1, 2)).transpose(1, 2)  # (K, C, 3)

        # Invariant channel norms and inner products (scaled by channel dimension)
        q = torch.sqrt(torch.sum(V ** 2, dim=-1) + 1e-8)  # (K, C)
        p = torch.sum(U * V, dim=-1) / math.sqrt(self.hidden_dim)  # (K, C)

        sq = torch.cat([self.update_norm(s_sites), q], dim=-1)  # (K, 2C)
        out_sq = self.update_mlp(sq)
        delta_s, gate_v = torch.chunk(out_sq, 2, dim=-1)
        delta_s = delta_s + self.w_p(p)

        s_out = s_sites + delta_s
        v_out = v_sites + torch.tanh(gate_v).unsqueeze(-1) * U

        return s_out, v_out, e_ij, r_cart, r_unit
