"""WyckoffPaiNN: Cartesian vector-equivariant Graph Neural Network for the Asymmetric Unit."""

import torch
import torch.nn as nn

from diffcsp.core.crystal_family import CrystalFamily
from diffcsp.core.scatter import scatter
from diffcsp.models.layers import generate_asymmetric_edges
from diffcsp.models.painn_layers import WyckoffPaiNNLayer

MAX_ATOMIC_NUM = 100


class WyckoffPaiNN(nn.Module):
    """Deep Cartesian vector-equivariant GNN denoiser on Wyckoff site anchors.

    Maintains dual representations (invariant scalar + equivariant vector) on the
    asymmetric unit (K unique sites). Wyckoff symmetry replicas across the unit
    cell transform their vector channels via exact Cartesian isometries M_j in O(3).
    Message passing is driven by real Euclidean distances in Angstroms via Bessel
    RBFs, completely eliminating the coordinate-axis basis mismatch and steric
    clash vulnerabilities of CSPNet.

    Reference:
        Innovation 1 & 3, docs/architectural-innovations.md
        docs/asymmetric-unit-deficit.md
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_rbf: int = 64,
        r_cut: float = 6.0,
        max_atoms: int = 100,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.r_cut = r_cut
        self.pooling = pooling
        self.crystal_family = CrystalFamily()

        self.node_embedding = nn.Embedding(max_atoms + 1, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim + 6, hidden_dim)

        self.painn_layers = nn.ModuleList(
            [WyckoffPaiNNLayer(hidden_dim=hidden_dim, num_rbf=num_rbf, r_cut=r_cut) for _ in range(num_layers)]
        )

        # Equivariant coordinate output head: linear combination of vector channels
        self.coord_out_weights = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.normal_(self.coord_out_weights.weight, std=0.01)

        self.final_norm = nn.LayerNorm(hidden_dim)

        # Lattice output head: maps rotationally-invariant pooled scalar state + current lattice state to 6D crystal family update
        self.lattice_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6, bias=False),
        )
        nn.init.zeros_(self.lattice_mlp[-1].weight)

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
        batch_ops: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward denoising step on the asymmetric unit.

        Args:
            t: (B, latent_dim) timestep embeddings.
            site_atom_types: (K_total,) atomic numbers of Wyckoff site anchors.
            site_coords: (K_total, 3) fractional coordinates of Wyckoff site anchors.
            full_coords: (N_total, 3) fractional coordinates of all unit cell atoms.
            lattices: (B, 6) crystal family vectors or (B, 3, 3) Cartesian lattice matrices.
            num_sites: (B,) number of Wyckoff site anchors per crystal.
            num_atoms: (B,) number of atoms in unit cell per crystal.
            site2graph: (K_total,) crystal index for each Wyckoff site anchor.
            inverse_site_map: (N_total,) mapping each unit cell atom to its Wyckoff site index.
            target_sites: Optional precomputed (E,) target site indices.
            source_atoms: Optional precomputed (E,) source atom indices.
            source_sites: Optional precomputed (E,) source site indices.
            edge2graph: Optional precomputed (E,) edge to graph indices.
            site_projectors: Optional (K_total, 3, 3) tangent space projectors P_k.
            batch_ops: Optional (N_total, 4, 4) space group affine operations.

        Returns:
            lattice_out: (B, 6) predicted lattice updates.
            coord_out: (K_total, 3) predicted coordinate score vectors strictly in tangent space T_k.
        """
        B = lattices.shape[0]

        # Ensure Cartesian (B, 3, 3) lattice representation and 6D Lie algebra vector
        if lattices.dim() == 2 and lattices.shape[1] == 6:
            crys_fam = lattices
            L = self.crystal_family.v2m(lattices)
        else:
            L = lattices
            crys_fam = self.crystal_family.m2v(self.crystal_family.de_so3(L))

        # Regularized matrix inverse to guarantee stability even under noisy early sampling steps
        L_safe = torch.nan_to_num(L, nan=1.0, posinf=50.0, neginf=-50.0)
        U, S, Vh = torch.linalg.svd(L_safe)
        S_inv = 1.0 / torch.clamp(S, min=0.5)
        inv_L = (Vh.transpose(-2, -1) * S_inv.unsqueeze(-2)) @ U.transpose(-2, -1)

        # Build bipartite asymmetric edge topology if not provided
        if target_sites is None or source_atoms is None:
            target_sites, source_atoms = generate_asymmetric_edges(
                num_sites, num_atoms, site_coords.device
            )
        if source_sites is None:
            source_sites = inverse_site_map[source_atoms]
        if edge2graph is None:
            edge2graph = site2graph[target_sites]

        # Compute exact Cartesian replica rotation matrices M_j in O(3)
        atom2graph = site2graph[inverse_site_map]
        if batch_ops is not None:
            inv_L_atoms = inv_L[atom2graph]
            L_atoms = L[atom2graph]
            R_T = batch_ops[:, :3, :3].transpose(-1, -2)
            M = inv_L_atoms @ R_T @ L_atoms
            replica_rotations = M[source_atoms]
        else:
            replica_rotations = None

        # Initialize dual feature streams
        node_features = self.node_embedding(site_atom_types.clamp(0, MAX_ATOMIC_NUM))
        t_per_site = t.repeat_interleave(num_sites, dim=0)
        crys_fam_per_site = crys_fam[site2graph]
        s_sites = self.atom_latent_emb(torch.cat([node_features, t_per_site, crys_fam_per_site], dim=-1))
        v_sites = torch.zeros(
            s_sites.shape[0], self.hidden_dim, 3, device=s_sites.device, dtype=s_sites.dtype
        )

        # Message passing layers
        last_e_ij = None
        last_r_cart = None
        for layer in self.painn_layers:
            s_sites, v_sites, last_e_ij, last_r_cart, _ = layer(
                s_sites,
                v_sites,
                site_coords,
                full_coords,
                L,
                target_sites,
                source_sites,
                source_atoms,
                edge2graph,
                replica_rotations=replica_rotations,
            )

        # 1. Equivariant coordinate readout
        # Map (K, C, 3) -> (K, 3) Cartesian vector update
        coord_cart = self.coord_out_weights(v_sites.transpose(1, 2)).squeeze(-1)  # (K, 3)
        coord_norm = torch.sqrt(torch.sum(coord_cart**2, dim=-1, keepdim=True) + 1e-8)
        coord_cart = coord_cart * torch.clamp(10.0 / coord_norm, max=1.0)
        
        # Convert Cartesian displacement to fractional: delta_x = coord_cart @ inv_L
        inv_L_sites = inv_L[site2graph]
        coord_frac = torch.einsum("ki, kij -> kj", coord_cart, inv_L_sites)

        # Project into Wyckoff tangent space T_k
        if site_projectors is not None:
            coord_out = (site_projectors @ coord_frac.unsqueeze(-1)).squeeze(-1)
        else:
            coord_out = coord_frac

        # 2. Lattice readout: pooled invariant scalar state + current lattice state
        s_sites_norm = self.final_norm(s_sites)
        graph_scalar = scatter(s_sites_norm, site2graph, dim=0, dim_size=B, reduce=self.pooling)
        crys_fam_clamped = crys_fam.clamp(-8.0, 8.0)
        lattice_out = self.lattice_mlp(torch.cat([graph_scalar, crys_fam_clamped], dim=-1))
        lattice_out = torch.clamp(lattice_out, -15.0, 15.0)

        return lattice_out, coord_out
