import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse

from cspnet import CSPLayer, SinusoidsEmbedding
from crystal_family import CrystalFamily
from orb_wrapper import (
    OrbBackboneWrapper,
    build_orb_backbone,
    frac_to_cart_coords,
    cart_forces_to_frac_forces
)

MAX_ATOMIC_NUM = 100


class CSPNetORB(nn.Module):
    """
    Lightweight Denoising Adapter on top of a frozen ORB MLIP Backbone.
    
    Instead of training a large 6-layer GNN from scratch, this model:
    1. Evaluates structures using a frozen pretrained ORB MLIP potential.
    2. Extracts atomic forces, virial stress, and pretrained node/graph representations.
    3. Transforms Cartesian quantities to fractional/lattice representations.
    4. Applies a lightweight adapter (1-2 graph layers) conditioned on diffusion timestep.
    5. Predicts coordinate scores and crystal family lattice updates.
    6. (By Design) Enforces that the coordinate denoising step can be zero ONLY IF
       the ORB forces are zero: S = 0 ==> F = 0. If F = 0, S can still be non-zero
       to escape local minima and search for the global minimum.
    """

    def __init__(
        self,
        orb_backbone: OrbBackboneWrapper = None,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_layers: int = 2,
        max_atoms: int = 100,
        act_fn: str = 'silu',
        dis_emb: str = 'sin',
        num_freqs: int = 64,
        pooling: str = 'mean',
        use_force_residual: bool = True,
        enforce_zero_force_condition: bool = True,
        enforce_zero_stress_condition: bool = False,
        gamma_min: float = 1e-3,
        max_force_clamp: float = 20.0,
        max_stress_clamp: float = 50.0,
        orb_node_dim: int = 256,
        orb_graph_dim: int = 256,
        device: str = 'cpu'
    ):
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

        # 1. Frozen ORB MLIP Backbone
        if orb_backbone is None:
            self.orb_backbone = build_orb_backbone(
                model_name="orb-v2",
                device=device,
                use_mock=False,
                node_dim=orb_node_dim,
                graph_dim=orb_graph_dim,
            )
        else:
            self.orb_backbone = orb_backbone

        # Ensure ORB backbone is strictly frozen
        self.orb_backbone.eval()
        for p in self.orb_backbone.parameters():
            p.requires_grad = False

        # 2. Embedding & Projection Layers
        self.node_embedding = nn.Embedding(max_atoms + 1, hidden_dim)

        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.ReLU()

        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        else:
            self.dis_emb = None

        # ORB feature projections
        self.orb_node_proj = nn.Linear(orb_node_dim, hidden_dim)
        self.orb_graph_proj = nn.Linear(orb_graph_dim, hidden_dim)
        # Cartesian force (3) + fractional force (3)
        self.force_proj = nn.Linear(6, hidden_dim)
        # Stress tensor (9) -> hidden_dim
        self.stress_proj = nn.Linear(9, hidden_dim)

        # Combine [atom_emb, time_emb, orb_node, force_feat] -> hidden_dim
        combined_dim = hidden_dim * 3 + latent_dim
        self.input_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            self.act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. Lightweight Message Passing Layers (Adapter)
        self.csp_layers = nn.ModuleList([
            CSPLayer(
                hidden_dim=hidden_dim,
                act_fn=self.act_fn,
                dis_emb=self.dis_emb,
                ln=True,
                ip=True
            )
            for _ in range(num_layers)
        ])

        # 4. Output Heads
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)

        # Strictly positive along-force scaling network: gamma = gamma_min + softplus(...)
        if self.enforce_zero_force_condition:
            self.gamma_head = nn.Sequential(
                nn.Linear(latent_dim + hidden_dim, 64),
                self.act_fn,
                nn.Linear(64, 1)
            )
        elif self.use_force_residual:
            self.time_gate = nn.Sequential(
                nn.Linear(latent_dim, 64),
                self.act_fn,
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        # Lattice head takes: pooled_nodes (hidden_dim) + orb_graph (hidden_dim)
        # + stress_feat (hidden_dim) + crys_fam (6) + time_emb (latent_dim)
        lattice_in_dim = hidden_dim * 3 + 6 + latent_dim
        self.lattice_mlp = nn.Sequential(
            nn.Linear(lattice_in_dim, hidden_dim),
            self.act_fn,
            nn.Linear(hidden_dim, 6, bias=False)
        )

        if self.enforce_zero_stress_condition:
            self.gamma_stress_head = nn.Sequential(
                nn.Linear(latent_dim + hidden_dim, 64),
                self.act_fn,
                nn.Linear(64, 1)
            )

    def gen_edges(self, num_atoms: torch.Tensor, frac_coords: torch.Tensor):
        """Generates intra-crystal fully connected graphs."""
        lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
        fc_graph = torch.block_diag(*lis)
        fc_edges, _ = dense_to_sparse(fc_graph)
        frac_diff = (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.
        return fc_edges, frac_diff

    def apply_zero_force_constraint(
        self,
        v_coord: torch.Tensor,
        f_frac: torch.Tensor,
        t_per_atom: torch.Tensor,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforces by design that the coordinate denoising step can be zero ONLY IF
        the ORB forces are zero: S = 0 ==> f_frac = 0.
        
        Mathematical construction:
            S = gamma * f_frac + v_perp
        where:
            gamma >= gamma_min > 0 is strictly positive.
            v_perp is orthogonal to f_frac (v_perp . f_frac == 0).
            
        Proof of properties:
        1. If f_frac != 0:
           ||S||^2 = gamma^2 ||f_frac||^2 + ||v_perp||^2 >= gamma_min^2 ||f_frac||^2 > 0.
           Therefore S cannot be zero.
        2. S == 0 implies f_frac == 0 (contrapositive).
        3. If f_frac == 0 (e.g. at a local minimum):
           v_perp = v_coord, so S = v_coord.
           The network can propose arbitrary non-zero steps (S != 0) to bypass local minima
           and search for the global minimum.
        """
        f_norm = torch.norm(f_frac, dim=-1, keepdim=True)  # (N, 1)
        has_force = (f_norm > 1e-7).float()
        f_unit = f_frac / (f_norm + 1e-8)

        # Decompose v_coord into parallel and perpendicular components to f_frac
        v_parallel = torch.sum(v_coord * f_unit, dim=-1, keepdim=True) * f_unit * has_force
        v_perp = v_coord - v_parallel

        # Compute strictly positive gamma: gamma >= gamma_min > 0
        gamma_raw = self.gamma_head(torch.cat([t_per_atom, node_features], dim=-1))
        gamma = self.gamma_min + F.softplus(gamma_raw)

        # S = gamma * f_frac + v_perp
        coord_score = gamma * f_frac + v_perp
        return coord_score

    def apply_zero_stress_constraint(
        self,
        v_lattice: torch.Tensor,
        stress: torch.Tensor,
        t: torch.Tensor,
        pooled_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Analogous zero-stress condition for the lattice update:
        Lattice update can be zero ONLY IF virial stress is zero.
        """
        batch_size = stress.shape[0]
        # Extract 6 independent components of symmetric stress tensor
        s_vec = torch.stack([
            stress[:, 0, 0], stress[:, 1, 1], stress[:, 2, 2],
            stress[:, 1, 2], stress[:, 0, 2], stress[:, 0, 1]
        ], dim=-1)  # (B, 6)

        s_norm = torch.norm(s_vec, dim=-1, keepdim=True)  # (B, 1)
        has_stress = (s_norm > 1e-7).float()
        s_unit = s_vec / (s_norm + 1e-8)

        v_parallel = torch.sum(v_lattice * s_unit, dim=-1, keepdim=True) * s_unit * has_stress
        v_perp = v_lattice - v_parallel

        gamma_raw = self.gamma_stress_head(torch.cat([t, pooled_nodes], dim=-1))
        gamma = self.gamma_min + F.softplus(gamma_raw)

        lattice_noise = gamma * s_vec + v_perp
        return lattice_noise

    def forward(
        self,
        t: torch.Tensor,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        crys_fam: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor
    ):
        """
        Forward pass compatible with CSPNet interface in DiffCSP++.
        
        Args:
            t: (B, latent_dim) sinusoidal time embeddings
            atom_types: (N,) atomic numbers
            frac_coords: (N, 3) fractional coordinates
            crys_fam: (B, 6) crystal family representation of lattices
            num_atoms: (B,) number of atoms per structure in batch
            node2graph: (N,) batch index mapping for each atom
            
        Returns:
            lattice_out: (B, 6) predicted noise for crystal family vector
            coord_out: (N, 3) predicted score/noise for fractional coordinates
        """
        batch_size = crys_fam.shape[0]

        # 1. Coordinate transformations for ORB
        lattices = self.crystal_family.v2m(crys_fam)  # (B, 3, 3)
        cart_coords = frac_to_cart_coords(frac_coords, lattices, node2graph)

        # 2. Frozen ORB MLIP evaluation (no gradients computed for backbone)
        self.orb_backbone.eval()
        with torch.no_grad():
            orb_out = self.orb_backbone(
                atom_types=atom_types,
                cart_coords=cart_coords,
                lattices=lattices,
                num_atoms=num_atoms,
                node2graph=node2graph
            )

        f_cart = orb_out['forces']  # (N, 3) clamped Cartesian forces
        f_frac = cart_forces_to_frac_forces(f_cart, lattices, node2graph)  # (N, 3)
        stress = orb_out['stress']  # (B, 3, 3) clamped Cauchy stress
        orb_node_emb = orb_out['node_emb']  # (N, orb_node_dim)
        orb_graph_emb = orb_out['graph_emb']  # (B, orb_graph_dim)

        # 3. Feature conditioning
        atom_feat = self.node_embedding(atom_types.clamp(0, 100))
        t_per_atom = t.repeat_interleave(num_atoms, dim=0)

        orb_node_feat = self.act_fn(self.orb_node_proj(orb_node_emb))
        force_feat = self.act_fn(self.force_proj(torch.cat([f_cart, f_frac], dim=-1)))

        # Fuse embeddings into initial hidden state
        fused_in = torch.cat([atom_feat, t_per_atom, orb_node_feat, force_feat], dim=-1)
        node_features = self.input_fusion(fused_in)

        # 4. Message Passing with Lightweight CSPLayers
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]

        for layer in self.csp_layers:
            node_features = layer(
                node_features=node_features,
                frac_coords=frac_coords,
                lattices=crys_fam,
                edge_index=edges,
                edge2graph=edge2graph,
                frac_diff=frac_diff
            )

        # 5. Output raw coordinate proposal v_coord
        v_coord = self.coord_out(node_features)

        # 6. Apply Zero-Force Condition by Design:
        # S = 0 ==> f_frac = 0, while f_frac = 0 allows S != 0 to bypass local minima
        if self.enforce_zero_force_condition:
            coord_score = self.apply_zero_force_constraint(
                v_coord=v_coord,
                f_frac=f_frac,
                t_per_atom=t_per_atom,
                node_features=node_features
            )
        elif self.use_force_residual:
            gate = self.time_gate(t).repeat_interleave(num_atoms, dim=0)
            coord_score = v_coord + gate * f_frac
        else:
            coord_score = v_coord

        # 7. Output lattice noise
        pooled_nodes = scatter(node_features, node2graph, dim=0, reduce=self.pooling, dim_size=batch_size)
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
                pooled_nodes=pooled_nodes
            )
        else:
            lattice_noise = v_lattice

        return lattice_noise, coord_score
