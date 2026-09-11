"""Frozen ORB MLIP Backbone Wrapper for DiffCSP++.

Wraps pretrained Machine Learning Interatomic Potentials (ORB v2/v3) from
orb_models, extracting frozen atomic representations, Cartesian forces,
and virial stresses. Includes MockOrbBackbone for testing and lightweight environments.

The whole batch is featurized in a single pass: one host transfer, one batched
neighbour-list construction on the GPU (``from_ase_atoms_list``, which dispatches
to the fused ``knn_alchemi`` kernel), and one ``predict`` call. Short-range
repulsion comes from ORB's own edge-based ZBL potential rather than a
per-structure Python loop -- orb-v3 evaluates it inside ``predict``, and for
backbones built without it (orb-v2) it is applied to the same batched graph.
"""

import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffcsp.core.scatter import scatter
from diffcsp.data.transforms import (
    clamp_forces_and_stress,
)
from diffcsp.models.repulsion import batched_crystal_repulsion

logger = logging.getLogger(__name__)

# Short aliases -> orb_models.forcefield.pretrained loader names.
#
# Defaults pick the *direct* orb-v3 heads: they regress forces and stress in a
# single forward pass, whereas the conservative variants differentiate the
# energy and so cannot run under the ``no_grad`` used here. The "20" variants
# cap the neighbour list at 20 edges/atom, which is what the batched kernel
# wants and is markedly cheaper than the 120 of the "inf" variants.
ORB_MODEL_ALIASES = {
    "orb-v3": "orb_v3_direct_20_mpa",
    "orb-v3-direct": "orb_v3_direct_20_mpa",
    "orb-v3-direct-mpa": "orb_v3_direct_20_mpa",
    "orb-v3-direct-omat": "orb_v3_direct_20_omat",
    "orb-v3-conservative": "orb_v3_conservative_20_mpa",
    "orb-v2": "orb_v2",
}


def _resolve_orb_loader(model_name: str):
    """Maps a model name or alias onto an ``orb_models`` pretrained loader.

    Raises ValueError on an unknown name rather than silently substituting a
    different backbone.
    """
    from orb_models.forcefield import pretrained

    key = model_name.strip()
    attr = ORB_MODEL_ALIASES.get(key, key.replace("-", "_"))
    loader = getattr(pretrained, attr, None)
    if loader is None or not callable(loader):
        known = sorted(ORB_MODEL_ALIASES)
        available = sorted(n for n in dir(pretrained) if n.startswith("orb_v"))
        raise ValueError(
            f"Unknown ORB model '{model_name}' (resolved to '{attr}'). "
            f"Aliases: {known}. Loaders: {available}."
        )
    return loader, attr


class MockOrbBackbone(nn.Module):
    """Lightweight mock backbone matching ORB's output signatures.

    Used for testing and CPU execution without downloading large weights.
    """

    def __init__(self, node_dim: int = 256, graph_dim: int = 256, max_atomic_num: int = 100) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        self.atom_embedding = nn.Embedding(max_atomic_num + 1, node_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim + 3, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(node_dim + 9, graph_dim),
            nn.SiLU(),
            nn.Linear(graph_dim, graph_dim),
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
        node2graph: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Mock forward pass."""
        atom_h = self.atom_embedding(atom_types.clamp(0, 100))
        node_in = torch.cat([atom_h, cart_coords], dim=-1)
        node_emb = self.node_proj(node_in)

        batch_size = lattices.shape[0]
        pooled_nodes = scatter(node_emb, node2graph, dim=0, dim_size=batch_size, reduce="mean")
        lattices_flat = lattices.reshape(batch_size, 9)
        graph_in = torch.cat([pooled_nodes, lattices_flat], dim=-1)
        graph_emb = self.graph_proj(graph_in)

        forces = self.force_head(node_emb)
        stress = self.stress_head(graph_emb).reshape(batch_size, 3, 3)
        # Symmetrize stress tensor
        stress = 0.5 * (stress + stress.transpose(-1, -2))
        energy = self.energy_head(graph_emb)

        return {
            "node_emb": node_emb,
            "graph_emb": graph_emb,
            "forces": forces,
            "stress": stress,
            "energy": energy,
        }


class OrbBackboneWrapper(nn.Module):
    """Wrapper around pretrained ORB MLIP model from `orb_models`.

    Provides frozen atomic forces, virial stress, and representations.
    Falls back gracefully to MockOrbBackbone if orb_models is not installed.
    """

    # Default cell volume per atom (A^3) below which a structure is not sent to
    # the MLIP. The neighbour search enumerates periodic images inside a 6 A
    # cutoff, so on a collapsed cell the image count explodes: construction
    # fails outright below ~1 A^3/atom, and stays expensive up to ~5, against
    # 10-30 for a real crystal. Such structures get the bounded repulsive core
    # instead, which needs no neighbour list and is where the MLIP would be far
    # out of distribution anyway.
    DEFAULT_MIN_VOLUME_PER_ATOM = 5.0

    def __init__(
        self,
        model_name: str = "orb-v3",
        device: str = "cpu",
        use_mock: bool = False,
        node_dim: int = 256,
        graph_dim: int = 256,
        max_force: float = 2.0,
        max_stress: float = 0.05,
        edge_method: str = "knn_alchemi",
        min_volume_per_atom: float | None = None,
        use_node_features: bool = True,
        normalize_node_features: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        self.max_force = max_force
        self.max_stress = max_stress
        self.use_mock = use_mock
        self.edge_method = edge_method
        self.min_volume_per_atom = (
            self.DEFAULT_MIN_VOLUME_PER_ATOM if min_volume_per_atom is None else min_volume_per_atom
        )
        self.use_node_features = use_node_features
        self.normalize_node_features = normalize_node_features
        self.is_loaded = False
        self.pair_repulsion_fn = None
        # Filled by a forward hook on the backbone's GNN so that predict() still
        # runs it exactly once; not a submodule, so it stays out of state_dict.
        self._node_feature_cache: dict[str, torch.Tensor] = {}

        if not use_mock:
            try:
                loader, attr = _resolve_orb_loader(model_name)
                self.orb_model, self.atoms_adapter = loader(device=device)
                logger.info("Loaded ORB backbone '%s' via pretrained.%s", model_name, attr)

                self.orb_model.eval()
                for p in self.orb_model.parameters():
                    p.requires_grad = False
                self.is_loaded = True

                if self.use_node_features:
                    self._register_node_feature_hook()

                # orb-v3 evaluates ZBL pair repulsion inside predict(); orb-v2 was
                # built without it, so attach one and apply it to the same batched
                # graph. Either way the short-range core is edge-based, giving a
                # bounded repulsive force on overlapping atoms at high diffusion
                # noise without a per-structure Python loop.
                if not getattr(self.orb_model, "pair_repulsion", False):
                    from orb_models.forcefield.models.pair_repulsion import ZBLBasis

                    self.pair_repulsion_fn = ZBLBasis(
                        p=6, compute_gradients=True, node_aggregation="sum"
                    ).to(device)
                    self.pair_repulsion_fn.eval()
                    logger.info(
                        "Backbone '%s' has no built-in pair repulsion; attached standalone ZBLBasis.",
                        model_name,
                    )
            except Exception as e:
                warnings.warn(
                    f"Could not initialize orb_models ({e}). Using MockOrbBackbone for execution.",
                    UserWarning,
                    stacklevel=2,
                )
                self.use_mock = True
                self.mock_backbone = MockOrbBackbone(node_dim=node_dim, graph_dim=graph_dim)
        else:
            self.mock_backbone = MockOrbBackbone(node_dim=node_dim, graph_dim=graph_dim)

        for p in self.parameters():
            p.requires_grad = False

    def _register_node_feature_hook(self) -> None:
        """Captures the backbone GNN's node representations during ``predict``.

        ``predict`` returns only head outputs, and re-running the GNN to get its
        latent would double the cost of the backbone. A forward hook on the GNN
        submodule reads the representations out of the pass ``predict`` already
        makes.
        """
        gnn = getattr(self.orb_model, "model", None)
        if gnn is None:
            self.use_node_features = False
            warnings.warn(
                f"ORB backbone '{self.model_name}' exposes no GNN submodule; "
                "node representations are unavailable and will be left at zero.",
                UserWarning,
                stacklevel=2,
            )
            return

        def _capture(_module, _args, output) -> None:
            if isinstance(output, dict) and "node_features" in output:
                self._node_feature_cache["node_features"] = output["node_features"]

        gnn.register_forward_hook(_capture)

    @torch.no_grad()
    def forward(
        self,
        atom_types: torch.Tensor,
        cart_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluates frozen interatomic potential on a batch of crystal structures.

        Returns the potential's forces and stress alongside its learned atomic
        representations, so the adapter sees both the physics (which way the
        atoms want to move) and the chemistry (what the atoms are, as the
        pretrained model encodes them).
        """
        if self.use_mock or not self.is_loaded:
            return self.mock_backbone(atom_types, cart_coords, lattices, num_atoms, node2graph)

        out = self._predict_batch(atom_types, cart_coords, lattices, num_atoms)

        clamped_forces, clamped_stress = clamp_forces_and_stress(
            out["forces"],
            out["stress"],
            max_force=self.max_force,
            max_stress=self.max_stress,
        )

        return {
            "node_emb": out["node_emb"],
            "graph_emb": out["graph_emb"],
            "forces": clamped_forces,
            "stress": clamped_stress,
            "energy": out["energy"],
        }

    def _fit_dim(self, features: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Pads or truncates a feature tensor's last dimension to ``target_dim``."""
        dim = features.shape[-1]
        if dim == target_dim:
            return features
        if dim < target_dim:
            return F.pad(features, (0, target_dim - dim))
        return features[..., :target_dim]

    def _build_atoms_list(
        self,
        atom_types: torch.Tensor,
        cart_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> list:
        """Converts a batched crystal tensor triple into a list of ``ase.Atoms``.

        Uses one host transfer per tensor for the whole batch rather than one
        per structure. Callers are expected to have screened out collapsed cells
        already; see :meth:`_predict_batch`.
        """
        from ase import Atoms

        types_np = atom_types.detach().to(torch.int32).cpu().numpy()
        coords_np = cart_coords.detach().to(torch.float64).cpu().numpy()
        cells_np = lattices.detach().to(torch.float64).cpu().numpy()
        counts = num_atoms.detach().cpu().numpy()
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        return [
            Atoms(
                numbers=np.clip(types_np[offsets[b] : offsets[b + 1]], 1, 118),
                positions=coords_np[offsets[b] : offsets[b + 1]],
                cell=cells_np[b],
                pbc=True,
            )
            for b in range(lattices.shape[0])
        ]

    def _predict_batch(
        self,
        atom_types: torch.Tensor,
        cart_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluates the frozen potential over the whole batch without per-structure loops.

        The batch is split once on cell density. Structures dense enough for a
        neighbour list go to the MLIP in a single batched call; the rest -- the
        collapsed cells produced at high diffusion noise, where the periodic
        image count explodes and the MLIP is out of distribution anyway -- get
        the bounded repulsive core, also batched. Both halves are written back
        into the batch layout.

        Representations are left at zero for the repulsion-only half: there is no
        graph to run the GNN over, and zero is the natural "no information"
        value for the adapter's projection.

        Returns raw (unclamped) forces (N, 3), Cauchy stress (B, 3, 3), energy
        (B, 1), node representations (N, node_dim) and pooled graph
        representations (B, graph_dim).
        """
        batch_size = lattices.shape[0]
        forces = torch.zeros_like(cart_coords)
        stress = torch.zeros((batch_size, 3, 3), dtype=lattices.dtype, device=lattices.device)
        energy = torch.zeros((batch_size, 1), dtype=cart_coords.dtype, device=cart_coords.device)
        node_emb = torch.zeros(
            (cart_coords.shape[0], self.node_dim), dtype=cart_coords.dtype, device=cart_coords.device
        )
        graph_emb = torch.zeros(
            (batch_size, self.graph_dim), dtype=cart_coords.dtype, device=cart_coords.device
        )

        volume_per_atom = torch.det(lattices).abs() / num_atoms.clamp(min=1)
        featurizable = volume_per_atom >= self.min_volume_per_atom
        # Single host sync per call, to size the two sub-batches.
        n_featurizable = int(featurizable.sum().item())

        if n_featurizable < batch_size:
            collapsed = ~featurizable
            node_sel = collapsed.repeat_interleave(num_atoms)
            f_rep, s_rep, e_rep = batched_crystal_repulsion(
                cart_coords=cart_coords[node_sel],
                lattices=lattices[collapsed],
                atom_types=atom_types[node_sel],
                num_atoms=num_atoms[collapsed],
                f_max=self.max_force,
                eta=0.70,
            )
            forces[node_sel] = f_rep
            stress[collapsed] = s_rep.to(stress.dtype)
            energy[collapsed] = e_rep.to(energy.dtype)
            logger.debug(
                "%d/%d structures below %.1f A^3/atom; using repulsive core.",
                batch_size - n_featurizable,
                batch_size,
                self.min_volume_per_atom,
            )

        if n_featurizable > 0:
            node_sel = featurizable.repeat_interleave(num_atoms)
            sub = self._predict_mlip(
                atom_types=atom_types[node_sel],
                cart_coords=cart_coords[node_sel],
                lattices=lattices[featurizable],
                num_atoms=num_atoms[featurizable],
            )
            forces[node_sel] = sub["forces"]
            stress[featurizable] = sub["stress"].to(stress.dtype)
            energy[featurizable] = sub["energy"].to(energy.dtype)
            if sub["node_emb"] is not None:
                node_emb[node_sel] = sub["node_emb"].to(node_emb.dtype)
                graph_emb[featurizable] = sub["graph_emb"].to(graph_emb.dtype)

        return {
            "forces": forces,
            "stress": stress,
            "energy": energy,
            "node_emb": node_emb,
            "graph_emb": graph_emb,
        }

    def _predict_mlip(
        self,
        atom_types: torch.Tensor,
        cart_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Runs the ORB backbone over a sub-batch in one graph build and one predict.

        The GNN's node representations are captured from the forward hook
        installed in ``__init__`` rather than recomputed, so ``predict`` still
        runs the backbone exactly once.

        Falls back to the batched repulsive core if the backbone or the
        neighbour search raises, so that one pathological structure cannot abort
        training.
        """
        batch_size = lattices.shape[0]
        try:
            atoms_list = self._build_atoms_list(atom_types, cart_coords, lattices, num_atoms)
            # One batched neighbour-list build on device: knn_alchemi is a single
            # fused kernel over the whole sub-batch.
            graph = self.atoms_adapter.from_ase_atoms_list(
                atoms_list, device=self.device, edge_method=self.edge_method
            )
            self._node_feature_cache.pop("node_features", None)
            pred = self.orb_model.predict(graph)

            forces = torch.as_tensor(pred["forces"], device=cart_coords.device, dtype=cart_coords.dtype)
            energy = torch.as_tensor(pred["energy"], device=cart_coords.device, dtype=cart_coords.dtype)
            stress = self._to_cauchy(
                torch.as_tensor(
                    pred.get("stress", torch.zeros((batch_size, 6))),
                    device=lattices.device,
                    dtype=lattices.dtype,
                ),
                batch_size,
            )

            # orb-v3 folds ZBL pair repulsion into predict(); orb-v2 needs it applied
            # to the same graph. Either way the short-range core is edge-based.
            if self.pair_repulsion_fn is not None:
                rep = self.pair_repulsion_fn(graph)
                forces = forces + torch.as_tensor(rep["forces"], device=forces.device, dtype=forces.dtype)
                stress = stress + self._to_cauchy(
                    torch.as_tensor(rep["stress"], device=stress.device, dtype=stress.dtype),
                    batch_size,
                )
                energy = energy + torch.as_tensor(
                    rep["energy"], device=energy.device, dtype=energy.dtype
                ).reshape(batch_size, -1).sum(dim=-1)

            node_emb, graph_emb = self._extract_representations(graph)

            forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
            stress = torch.nan_to_num(stress, nan=0.0, posinf=0.0, neginf=0.0)
            energy = torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
            return {
                "forces": forces,
                "stress": stress,
                "energy": energy.reshape(batch_size, 1),
                "node_emb": node_emb,
                "graph_emb": graph_emb,
            }
        except Exception as exc:
            logger.warning(
                "Batched ORB prediction failed for %d structures (%s). Using repulsive core.",
                batch_size,
                exc,
            )
            f_rep, s_rep, e_rep = batched_crystal_repulsion(
                cart_coords=cart_coords,
                lattices=lattices,
                atom_types=atom_types,
                num_atoms=num_atoms,
                f_max=self.max_force,
                eta=0.70,
            )
            return {
                "forces": f_rep,
                "stress": s_rep,
                "energy": e_rep,
                "node_emb": None,
                "graph_emb": None,
            }

    def _extract_representations(self, graph) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Returns per-atom and pooled graph representations from the captured GNN output.

        The raw latent has a much wider dynamic range (std ~4, max ~45) than the
        clamped forces the adapter's sibling projection sees, so it is normalized
        per atom. The normalization is affine-free, keeping the backbone
        parameter-free and the adapter's tensor shapes unchanged.
        """
        if not self.use_node_features:
            return None, None

        features = self._node_feature_cache.pop("node_features", None)
        if features is None:
            logger.warning("ORB node representations were not captured; leaving them at zero.")
            return None, None

        from orb_models.common.models import segment_ops

        features = features.detach()
        if self.normalize_node_features:
            features = F.layer_norm(features, (features.shape[-1],))

        pooled = segment_ops.aggregate_nodes(features, graph.n_node, reduction="mean")
        return self._fit_dim(features, self.node_dim), self._fit_dim(pooled, self.graph_dim)

    @staticmethod
    def _to_cauchy(stress: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Normalizes an ORB stress prediction to a (B, 3, 3) Cauchy tensor.

        ORB heads emit Voigt-6 ``[xx, yy, zz, yz, xz, xy]``; a flat 9 or an
        already-shaped (B, 3, 3) tensor is passed through.
        """
        if stress.dim() == 1:
            stress = stress.unsqueeze(0)
        if stress.dim() == 3 and stress.shape[-2:] == (3, 3):
            return stress
        if stress.shape[-1] == 9:
            return stress.reshape(batch_size, 3, 3)
        if stress.shape[-1] == 6:
            idx_map = torch.tensor(
                [[0, 5, 4],
                 [5, 1, 3],
                 [4, 3, 2]],
                device=stress.device,
                dtype=torch.long,
            )
            return stress[:, idx_map]
        raise ValueError(f"Unexpected ORB stress shape {tuple(stress.shape)}")

def build_orb_backbone(
    model_name: str = "orb-v3",
    device: str = "cpu",
    use_mock: bool = False,
    node_dim: int = 256,
    graph_dim: int = 256,
    edge_method: str = "knn_alchemi",
    min_volume_per_atom: float | None = None,
    use_node_features: bool = True,
) -> OrbBackboneWrapper:
    """Factory for constructing frozen ORB MLIP backbones."""
    return OrbBackboneWrapper(
        model_name=model_name,
        device=device,
        use_mock=use_mock,
        node_dim=node_dim,
        graph_dim=graph_dim,
        edge_method=edge_method,
        min_volume_per_atom=min_volume_per_atom,
        use_node_features=use_node_features,
    )
