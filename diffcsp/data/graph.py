"""Crystal graph extraction from CIFs and pyXtal seed representations.

Unified implementation of CrystalNN / PyMatGen graph extraction,
Wyckoff site operations, and periodic boundary condition neighbor graphs.
"""

import logging
import warnings
from typing import Any

import numpy as np
import torch
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Lattice, Structure
from pyxtal import pyxtal
from scipy.linalg import pinv

from diffcsp.core.scatter import segment_coo, segment_csr
from diffcsp.data.transforms import lattice_params_to_matrix_torch

logger = logging.getLogger(__name__)

# Standard CrystalNN strategy
_CRYSTAL_NN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)
_CRYSTAL_NN_TMP = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=10
)


def build_crystal(crystal_str: str, niggli: bool = True, primitive: bool = False) -> Structure:
    """Builds a canonical pymatgen Structure from CIF string.

    Args:
        crystal_str: CIF file string.
        niggli: If True, performs Niggli reduction.
        primitive: If True, finds primitive cell.

    Returns:
        Canonical pymatgen Structure.
    """
    crystal = Structure.from_str(crystal_str, fmt="cif")
    if primitive:
        crystal = crystal.get_primitive_structure()
    if niggli:
        crystal = crystal.get_reduced_structure()

    return Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )


def build_crystal_graph(crystal: Any, graph_method: str = "crystalnn", tol: float = 0.1) -> tuple:
    """Extracts crystal graph, symmetry operations, and Wyckoff site anchors.

    Args:
        crystal: Either a pymatgen Structure, dictionary, or pyxtal seed.
        graph_method: 'crystalnn' or 'none'.
        tol: Tolerance for pyxtal symmetry detection.

    Returns:
        Tuple of (frac_coords, atom_types, lengths, angles, edge_indices,
                  to_jimages, num_atoms, operation, inv_rotation, anchor_idxs, space_group)
    """
    c = pyxtal()
    if isinstance(crystal, dict):
        c.from_random(**crystal, max_count=30)
    else:
        c.from_seed(crystal, tol=tol)

    space_group = c.group.number
    pmg_crystal = c.to_pymatgen(resort=False)

    crystal_graph = None
    if graph_method == "crystalnn":
        try:
            crystal_graph = StructureGraph.from_local_env_strategy(pmg_crystal, _CRYSTAL_NN)
        except Exception:
            crystal_graph = StructureGraph.from_local_env_strategy(pmg_crystal, _CRYSTAL_NN_TMP)
    elif graph_method != "none":
        raise NotImplementedError(f"Unsupported graph method: {graph_method}")

    operation = []
    inv_rotation = []
    anchor_idxs = []

    for site in c.atom_sites:
        anchor_idxs.extend([len(operation) for _ in site.wp.ops])
        operation.extend([op.affine_matrix for op in site.wp.ops])
        inv_rotation.extend([pinv(op.rotation_matrix) for op in site.wp.ops])

    if len(operation) != len(pmg_crystal.frac_coords):
        raise ValueError(
            f"Operation count ({len(operation)}) != coordinate count ({len(pmg_crystal.frac_coords)})"
        )

    operation = np.stack(operation)
    inv_rotation = np.stack(inv_rotation)
    anchor_idxs = np.array(anchor_idxs)

    frac_coords = pmg_crystal.frac_coords
    atom_types = pmg_crystal.atomic_numbers
    lattice_params = pmg_crystal.lattice.parameters
    lengths = np.array(lattice_params[:3])
    angles = np.array(lattice_params[3:])

    edge_indices, to_jimages = [], []
    if crystal_graph is not None:
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    edge_indices = np.array(edge_indices) if edge_indices else np.empty((0, 2), dtype=int)
    to_jimages = np.array(to_jimages) if to_jimages else np.empty((0, 3), dtype=int)
    num_atoms = atom_types.shape[0]

    return (
        frac_coords,
        atom_types,
        lengths,
        angles,
        edge_indices,
        to_jimages,
        num_atoms,
        operation,
        inv_rotation,
        anchor_idxs,
        space_group,
    )


def process_one(
    row: Any, niggli: bool = True, primitive: bool = False, graph_method: str = "crystalnn"
) -> dict[str, Any] | None:
    """Processes a single dataset record (either DataFrame row with CIF or Wyckoff dict)."""
    try:
        if isinstance(row, dict) and "cif" not in row:
            # Wyckoff dict input
            graph_arrays = build_crystal_graph(row, graph_method=graph_method)
            return {"graph_arrays": graph_arrays}

        # CIF input
        crystal_str = row["cif"]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Issues encountered while parsing CIF",
                category=UserWarning,
                module="pymatgen.io.cif",
            )
            crystal = build_crystal(crystal_str, niggli=niggli, primitive=primitive)

        graph_arrays = build_crystal_graph(crystal, graph_method=graph_method)
        return {
            "mp_id": row.get("material_id", ""),
            "cif": crystal_str,
            "graph_arrays": graph_arrays,
        }
    except (RuntimeError, TypeError, ValueError) as e:
        logger.warning("Error processing crystal: %s", e)
        return None


def get_max_neighbors_mask(
    natoms: torch.Tensor, index: torch.Tensor, atom_distance: torch.Tensor, max_num_neighbors_threshold: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filters edges so each atom has at most `max_num_neighbors_threshold` neighbors."""
    device = natoms.device
    num_atoms = natoms.sum()

    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    if max_num_neighbors <= max_num_neighbors_threshold or max_num_neighbors_threshold <= 0:
        mask_num_neighbors = torch.ones_like(index, dtype=torch.bool, device=device)
        return mask_num_neighbors, num_neighbors_image

    return torch.ones_like(index, dtype=torch.bool, device=device), num_neighbors_image


def radius_graph_pbc(
    pos: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    natoms: torch.Tensor,
    radius: float,
    max_num_neighbors_threshold: int | None,
    device: torch.device,
    lattices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes periodic boundary condition radius graph for a batch of crystals."""
    batch_size = len(natoms)
    cell = lattice_params_to_matrix_torch(lengths, angles) if lattices is None else lattices
    atom_pos = pos

    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(num_atoms_per_image, num_atoms_per_image_sqr)

    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor") + index_offset_expand
    )
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand

    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    cross_a2a3 = torch.linalg.cross(cell[:, 1], cell[:, 2])
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    min_dist_a1 = (1.0 / inv_min_dist_a1).reshape(-1, 1)

    cross_a3a1 = torch.linalg.cross(cell[:, 2], cell[:, 0])
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    min_dist_a2 = (1.0 / inv_min_dist_a2).reshape(-1, 1)

    cross_a1a2 = torch.linalg.cross(cell[:, 0], cell[:, 1])
    inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
    min_dist_a3 = (1.0 / inv_min_dist_a3).reshape(-1, 1)

    max_rep = torch.ones(3, dtype=torch.long, device=device)
    min_dist = torch.cat([min_dist_a1, min_dist_a2, min_dist_a3], dim=-1)

    cells_per_dim = [torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep]
    unit_cell = torch.cat([g.reshape(-1, 1) for g in torch.meshgrid(*cells_per_dim, indexing="ij")], dim=-1)

    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, num_atoms_per_image_sqr, dim=0)

    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    pos2 = pos2 + pbc_offsets_per_atom

    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1).view(-1)
    radius_real = min_dist.min(dim=-1)[0] + 0.01
    radius_real = torch.repeat_interleave(radius_real, num_atoms_per_image_sqr * num_cells)

    mask_within_radius = torch.le(atom_distance_sqr, radius_real * radius_real)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)

    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)).view(
        -1, 3
    )
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    if max_num_neighbors_threshold is not None:
        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=natoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )
        if not torch.all(mask_num_neighbors):
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)
            unit_cell = torch.masked_select(
                unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
            ).view(-1, 3)
    else:
        ones = index1.new_ones(1).expand_as(index1)
        num_neighbors = segment_coo(ones, index1, dim_size=natoms.sum())
        image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
        image_indptr[1:] = torch.cumsum(natoms, dim=0)
        num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    edge_index = torch.stack((index2, index1))
    return edge_index, unit_cell, num_neighbors_image


def repeat_blocks(
    sizes: torch.Tensor,
    repeats: Any,
    continuous_indexing: bool = True,
    start_idx: int = 0,
    block_inc: Any = 0,
    repeat_inc: Any = 0,
) -> torch.Tensor:
    """Repeats blocks of indices with configurable offsets."""
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = (repeats[0] == 0).item()
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    r1 = torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), repeats)
    total_len = (sizes * repeats).sum()

    id_ar = torch.ones(total_len, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(block_inc[: r1[-1]], indptr, reduce="sum")
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            insert_val[idx] = 1
        insert_val[idx] += block_inc

    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        repeat_inc_inner = (
            repeat_inc[repeats > 0][:-1] if isinstance(repeats, torch.Tensor) else repeat_inc[:-1]
        )
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    repeats_inner = repeats[repeats > 0][:-1] if isinstance(repeats, torch.Tensor) else repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    id_ar[insert_index] = insert_val
    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    id_ar[0] += start_idx
    return id_ar.cumsum(0)
