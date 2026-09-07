"""Robust scatter and segment operations with automatic fallback.

Provides scatter, segment_coo, and segment_csr operations compatible with both
torch_scatter and torch_geometric.utils.scatter, eliminating hard dependencies
on custom-compiled C++ wheels.
"""

import torch

try:
    from torch_scatter import scatter as _ts_scatter
    from torch_scatter import segment_coo as _ts_segment_coo
    from torch_scatter import segment_csr as _ts_segment_csr

    _HAS_TORCH_SCATTER = True
except ImportError:
    _HAS_TORCH_SCATTER = False
    from torch_geometric.utils import scatter as _pyg_scatter


def scatter(
    src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None, reduce: str = "mean"
) -> torch.Tensor:
    """Aggregates all values from the `src` tensor at indices specified in `index`.

    Args:
        src: The source tensor.
        index: The indices of elements to scatter.
        dim: The axis along which to index.
        dim_size: The target output size along `dim`. If None, determined by index.max() + 1.
        reduce: Reduction operation ('sum', 'mean', 'max', 'min').

    Returns:
        The reduced output tensor.
    """
    if _HAS_TORCH_SCATTER:
        return _ts_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


def segment_coo(
    src: torch.Tensor, index: torch.Tensor, dim_size: int | None = None, reduce: str = "sum"
) -> torch.Tensor:
    """Segments elements in `src` based on sorted `index` tensor (COO format)."""
    if _HAS_TORCH_SCATTER:
        return _ts_segment_coo(src, index, dim_size=dim_size, reduce=reduce)
    return scatter(src, index, dim=0, dim_size=dim_size, reduce=reduce)


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    """Segments elements in `src` based on CSR pointer indices (`indptr`)."""
    if _HAS_TORCH_SCATTER:
        return _ts_segment_csr(src, indptr, reduce=reduce)

    # Deconstruct CSR indptr to element-wise cluster indices
    diff = indptr[1:] - indptr[:-1]
    idx = torch.repeat_interleave(torch.arange(len(diff), device=indptr.device, dtype=torch.long), diff)
    dim_size = len(diff)
    return scatter(src, idx, dim=0, dim_size=dim_size, reduce=reduce)
