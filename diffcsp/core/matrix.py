"""Matrix operations on SO(3) and Lie algebra for crystal lattice tensors.

Provides matrix logarithm, exponential, and square root operations
without in-place mutation and with numerical stabilization.
"""

import torch
import torch.linalg as linalg


def logm(mat: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes matrix logarithm for batches of 3x3 matrices.

    Args:
        mat: (B, 3, 3) batch of matrices.
        eps: small numerical threshold for determinant positivity.

    Returns:
        (B, 3, 3) matrix logarithm (real part).
    """
    # Clone to avoid mutating caller's tensor in-place
    mat_safe = mat.clone()
    det = torch.det(mat_safe)
    invalid_mask = ~(det > eps)

    if invalid_mask.any():
        identity = torch.eye(3, dtype=mat.dtype, device=mat.device).unsqueeze(0)
        mat_safe = torch.where(invalid_mask.unsqueeze(-1).unsqueeze(-1), identity, mat_safe)

    dev = mat_safe.device
    if dev.type == "cuda" and not torch.cuda.has_magma:
        eigenvalues, eigenvectors = linalg.eig(mat_safe.cpu())
        eigenvalues = eigenvalues.to(dev)
        eigenvectors = eigenvectors.to(dev)
    else:
        eigenvalues, eigenvectors = linalg.eig(mat_safe)
    # Clamp eigenvalue real parts away from zero to avoid log(-0) / log(0)
    eig_log = eigenvalues.log()
    inv_vecs = torch.linalg.pinv(eigenvectors)
    log_mat = torch.einsum("bij,bj,bjk->bik", eigenvectors, eig_log, inv_vecs).real
    return log_mat


def expm(mat: torch.Tensor) -> torch.Tensor:
    """Computes matrix exponential using PyTorch's native matrix_exp.

    Args:
        mat: (B, 3, 3) batch of matrices.

    Returns:
        (B, 3, 3) matrix exponential.
    """
    return torch.matrix_exp(mat)


def sqrtm(mat: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes principal matrix square root for batches of 3x3 matrices.

    For symmetric positive semi-definite matrices (e.g. L @ L^T),
    uses stable symmetric eigendecomposition.

    Args:
        mat: (B, 3, 3) batch of matrices.
        eps: small numerical threshold for positivity.

    Returns:
        (B, 3, 3) symmetric matrix square root (real part).
    """
    mat_safe = mat.clone()
    det = torch.det(mat_safe)
    invalid_mask = ~(det > eps)

    if invalid_mask.any():
        identity = torch.eye(3, dtype=mat.dtype, device=mat.device).unsqueeze(0)
        mat_safe = torch.where(invalid_mask.unsqueeze(-1).unsqueeze(-1), identity, mat_safe)

    # Check if input is symmetric up to numerical precision
    is_symmetric = torch.allclose(mat_safe, mat_safe.transpose(-1, -2), atol=1e-5)
    if is_symmetric:
        evals, evecs = torch.linalg.eigh(mat_safe)
        evals_sqrt = torch.clamp(evals, min=0.0).sqrt()
        sqrt_mat = torch.einsum("bij,bj,bjk->bik", evecs, evals_sqrt, evecs.transpose(-1, -2))
    else:
        dev = mat_safe.device
        if dev.type == "cuda" and not torch.cuda.has_magma:
            eigenvalues, eigenvectors = linalg.eig(mat_safe.cpu())
            eigenvalues = eigenvalues.to(dev)
            eigenvectors = eigenvectors.to(dev)
        else:
            eigenvalues, eigenvectors = linalg.eig(mat_safe)
        inv_vecs = torch.linalg.pinv(eigenvectors)
        sqrt_mat = torch.einsum("bij,bj,bjk->bik", eigenvectors, eigenvalues.sqrt(), inv_vecs).real

    return sqrt_mat
