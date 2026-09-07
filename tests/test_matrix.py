"""Tests for diffcsp.core.matrix (logm, expm, sqrtm)."""

import torch

from diffcsp.core.matrix import expm, logm, sqrtm


def test_matrix_exponential_and_logarithm_roundtrip():
    # Symmetric positive definite matrix
    a = torch.tensor(
        [
            [[2.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.5]],
            [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        ]
    )
    log_a = logm(a)
    exp_log_a = expm(log_a)
    assert torch.allclose(a, exp_log_a, atol=1e-4), "expm(logm(A)) must reconstruct A"


def test_matrix_sqrt_correctness():
    # Symmetric positive definite matrix
    s = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 16.0]],
            [[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 2.5]],
        ]
    )
    sq = sqrtm(s)
    sq_sq = sq @ sq
    assert torch.allclose(s, sq_sq, atol=1e-4), "sqrtm(S) @ sqrtm(S) must equal S"


def test_no_in_place_mutation():
    """Verify that logm and sqrtm do NOT mutate caller tensors in-place."""
    bad_mat = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # det < 0
        ]
    )
    orig = bad_mat.clone()
    _ = logm(bad_mat)
    assert torch.equal(bad_mat, orig), "logm must not mutate input tensor in-place"

    _ = sqrtm(bad_mat)
    assert torch.equal(bad_mat, orig), "sqrtm must not mutate input tensor in-place"
