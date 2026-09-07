"""Tests for diffcsp.core.scatter."""

import torch

from diffcsp.core.scatter import scatter, segment_coo, segment_csr


def test_scatter_mean_sum_max():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 0, 1])

    out_sum = scatter(src, index, dim=0, reduce="sum")
    expected_sum = torch.tensor([[4.0, 6.0], [5.0, 6.0]])
    assert torch.allclose(out_sum, expected_sum)

    out_mean = scatter(src, index, dim=0, reduce="mean")
    expected_mean = torch.tensor([[2.0, 3.0], [5.0, 6.0]])
    assert torch.allclose(out_mean, expected_mean)


def test_segment_csr():
    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    indptr = torch.tensor([0, 2, 5])
    out = segment_csr(src, indptr, reduce="sum")
    expected = torch.tensor([3.0, 12.0])
    assert torch.allclose(out, expected)


def test_segment_coo():
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    index = torch.tensor([0, 0, 1, 1])
    out = segment_coo(src, index, dim_size=2, reduce="sum")
    expected = torch.tensor([3.0, 7.0])
    assert torch.allclose(out, expected)
