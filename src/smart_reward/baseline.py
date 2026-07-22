"""Bradley--Terry baseline objectives for repeated binary preferences."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def repeated_btl_nll(
    margins: torch.Tensor,
    left_wins: torch.Tensor,
    num_annotations: torch.Tensor,
) -> torch.Tensor:
    """Return the label-level mean BTL negative log-likelihood.

    For edge ``i`` with margin ``t_i``, ``S_i`` left wins and ``N_i`` total
    labels, the summed loss is ``N_i * softplus(t_i) - S_i * t_i``.  This is
    exactly equivalent to expanding every repeated Bernoulli label.  In
    contrast to ProRM+, the baseline intentionally weights an edge by ``N_i``.
    """

    if not all(isinstance(value, torch.Tensor) for value in (margins, left_wins, num_annotations)):
        raise TypeError("all inputs must be torch.Tensor objects")
    if margins.ndim != 1 or margins.numel() == 0:
        raise ValueError("margins must be a non-empty one-dimensional tensor")
    if left_wins.shape != margins.shape or num_annotations.shape != margins.shape:
        raise ValueError("all inputs must have the same shape")
    if not margins.is_floating_point():
        raise TypeError("margins must be floating point")
    if not bool(torch.isfinite(margins).all()):
        raise ValueError("margins must be finite")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if left_wins.dtype not in integer_dtypes or num_annotations.dtype not in integer_dtypes:
        raise TypeError("left_wins and num_annotations must use integer dtypes")
    if left_wins.device != margins.device or num_annotations.device != margins.device:
        raise ValueError("all inputs must be on the same device")
    if bool((num_annotations < 1).any()):
        raise ValueError("every edge must have at least one annotation")
    if bool(((left_wins < 0) | (left_wins > num_annotations)).any()):
        raise ValueError("left_wins must satisfy 0 <= left_wins <= num_annotations")

    counts = num_annotations.to(dtype=margins.dtype)
    wins = left_wins.to(dtype=margins.dtype)
    per_edge_sum = counts * functional.softplus(margins) - wins * margins
    return per_edge_sum.sum() / counts.sum()


__all__ = ["repeated_btl_nll"]
