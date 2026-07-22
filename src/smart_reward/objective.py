"""The empirical Fisher-GMM/SRM+ objective with fixed normalization.

The factor convention in this module is deliberately explicit:

``m = Z.T @ (t - h) / (2 n)`` and
``L = m.T @ (F + damping I)^(-1) @ m / (2 beta)``.

Consequently, when an outer training loss is implemented as a mean over
edges, its envelope-theorem weight is ``(Z @ v) / (2 beta)``.
"""

from __future__ import annotations

import math

import torch


def _validate_positive_beta(beta: float) -> float:
    value = float(beta)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    return value


def _validate_edge_inputs(
    edge_features: torch.Tensor,
    margins: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int]:
    if not all(isinstance(item, torch.Tensor) for item in (edge_features, margins, targets)):
        raise TypeError("edge_features, margins, and targets must be torch.Tensor objects")
    if edge_features.ndim != 2:
        raise ValueError("edge_features must have shape (num_edges, dimension)")
    if margins.ndim != 1 or targets.ndim != 1:
        raise ValueError("margins and targets must be one-dimensional")
    num_edges, dimension = edge_features.shape
    if num_edges < 1 or dimension < 1:
        raise ValueError("edge_features dimensions must both be positive")
    if margins.shape != (num_edges,) or targets.shape != (num_edges,):
        raise ValueError("margins and targets must have length num_edges")
    if not edge_features.is_floating_point():
        raise TypeError("edge_features must have a floating-point dtype")
    if not margins.is_floating_point() or not targets.is_floating_point():
        raise TypeError("margins and targets must have floating-point dtypes")
    for name, tensor in (("margins", margins), ("targets", targets)):
        if tensor.dtype != edge_features.dtype or tensor.device != edge_features.device:
            raise ValueError(f"{name} must have the same dtype and device as edge_features")
    if not all(bool(torch.isfinite(item).all()) for item in (edge_features, margins, targets)):
        raise ValueError("edge_features, margins, and targets must be finite")
    return num_edges, dimension


def empirical_moment(
    edge_features: torch.Tensor,
    margins: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return ``Z.T @ (margins - targets) / (2 * num_edges)``."""

    num_edges, _ = _validate_edge_inputs(edge_features, margins, targets)
    return edge_features.mT @ (margins - targets) / (2.0 * num_edges)


def dual_loss(
    moment: torch.Tensor,
    optimal_direction: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Evaluate ``m.T v / (2 beta)`` for ``v = (F + lambda I)^-1 m``.

    This function assumes ``optimal_direction`` solves the stated linear
    system.  Use :func:`dual_saddle_value` to evaluate an arbitrary direction.
    """

    beta_value = _validate_positive_beta(beta)
    _validate_matching_vectors(moment, optimal_direction)
    return 0.5 * torch.dot(moment, optimal_direction) / beta_value


def dual_saddle_value(
    moment: torch.Tensor,
    direction: torch.Tensor,
    operator_direction: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Evaluate ``(v.T m - 0.5 v.T A v) / beta`` for arbitrary ``v``."""

    beta_value = _validate_positive_beta(beta)
    _validate_matching_vectors(moment, direction)
    _validate_matching_vectors(moment, operator_direction)
    return (
        torch.dot(direction, moment) - 0.5 * torch.dot(direction, operator_direction)
    ) / beta_value


def _validate_matching_vectors(first: torch.Tensor, second: torch.Tensor) -> None:
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("inputs must be torch.Tensor objects")
    if first.ndim != 1 or first.numel() < 1:
        raise ValueError("inputs must be non-empty one-dimensional tensors")
    if second.shape != first.shape:
        raise ValueError("vector shapes must match")
    if not first.is_floating_point() or not second.is_floating_point():
        raise TypeError("vectors must have floating-point dtypes")
    if first.dtype != second.dtype or first.device != second.device:
        raise ValueError("vectors must have the same dtype and device")
    if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
        raise ValueError("vectors must be finite")


def envelope_weights(
    edge_features: torch.Tensor,
    optimal_direction: torch.Tensor,
    *,
    beta: float = 1.0,
    detach_direction: bool = True,
) -> torch.Tensor:
    """Return per-edge weights for a mean-reduced envelope surrogate.

    For ``m = Z.T(t-h)/(2n)`` and ``v = A^-1 m``, the exact gradient is

    ``dL/dt_i = (z_i.T v) / (2 beta n)``.

    Thus ``mean(envelope_weights(Z, v) * (t-h))`` has precisely this gradient.
    The direction is detached by default because the envelope theorem holds it
    fixed during the outer reward-model update.
    """

    beta_value = _validate_positive_beta(beta)
    if not isinstance(edge_features, torch.Tensor) or not isinstance(
        optimal_direction, torch.Tensor
    ):
        raise TypeError("edge_features and optimal_direction must be torch.Tensor objects")
    if edge_features.ndim != 2 or edge_features.shape[0] < 1 or edge_features.shape[1] < 1:
        raise ValueError("edge_features must have non-empty shape (num_edges, dimension)")
    if optimal_direction.shape != (edge_features.shape[1],):
        raise ValueError("optimal_direction length must match the edge feature dimension")
    if not edge_features.is_floating_point() or not optimal_direction.is_floating_point():
        raise TypeError("edge_features and optimal_direction must be floating point")
    if (
        edge_features.dtype != optimal_direction.dtype
        or edge_features.device != optimal_direction.device
    ):
        raise ValueError("edge_features and optimal_direction must share dtype and device")
    if not bool(torch.isfinite(edge_features).all()) or not bool(
        torch.isfinite(optimal_direction).all()
    ):
        raise ValueError("edge_features and optimal_direction must be finite")

    direction = optimal_direction.detach() if detach_direction else optimal_direction
    return 0.5 * (edge_features @ direction) / beta_value


def envelope_surrogate(
    margins: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return ``mean(weights * (margins - targets))`` with strict checks."""

    if not all(isinstance(item, torch.Tensor) for item in (margins, targets, weights)):
        raise TypeError("margins, targets, and weights must be torch.Tensor objects")
    if margins.ndim != 1 or margins.numel() < 1:
        raise ValueError("margins must be a non-empty one-dimensional tensor")
    if targets.shape != margins.shape or weights.shape != margins.shape:
        raise ValueError("margins, targets, and weights must have identical shapes")
    for name, tensor in (("targets", targets), ("weights", weights)):
        if tensor.dtype != margins.dtype or tensor.device != margins.device:
            raise ValueError(f"{name} must have the same dtype and device as margins")
    if not all(tensor.is_floating_point() for tensor in (margins, targets, weights)):
        raise TypeError("margins, targets, and weights must be floating point")
    if not all(bool(torch.isfinite(item).all()) for item in (margins, targets, weights)):
        raise ValueError("margins, targets, and weights must be finite")
    return torch.mean(weights * (margins - targets))


# Explicit domain name retained for callers that prefer it.
srm_dual_loss = dual_loss


__all__ = [
    "dual_loss",
    "dual_saddle_value",
    "empirical_moment",
    "envelope_surrogate",
    "envelope_weights",
    "srm_dual_loss",
]
