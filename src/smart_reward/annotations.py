r"""Repeated-annotation sampling and unbiased log-odds statistics.

The randomized-truncation estimator implemented here uses

.. math::

    N \sim \mathrm{Geom}(1-\gamma), \qquad
    h = \sum_{k=1}^{N}
        \frac{U^+_{k,N}-U^-_{k,N}}{k\gamma^{k-1}},

where ``U+`` and ``U-`` are the order-``k`` U-statistics for all-one and
all-zero subsets.  Conditional on a pair with left-win probability ``p``,
``E[h] = log(p / (1-p))``.  The implementation updates the *weighted*
U-statistics recursively, avoiding explicit binomial coefficients and the
separate formation of potentially tiny powers ``gamma ** (k - 1)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


def _validate_gamma(gamma: float, *, allow_one: bool) -> float:
    """Validate and return a continuation probability."""

    value = float(gamma)
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if not math.isfinite(value) or value <= 0.0 or not upper_ok:
        interval = "(0, 1]" if allow_one else "(0, 1)"
        raise ValueError(f"gamma must be finite and lie in {interval}; got {gamma!r}")
    return value


def geometric_annotation_counts(
    num_pairs: int,
    gamma: float = 0.9,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Draw positive annotation counts with geometric survival ``gamma``.

    The returned counts satisfy ``P(N >= k) = gamma ** (k - 1)`` and hence
    ``E[N] = 1 / (1 - gamma)``.  No clipping is performed, because clipping
    would destroy the exact unbiasedness of the downstream estimator.

    Args:
        num_pairs: Number of independent counts to draw.
        gamma: Geometric continuation probability, strictly between zero and
            one.
        generator: Optional PyTorch random generator.
        device: Device on which to draw and return the counts.

    Returns:
        A one-dimensional ``torch.int64`` tensor of positive counts.
    """

    if isinstance(num_pairs, bool) or not isinstance(num_pairs, int) or num_pairs < 0:
        raise ValueError("num_pairs must be a non-negative integer")
    gamma_value = _validate_gamma(gamma, allow_one=False)
    if num_pairs == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    # If U is uniform on (0, 1], floor(log(U) / log(gamma)) + 1 has the
    # requested geometric law.  Clamp only the impossible-in-the-continuum
    # floating-point draw U=0; this is not a user-visible truncation of N.
    uniform = torch.rand(
        num_pairs,
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    uniform = uniform.clamp_min(torch.finfo(uniform.dtype).tiny)
    raw_counts = torch.floor(torch.log(uniform) / math.log(gamma_value)) + 1.0
    if not bool(torch.isfinite(raw_counts).all()):
        raise FloatingPointError("geometric count generation produced a non-finite value")
    max_int64 = torch.iinfo(torch.int64).max
    if bool((raw_counts > max_int64).any()):
        raise OverflowError("a sampled annotation count exceeds int64 capacity")
    return raw_counts.to(dtype=torch.int64)


@dataclass(frozen=True)
class RepeatedLabelBatch:
    """A ragged batch of repeated binary labels in flat representation.

    ``counts`` retains the original probability tensor's shape.  The
    one-dimensional ``pair_indices`` maps every entry in ``labels`` to the
    corresponding flattened pair index.
    """

    counts: torch.Tensor
    pair_indices: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        if self.counts.dtype != torch.int64 or self.pair_indices.dtype != torch.int64:
            raise TypeError("counts and pair_indices must have dtype torch.int64")
        if self.labels.dtype != torch.int64:
            raise TypeError("labels must have dtype torch.int64")
        if self.pair_indices.ndim != 1 or self.labels.ndim != 1:
            raise ValueError("pair_indices and labels must be one-dimensional")
        if self.pair_indices.shape != self.labels.shape:
            raise ValueError("pair_indices and labels must have identical shapes")
        if (
            self.counts.device != self.labels.device
            or self.labels.device != self.pair_indices.device
        ):
            raise ValueError("all repeated-label tensors must be on the same device")
        if bool((self.counts < 1).any()):
            raise ValueError("every pair must have at least one annotation")
        if int(self.counts.sum().item()) != self.labels.numel():
            raise ValueError("sum(counts) must equal the number of flat labels")
        if self.pair_indices.numel() > 0:
            if bool((self.pair_indices < 0).any()) or int(
                self.pair_indices.max()
            ) >= self.counts.numel():
                raise ValueError("pair_indices contains an out-of-range pair index")
            observed_counts = torch.bincount(
                self.pair_indices,
                minlength=self.counts.numel(),
            ).reshape(self.counts.shape)
            if not torch.equal(observed_counts, self.counts):
                raise ValueError("the pair_indices histogram must equal counts")
        if bool(((self.labels != 0) & (self.labels != 1)).any()):
            raise ValueError("labels must be binary")

    @property
    def wins(self) -> torch.Tensor:
        """Return left-win counts with the same shape as ``counts``."""

        flat_wins = torch.zeros(
            self.counts.numel(),
            dtype=torch.int64,
            device=self.counts.device,
        )
        flat_wins.scatter_add_(0, self.pair_indices, self.labels)
        return flat_wins.reshape(self.counts.shape)

    def logit_estimates(self, gamma: float = 0.9) -> torch.Tensor:
        """Compute randomized-truncation U-statistic estimates for all pairs."""

        return randomized_truncation_u_statistic_from_counts(
            self.wins,
            self.counts,
            gamma=gamma,
        )


def sample_geometric_repeated_labels(
    probabilities: torch.Tensor,
    gamma: float = 0.9,
    *,
    generator: torch.Generator | None = None,
    max_total_annotations: int | None = None,
) -> RepeatedLabelBatch:
    """Sample geometric counts and conditionally independent Bernoulli labels.

    Args:
        probabilities: Tensor of left-win probabilities in ``[0, 1]``.
        gamma: Geometric continuation probability.
        generator: Optional PyTorch random generator.
        max_total_annotations: Optional fail-fast memory guard.  If the sampled
            total exceeds this value, the function raises instead of clipping;
            clipping would invalidate unbiasedness.

    Returns:
        A :class:`RepeatedLabelBatch` containing every sampled label.
    """

    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a torch.Tensor")
    if not probabilities.is_floating_point():
        raise TypeError("probabilities must have a floating-point dtype")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite")
    if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
        raise ValueError("probabilities must lie in [0, 1]")
    if max_total_annotations is not None and (
            isinstance(max_total_annotations, bool)
            or not isinstance(max_total_annotations, int)
            or max_total_annotations < 0
    ):
        raise ValueError("max_total_annotations must be a non-negative integer")

    flat_probabilities = probabilities.reshape(-1)
    counts = geometric_annotation_counts(
        flat_probabilities.numel(),
        gamma=gamma,
        generator=generator,
        device=probabilities.device,
    )
    total = int(counts.sum().item())
    if max_total_annotations is not None and total > max_total_annotations:
        raise RuntimeError(
            f"sampled {total} annotations, exceeding max_total_annotations="
            f"{max_total_annotations}; no samples were clipped"
        )

    pair_indices = torch.repeat_interleave(
        torch.arange(flat_probabilities.numel(), device=probabilities.device),
        counts,
    )
    label_probabilities = flat_probabilities[pair_indices]
    labels = torch.bernoulli(label_probabilities, generator=generator).to(torch.int64)
    return RepeatedLabelBatch(
        counts=counts.reshape(probabilities.shape),
        pair_indices=pair_indices,
        labels=labels,
    )


def _validate_count_tensors(wins: torch.Tensor, totals: torch.Tensor) -> None:
    if not isinstance(wins, torch.Tensor) or not isinstance(totals, torch.Tensor):
        raise TypeError("wins and totals must be torch.Tensor objects")
    if wins.shape != totals.shape:
        raise ValueError("wins and totals must have identical shapes")
    if wins.device != totals.device:
        raise ValueError("wins and totals must be on the same device")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if wins.dtype not in integer_dtypes or totals.dtype not in integer_dtypes:
        raise TypeError("wins and totals must have integer dtypes")
    if bool((totals < 1).any()):
        raise ValueError("totals must be at least one")
    if bool(((wins < 0) | (wins > totals)).any()):
        raise ValueError("wins must satisfy 0 <= wins <= totals")


def randomized_truncation_u_statistic_from_counts(
    wins: torch.Tensor,
    totals: torch.Tensor,
    gamma: float = 0.9,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Compute the randomized-truncation log-odds U-statistic from counts.

    ``totals`` must be the realized geometric truncation levels paired with
    the same ``gamma``.  Passing ``gamma=1`` is also supported and yields the
    finite-order statistic used when a fixed number of labels is collected;
    that finite-order version is generally biased for the full log-odds.

    The recurrence maintains ``U_k / gamma**(k-1)`` directly.  This is more
    stable than evaluating binomial coefficients or dividing two separately
    underflowing quantities.
    """

    _validate_count_tensors(wins, totals)
    gamma_value = _validate_gamma(gamma, allow_one=True)
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype")
    if wins.numel() == 0:
        return torch.empty_like(wins, dtype=dtype)

    wins_i64 = wins.to(torch.int64)
    totals_i64 = totals.to(torch.int64)
    losses_i64 = totals_i64 - wins_i64
    weighted_positive = torch.ones_like(wins_i64, dtype=dtype)
    weighted_negative = torch.ones_like(wins_i64, dtype=dtype)
    estimate = torch.zeros_like(weighted_positive)
    max_order = int(totals_i64.max().item())

    for order in range(1, max_order + 1):
        active = totals_i64 >= order
        denominator = (totals_i64 - order + 1).clamp_min(1).to(dtype)
        continuation = 1.0 if order == 1 else gamma_value

        positive_ratio = (wins_i64 - order + 1).clamp_min(0).to(dtype) / denominator
        negative_ratio = (losses_i64 - order + 1).clamp_min(0).to(dtype) / denominator
        next_positive = weighted_positive * positive_ratio / continuation
        next_negative = weighted_negative * negative_ratio / continuation
        weighted_positive = torch.where(active, next_positive, weighted_positive)
        weighted_negative = torch.where(active, next_negative, weighted_negative)
        estimate = estimate + torch.where(
            active,
            (weighted_positive - weighted_negative) / float(order),
            torch.zeros_like(estimate),
        )

    if not bool(torch.isfinite(estimate).all()):
        raise FloatingPointError(
            "the U-statistic overflowed; use float64 and avoid extreme gamma/count combinations"
        )
    return estimate


def repeated_labels_to_h(
    labels: torch.Tensor | Sequence[int | bool],
    gamma: float = 0.9,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Compute one log-odds U-statistic from a non-empty label sequence."""

    label_tensor = torch.as_tensor(labels)
    if label_tensor.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if label_tensor.numel() == 0:
        raise ValueError("at least one repeated label is required")
    if label_tensor.is_floating_point() and not bool(torch.isfinite(label_tensor).all()):
        raise ValueError("labels must be finite")
    if bool(((label_tensor != 0) & (label_tensor != 1)).any()):
        raise ValueError("labels must contain only zeros and ones")

    wins = label_tensor.to(torch.int64).sum()
    total = torch.tensor(
        label_tensor.numel(),
        dtype=torch.int64,
        device=label_tensor.device,
    )
    return randomized_truncation_u_statistic_from_counts(
        wins,
        total,
        gamma=gamma,
        dtype=dtype,
    )


__all__ = [
    "RepeatedLabelBatch",
    "geometric_annotation_counts",
    "randomized_truncation_u_statistic_from_counts",
    "repeated_labels_to_h",
    "sample_geometric_repeated_labels",
]
