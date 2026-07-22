"""Controlled-oracle calibration and Bradley--Terry probabilities.

The calibration is deliberately fit once from raw training-node scores.  The
returned transform contains only frozen scalar statistics and can therefore be
applied unchanged to validation and test nodes without split leakage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch

_MAD_NORMALIZATION = 1.4826
_SCALE_FLOOR = 1.0e-6


def _validate_floating_tensor(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.numel() < 1:
        raise ValueError(f"{name} must be non-empty")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class RobustOracleTransform:
    """Frozen median/MAD statistics for the controlled oracle transform."""

    b: float
    tau: float

    def __post_init__(self) -> None:
        if isinstance(self.b, bool) or not isinstance(self.b, Real):
            raise TypeError("b must be a real scalar")
        if isinstance(self.tau, bool) or not isinstance(self.tau, Real):
            raise TypeError("tau must be a real scalar")

        b = float(self.b)
        tau = float(self.tau)
        if not math.isfinite(b):
            raise ValueError("b must be finite")
        if not math.isfinite(tau):
            raise ValueError("tau must be finite")
        if tau < _SCALE_FLOOR:
            raise ValueError(f"tau must be at least {_SCALE_FLOOR}")
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "tau", tau)

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Map raw scores into ``[-log(3)/2, log(3)/2]``.

        Python scalar calibration statistics do not impose a device or dtype:
        the returned tensor consequently retains both properties of ``scores``.
        """

        _validate_floating_tensor(scores, name="scores")
        half_log_three = torch.log(scores.new_tensor(3.0)) / 2.0
        transformed = half_log_three * torch.tanh((scores - self.b) / self.tau)
        if not bool(torch.isfinite(transformed).all()):
            raise ValueError("transformed oracle scores must be finite")
        return transformed

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`transform`."""

        return self.transform(scores)


def fit_robust_oracle_transform(train_scores: torch.Tensor) -> RobustOracleTransform:
    """Fit the frozen robust transform using training scores only.

    ``b`` is the sample median and ``tau`` is the Gaussian-consistent scaled
    median absolute deviation, floored at ``1e-6``.
    """

    _validate_floating_tensor(train_scores, name="train_scores")
    detached_scores = train_scores.detach()
    center = torch.median(detached_scores)
    median_absolute_deviation = torch.median(torch.abs(detached_scores - center))
    scaled_mad = _MAD_NORMALIZATION * median_absolute_deviation
    if not bool(torch.isfinite(center)) or not bool(torch.isfinite(scaled_mad)):
        raise ValueError("robust oracle calibration statistics must be finite")
    # Apply the floor after extracting the statistic so it remains exactly the
    # documented Python value even when train_scores uses float16/float32.
    scale = max(_SCALE_FLOOR, scaled_mad.item())
    return RobustOracleTransform(b=center.item(), tau=scale)


def pair_margins(left_scores: torch.Tensor, right_scores: torch.Tensor) -> torch.Tensor:
    """Return oriented pair margins as ``left_scores - right_scores``."""

    _validate_floating_tensor(left_scores, name="left_scores")
    _validate_floating_tensor(right_scores, name="right_scores")
    if left_scores.shape != right_scores.shape:
        raise ValueError("left_scores and right_scores must have identical shapes")
    if left_scores.dtype != right_scores.dtype or left_scores.device != right_scores.device:
        raise ValueError("left_scores and right_scores must have the same dtype and device")

    margins = left_scores - right_scores
    if not bool(torch.isfinite(margins).all()):
        raise ValueError("pair margins must be finite")
    return margins


def btl_probabilities(margins: torch.Tensor) -> torch.Tensor:
    """Convert controlled-oracle margins to left-win BTL probabilities.

    Scores transformed by :class:`RobustOracleTransform` imply margins in
    ``[-log(3), log(3)]`` and hence probabilities in ``[0.25, 0.75]``.  The
    explicit check catches accidental use of raw, uncalibrated oracle margins.
    """

    _validate_floating_tensor(margins, name="margins")
    probabilities = torch.sigmoid(margins)
    tolerance = 8.0 * torch.finfo(probabilities.dtype).eps
    lower_bound = 0.25 - tolerance
    upper_bound = 0.75 + tolerance
    if bool((probabilities < lower_bound).any()) or bool(
        (probabilities > upper_bound).any()
    ):
        raise AssertionError("controlled-oracle BTL probabilities must lie in [0.25, 0.75]")
    return probabilities


__all__ = [
    "RobustOracleTransform",
    "btl_probabilities",
    "fit_robust_oracle_transform",
    "pair_margins",
]
