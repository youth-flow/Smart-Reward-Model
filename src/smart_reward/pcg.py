"""Robust preconditioned conjugate gradients for Fisher systems."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch


class PCGBreakdownError(RuntimeError):
    """Raised when PCG observes non-finite or non-positive curvature."""


@dataclass(frozen=True)
class PCGResult:
    """Result and explicit convergence diagnostics from :func:`pcg`."""

    solution: torch.Tensor
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    reason: Literal["converged", "zero_rhs", "max_iterations"]

    @property
    def x(self) -> torch.Tensor:
        """Alias for ``solution`` for concise numerical code."""

        return self.solution


def _validate_vector(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must have the same dtype and device as rhs")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@torch.no_grad()
def pcg(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    inverse_diagonal: torch.Tensor | None = None,
    x0: torch.Tensor | None = None,
    *,
    max_iterations: int = 100,
    tolerance: float = 1.0e-5,
    absolute_tolerance: float = 0.0,
    residual_recompute_interval: int = 20,
) -> PCGResult:
    """Solve an SPD system with Jacobi-preconditioned conjugate gradients.

    Args:
        matvec: Callable applying the symmetric positive-definite matrix.
        rhs: One-dimensional right-hand side.
        inverse_diagonal: Optional positive diagonal of ``M^{-1}``.  Omitting
            it gives ordinary conjugate gradients.
        x0: Optional warm start.
        max_iterations: Maximum number of Krylov iterations.
        tolerance: Relative residual tolerance.
        absolute_tolerance: Absolute residual tolerance.
        residual_recompute_interval: Frequency for recomputing ``rhs-Ax`` to
            limit recursive residual drift.

    Returns:
        :class:`PCGResult`.  Hitting ``max_iterations`` is reported rather
        than raised, while invalid inputs and observed non-SPD curvature raise.

    Notes:
        For an exactly zero right-hand side, the unique solution of an SPD
        system is returned immediately as zero.  This deliberately ignores a
        nonzero warm start and avoids an undefined relative residual ``0/0``.
    """

    if not callable(matvec):
        raise TypeError("matvec must be callable")
    if not isinstance(rhs, torch.Tensor):
        raise TypeError("rhs must be a torch.Tensor")
    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")
    if rhs.numel() < 1:
        raise ValueError("rhs must be non-empty")
    if not rhs.is_floating_point():
        raise TypeError("rhs must have a floating-point dtype")
    if not bool(torch.isfinite(rhs).all()):
        raise ValueError("rhs must be finite")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 0
    ):
        raise ValueError("max_iterations must be a non-negative integer")
    if (
        isinstance(residual_recompute_interval, bool)
        or not isinstance(residual_recompute_interval, int)
        or residual_recompute_interval < 1
    ):
        raise ValueError("residual_recompute_interval must be a positive integer")
    tolerance_value = float(tolerance)
    absolute_tolerance_value = float(absolute_tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if not math.isfinite(absolute_tolerance_value) or absolute_tolerance_value < 0.0:
        raise ValueError("absolute_tolerance must be finite and non-negative")

    if x0 is not None:
        _validate_vector("x0", x0, rhs)
    if inverse_diagonal is not None:
        _validate_vector("inverse_diagonal", inverse_diagonal, rhs)
        if bool((inverse_diagonal <= 0.0).any()):
            raise ValueError("inverse_diagonal must be strictly positive")

    rhs_norm_tensor = torch.linalg.vector_norm(rhs)
    rhs_norm = float(rhs_norm_tensor.item())
    if rhs_norm == 0.0:
        return PCGResult(
            solution=torch.zeros_like(rhs),
            converged=True,
            iterations=0,
            residual_norm=0.0,
            relative_residual=0.0,
            reason="zero_rhs",
        )

    def checked_matvec(vector: torch.Tensor) -> torch.Tensor:
        product = matvec(vector)
        _validate_vector("matvec output", product, rhs)
        return product

    solution = torch.zeros_like(rhs) if x0 is None else x0.clone()
    residual = rhs - checked_matvec(solution)
    residual_norm = float(torch.linalg.vector_norm(residual).item())
    threshold = max(absolute_tolerance_value, tolerance_value * rhs_norm)
    if residual_norm <= threshold:
        return PCGResult(
            solution=solution,
            converged=True,
            iterations=0,
            residual_norm=residual_norm,
            relative_residual=residual_norm / rhs_norm,
            reason="converged",
        )

    preconditioned = residual if inverse_diagonal is None else inverse_diagonal * residual
    residual_preconditioned = torch.dot(residual, preconditioned)
    rz_value = float(residual_preconditioned.item())
    if not math.isfinite(rz_value) or rz_value <= 0.0:
        raise PCGBreakdownError(
            "non-positive preconditioned residual norm; the preconditioner is not SPD"
        )
    direction = preconditioned.clone()

    for iteration in range(1, max_iterations + 1):
        matrix_direction = checked_matvec(direction)
        curvature = torch.dot(direction, matrix_direction)
        curvature_value = float(curvature.item())
        if not math.isfinite(curvature_value) or curvature_value <= 0.0:
            raise PCGBreakdownError(
                "PCG observed non-positive curvature p^T A p; the operator is not SPD"
            )

        alpha = residual_preconditioned / curvature
        if not bool(torch.isfinite(alpha)):
            raise PCGBreakdownError("PCG produced a non-finite step size")
        solution.add_(direction, alpha=float(alpha.item()))
        residual.add_(matrix_direction, alpha=-float(alpha.item()))

        recursive_norm = float(torch.linalg.vector_norm(residual).item())
        should_recompute = (
            iteration % residual_recompute_interval == 0 or recursive_norm <= threshold
        )
        if should_recompute:
            residual = rhs - checked_matvec(solution)
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        if not math.isfinite(residual_norm):
            raise PCGBreakdownError("PCG residual became non-finite")
        if residual_norm <= threshold:
            return PCGResult(
                solution=solution,
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                relative_residual=residual_norm / rhs_norm,
                reason="converged",
            )

        preconditioned = residual if inverse_diagonal is None else inverse_diagonal * residual
        next_residual_preconditioned = torch.dot(residual, preconditioned)
        next_rz_value = float(next_residual_preconditioned.item())
        if not math.isfinite(next_rz_value) or next_rz_value <= 0.0:
            raise PCGBreakdownError(
                "non-positive preconditioned residual norm during iteration"
            )
        beta = next_residual_preconditioned / residual_preconditioned
        if not bool(torch.isfinite(beta)):
            raise PCGBreakdownError("PCG produced a non-finite conjugacy coefficient")
        direction.mul_(float(beta.item())).add_(preconditioned)
        residual_preconditioned = next_residual_preconditioned

    # Always report the true residual, not only the recursive estimate.
    residual = rhs - checked_matvec(solution)
    residual_norm = float(torch.linalg.vector_norm(residual).item())
    return PCGResult(
        solution=solution,
        converged=residual_norm <= threshold,
        iterations=max_iterations,
        residual_norm=residual_norm,
        relative_residual=residual_norm / rhs_norm,
        reason="converged" if residual_norm <= threshold else "max_iterations",
    )


__all__ = ["PCGBreakdownError", "PCGResult", "pcg"]
