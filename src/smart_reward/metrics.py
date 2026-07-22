"""Policy-relevant local-regret and natural-direction metrics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .linear import fisher_diagonal as score_fisher_diagonal
from .linear import fisher_matvec as score_fisher_matvec
from .pcg import pcg


def gauge_center(rewards: torch.Tensor, *, candidate_dim: int = -1) -> torch.Tensor:
    """Remove the unidentifiable per-prompt additive reward gauge.

    ``candidate_dim`` identifies the responses belonging to the same prompt.
    For the standard layout ``(num_prompts, num_candidates)``, use the default.
    """

    if not isinstance(rewards, torch.Tensor):
        raise TypeError("rewards must be a torch.Tensor")
    if not rewards.is_floating_point():
        raise TypeError("rewards must have a floating-point dtype")
    if rewards.ndim < 1 or rewards.numel() < 1:
        raise ValueError("rewards must be non-empty")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must be finite")
    normalized_dim = candidate_dim if candidate_dim >= 0 else rewards.ndim + candidate_dim
    if normalized_dim < 0 or normalized_dim >= rewards.ndim:
        raise ValueError("candidate_dim is out of range")
    return rewards - rewards.mean(dim=normalized_dim, keepdim=True)


def _validate_metric_inputs(
    score_matrix: torch.Tensor,
    rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(score_matrix, torch.Tensor) or not isinstance(rewards, torch.Tensor):
        raise TypeError("score_matrix and rewards must be torch.Tensor objects")
    if rewards.ndim < 1 or rewards.numel() < 1:
        raise ValueError("rewards must be non-empty")
    if score_matrix.ndim != rewards.ndim + 1:
        raise ValueError("score_matrix must have shape rewards.shape + (dimension,)")
    if score_matrix.shape[:-1] != rewards.shape or score_matrix.shape[-1] < 1:
        raise ValueError("score_matrix must have shape rewards.shape + (dimension,)")
    if not score_matrix.is_floating_point() or not rewards.is_floating_point():
        raise TypeError("score_matrix and rewards must have floating-point dtypes")
    if score_matrix.dtype != rewards.dtype or score_matrix.device != rewards.device:
        raise ValueError("score_matrix and rewards must share dtype and device")
    if not bool(torch.isfinite(score_matrix).all()) or not bool(torch.isfinite(rewards).all()):
        raise ValueError("score_matrix and rewards must be finite")
    return score_matrix.reshape(-1, score_matrix.shape[-1]), rewards.reshape(-1)


def _prepare_covariance_moment_samples(
    score_matrix: torch.Tensor,
    rewards: torch.Tensor,
    *,
    center_candidates: bool,
    candidate_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened samples normalized for the reward covariance moment.

    With prompt/candidate data, both score and reward are centered within each
    prompt and multiplied by ``sqrt(M / (M - 1))``.  Taking an ordinary mean
    over the resulting ``P*M`` rows is therefore exactly the average unbiased
    sample cross-covariance, whose denominator is ``P*(M-1)``.  This scaling
    applies to ``A_hat r`` only; the default Fisher remains the node estimator
    ``sum(s s.T)/(P*M)`` used during training.
    """

    flat_scores, flat_rewards = _validate_metric_inputs(score_matrix, rewards)
    if not center_candidates:
        return flat_scores, flat_rewards
    if rewards.ndim < 2:
        raise ValueError(
            "center_candidates=True requires a prompt/candidate reward layout; "
            "set it to False for flat population samples"
        )
    normalized_dim = candidate_dim if candidate_dim >= 0 else rewards.ndim + candidate_dim
    if normalized_dim < 0 or normalized_dim >= rewards.ndim:
        raise ValueError("candidate_dim is out of range")
    num_candidates = rewards.shape[normalized_dim]
    if num_candidates < 2:
        raise ValueError("per-prompt covariance requires at least two candidates")

    centered_scores = score_matrix - score_matrix.mean(
        dim=normalized_dim,
        keepdim=True,
    )
    centered_rewards = rewards - rewards.mean(dim=normalized_dim, keepdim=True)
    bessel_scale = math.sqrt(num_candidates / (num_candidates - 1.0))
    return (
        (centered_scores * bessel_scale).reshape(-1, score_matrix.shape[-1]),
        (centered_rewards * bessel_scale).reshape(-1),
    )


def empirical_fisher_matrix(score_matrix: torch.Tensor) -> torch.Tensor:
    """Form ``S.T S / n`` for metric evaluation on controlled test pools."""

    if not isinstance(score_matrix, torch.Tensor):
        raise TypeError("score_matrix must be a torch.Tensor")
    if score_matrix.ndim < 2 or score_matrix.shape[-1] < 1:
        raise ValueError("score_matrix must have at least one sample and one feature dimension")
    if score_matrix.numel() == 0:
        raise ValueError("score_matrix must be non-empty")
    if not score_matrix.is_floating_point():
        raise TypeError("score_matrix must have a floating-point dtype")
    if not bool(torch.isfinite(score_matrix).all()):
        raise ValueError("score_matrix must be finite")
    flat_scores = score_matrix.reshape(-1, score_matrix.shape[-1])
    return flat_scores.mT @ flat_scores / flat_scores.shape[0]


def policy_reward_moment(
    score_matrix: torch.Tensor,
    rewards: torch.Tensor,
    *,
    center_candidates: bool = True,
    candidate_dim: int = -1,
) -> torch.Tensor:
    """Estimate ``A r`` from node scores and rewards.

    By default this is the average per-prompt sample cross-covariance

    ``sum((s_ij-sbar_i) * (r_ij-rbar_i)) / (P * (M-1))``.

    This is gauge invariant and uses the finite-sample covariance correction.
    With ``center_candidates=False``, it instead returns the raw node moment
    ``S.T r / n``.
    """

    flat_scores, flat_rewards = _prepare_covariance_moment_samples(
        score_matrix,
        rewards,
        center_candidates=center_candidates,
        candidate_dim=candidate_dim,
    )
    return flat_scores.mT @ flat_rewards / flat_scores.shape[0]


def _validate_damping_beta(damping: float, beta: float = 1.0) -> tuple[float, float]:
    damping_value = float(damping)
    beta_value = float(beta)
    if not math.isfinite(damping_value) or damping_value < 0.0:
        raise ValueError("damping must be finite and non-negative")
    if not math.isfinite(beta_value) or beta_value <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    return damping_value, beta_value


def _resolve_fisher_geometry(
    flat_scores: torch.Tensor,
    *,
    fisher_matrix: torch.Tensor | None,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor] | None,
    fisher_diagonal: torch.Tensor | None,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor | None]:
    """Resolve the default node Fisher or a caller-supplied test geometry."""

    dimension = flat_scores.shape[1]
    if fisher_matrix is not None and fisher_operator is not None:
        raise ValueError("provide at most one of fisher_matrix and fisher_operator")
    if fisher_matrix is not None:
        if not isinstance(fisher_matrix, torch.Tensor):
            raise TypeError("fisher_matrix must be a torch.Tensor")
        if fisher_matrix.shape != (dimension, dimension):
            raise ValueError(f"fisher_matrix must have shape ({dimension}, {dimension})")
        if fisher_matrix.dtype != flat_scores.dtype or fisher_matrix.device != flat_scores.device:
            raise ValueError("fisher_matrix must share dtype and device with score_matrix")
        if not bool(torch.isfinite(fisher_matrix).all()):
            raise ValueError("fisher_matrix must be finite")
        if not torch.allclose(fisher_matrix, fisher_matrix.mT):
            raise ValueError("fisher_matrix must be symmetric")
        if fisher_diagonal is not None:
            raise ValueError("fisher_diagonal is redundant when fisher_matrix is provided")
        return lambda vector: fisher_matrix @ vector, torch.diagonal(fisher_matrix)

    if fisher_operator is None:
        return (
            lambda vector: score_fisher_matvec(vector, flat_scores, damping=0.0),
            score_fisher_diagonal(flat_scores, damping=0.0),
        )
    if not callable(fisher_operator):
        raise TypeError("fisher_operator must be callable")

    def checked_operator(vector: torch.Tensor) -> torch.Tensor:
        product = fisher_operator(vector)
        if not isinstance(product, torch.Tensor):
            raise TypeError("fisher_operator must return a torch.Tensor")
        if product.shape != (dimension,):
            raise ValueError(f"fisher_operator output must have shape ({dimension},)")
        if product.dtype != flat_scores.dtype or product.device != flat_scores.device:
            raise ValueError("fisher_operator output must share dtype and device with score_matrix")
        if not bool(torch.isfinite(product).all()):
            raise ValueError("fisher_operator output must be finite")
        return product

    if fisher_diagonal is not None:
        if not isinstance(fisher_diagonal, torch.Tensor):
            raise TypeError("fisher_diagonal must be a torch.Tensor")
        if fisher_diagonal.shape != (dimension,):
            raise ValueError(f"fisher_diagonal must have shape ({dimension},)")
        if (
            fisher_diagonal.dtype != flat_scores.dtype
            or fisher_diagonal.device != flat_scores.device
        ):
            raise ValueError("fisher_diagonal must share dtype and device with score_matrix")
        if not bool(torch.isfinite(fisher_diagonal).all()):
            raise ValueError("fisher_diagonal must be finite")
        if bool((fisher_diagonal < 0.0).any()):
            raise ValueError("fisher_diagonal must be non-negative")
    return checked_operator, fisher_diagonal


def _matrix_free_damped_solve(
    fisher_operator: Callable[[torch.Tensor], torch.Tensor],
    fisher_diagonal: torch.Tensor | None,
    moment: torch.Tensor,
    damping: float,
    *,
    pcg_tolerance: float,
    pcg_max_iterations: int,
) -> torch.Tensor:
    def damped_operator(vector: torch.Tensor) -> torch.Tensor:
        return fisher_operator(vector) + damping * vector

    damped_diagonal = None if fisher_diagonal is None else fisher_diagonal + damping
    inverse_diagonal = (
        damped_diagonal.reciprocal()
        if damped_diagonal is not None and bool((damped_diagonal > 0.0).all())
        else None
    )
    result = pcg(
        damped_operator,
        moment,
        inverse_diagonal=inverse_diagonal,
        tolerance=pcg_tolerance,
        max_iterations=pcg_max_iterations,
    )
    if not result.converged:
        raise RuntimeError(
            "natural-direction PCG did not converge: "
            f"relative residual={result.relative_residual:.3e} after "
            f"{result.iterations} iterations"
        )
    return result.solution


def local_regret(
    score_matrix: torch.Tensor,
    predicted_rewards: torch.Tensor,
    target_rewards: torch.Tensor,
    *,
    damping: float,
    beta: float = 1.0,
    center_candidates: bool = True,
    candidate_dim: int = -1,
    pcg_tolerance: float = 1.0e-5,
    pcg_max_iterations: int = 100,
    fisher_matrix: torch.Tensor | None = None,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    fisher_diagonal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute held-out damped local regret.

    The returned value is

    ``m_error.T @ (F + damping I)^-1 @ m_error / (2 beta)``,

    By default ``F=sum(s s.T)/(P*M)`` is the same node Fisher estimator used
    in training, while ``m_error`` is the per-prompt sample cross-covariance
    with denominator ``P*(M-1)``.  ``fisher_matrix`` or ``fisher_operator``
    can replace the default test Fisher without changing the reward moment.
    """

    damping_value, beta_value = _validate_damping_beta(damping, beta)
    flat_scores, _ = _validate_metric_inputs(score_matrix, predicted_rewards)
    _validate_metric_inputs(score_matrix, target_rewards)
    if predicted_rewards.shape != target_rewards.shape:
        raise ValueError("predicted_rewards and target_rewards must have identical shapes")
    if (
        predicted_rewards.dtype != target_rewards.dtype
        or predicted_rewards.device != target_rewards.device
    ):
        raise ValueError("predicted_rewards and target_rewards must share dtype and device")

    error = predicted_rewards - target_rewards
    moment = policy_reward_moment(
        score_matrix,
        error,
        center_candidates=center_candidates,
        candidate_dim=candidate_dim,
    )
    resolved_operator, resolved_diagonal = _resolve_fisher_geometry(
        flat_scores,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    direction = _matrix_free_damped_solve(
        resolved_operator,
        resolved_diagonal,
        moment,
        damping_value,
        pcg_tolerance=pcg_tolerance,
        pcg_max_iterations=pcg_max_iterations,
    )
    value = 0.5 * torch.dot(moment, direction) / beta_value
    # A negative result can only be numerical noise or an invalid solve.
    tolerance = 32.0 * torch.finfo(value.dtype).eps * max(1.0, abs(float(value.item())))
    if float(value.item()) < -tolerance:
        raise FloatingPointError(
            "local regret is negative; the Fisher solve is numerically invalid"
        )
    return value.clamp_min(0.0)


def natural_direction(
    score_matrix: torch.Tensor,
    rewards: torch.Tensor,
    *,
    damping: float,
    center_candidates: bool = True,
    candidate_dim: int = -1,
    pcg_tolerance: float = 1.0e-5,
    pcg_max_iterations: int = 100,
    fisher_matrix: torch.Tensor | None = None,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    fisher_diagonal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``(F + damping I)^-1 A_hat r``."""

    damping_value, _ = _validate_damping_beta(damping)
    flat_scores, _ = _validate_metric_inputs(score_matrix, rewards)
    moment = policy_reward_moment(
        score_matrix,
        rewards,
        center_candidates=center_candidates,
        candidate_dim=candidate_dim,
    )
    resolved_operator, resolved_diagonal = _resolve_fisher_geometry(
        flat_scores,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    return _matrix_free_damped_solve(
        resolved_operator,
        resolved_diagonal,
        moment,
        damping_value,
        pcg_tolerance=pcg_tolerance,
        pcg_max_iterations=pcg_max_iterations,
    )


@dataclass(frozen=True)
class NaturalDirectionMetrics:
    """Comparison of predicted and target local policy update directions."""

    predicted_direction: torch.Tensor
    target_direction: torch.Tensor
    squared_fisher_error: torch.Tensor
    fisher_cosine: torch.Tensor
    predicted_fisher_norm: torch.Tensor
    target_fisher_norm: torch.Tensor


def natural_direction_metrics(
    score_matrix: torch.Tensor,
    predicted_rewards: torch.Tensor,
    target_rewards: torch.Tensor,
    *,
    damping: float,
    center_candidates: bool = True,
    candidate_dim: int = -1,
    pcg_tolerance: float = 1.0e-5,
    pcg_max_iterations: int = 100,
    fisher_matrix: torch.Tensor | None = None,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    fisher_diagonal: torch.Tensor | None = None,
) -> NaturalDirectionMetrics:
    """Report Fisher error and Fisher cosine between natural directions.

    Directions use the damped inverse, while norms and the error metric use
    the undamped empirical Fisher, matching the local policy geometry.
    ``fisher_cosine`` is ``NaN`` when either direction has zero Fisher norm.
    """

    _validate_metric_inputs(score_matrix, predicted_rewards)
    _validate_metric_inputs(score_matrix, target_rewards)
    if predicted_rewards.shape != target_rewards.shape:
        raise ValueError("predicted_rewards and target_rewards must have identical shapes")
    if (
        predicted_rewards.dtype != target_rewards.dtype
        or predicted_rewards.device != target_rewards.device
    ):
        raise ValueError("predicted_rewards and target_rewards must share dtype and device")

    predicted_direction = natural_direction(
        score_matrix,
        predicted_rewards,
        damping=damping,
        center_candidates=center_candidates,
        candidate_dim=candidate_dim,
        pcg_tolerance=pcg_tolerance,
        pcg_max_iterations=pcg_max_iterations,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    target_direction = natural_direction(
        score_matrix,
        target_rewards,
        damping=damping,
        center_candidates=center_candidates,
        candidate_dim=candidate_dim,
        pcg_tolerance=pcg_tolerance,
        pcg_max_iterations=pcg_max_iterations,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    flat_scores, _ = _validate_metric_inputs(score_matrix, predicted_rewards)
    resolved_operator, _ = _resolve_fisher_geometry(
        flat_scores,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    difference = predicted_direction - target_direction
    fisher_difference = resolved_operator(difference)
    squared_error = torch.dot(difference, fisher_difference).clamp_min(0.0)
    fisher_predicted = resolved_operator(predicted_direction)
    fisher_target = resolved_operator(target_direction)
    predicted_squared_norm = torch.dot(predicted_direction, fisher_predicted).clamp_min(0.0)
    target_squared_norm = torch.dot(target_direction, fisher_target).clamp_min(0.0)
    predicted_norm = torch.sqrt(predicted_squared_norm)
    target_norm = torch.sqrt(target_squared_norm)
    denominator = predicted_norm * target_norm
    if float(denominator.item()) == 0.0:
        cosine = torch.full((), float("nan"), dtype=flat_scores.dtype, device=flat_scores.device)
    else:
        cosine = (torch.dot(predicted_direction, fisher_target) / denominator).clamp(
            min=-1.0, max=1.0
        )
    return NaturalDirectionMetrics(
        predicted_direction=predicted_direction,
        target_direction=target_direction,
        squared_fisher_error=squared_error,
        fisher_cosine=cosine,
        predicted_fisher_norm=predicted_norm,
        target_fisher_norm=target_norm,
    )


__all__ = [
    "NaturalDirectionMetrics",
    "empirical_fisher_matrix",
    "gauge_center",
    "local_regret",
    "natural_direction",
    "natural_direction_metrics",
    "policy_reward_moment",
]
