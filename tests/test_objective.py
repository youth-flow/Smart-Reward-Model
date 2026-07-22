import pytest
import torch

from smart_reward.objective import (
    dual_loss,
    dual_saddle_value,
    empirical_moment,
    envelope_surrogate,
    envelope_weights,
)


def _problem() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(314)
    edge_features = torch.randn(11, 4, generator=generator, dtype=torch.float64)
    margins = torch.randn(11, generator=generator, dtype=torch.float64)
    targets = torch.randn(11, generator=generator, dtype=torch.float64)
    raw = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    operator = raw.mT @ raw / 4.0 + 0.4 * torch.eye(4, dtype=torch.float64)
    return edge_features, margins, targets, operator


def test_dual_value_equals_quadratic_loss_at_optimum() -> None:
    edge_features, margins, targets, operator = _problem()
    moment = empirical_moment(edge_features, margins, targets)
    direction = torch.linalg.solve(operator, moment)

    quadratic = 0.5 * torch.dot(moment, torch.linalg.solve(operator, moment))
    primal_form = dual_loss(moment, direction)
    saddle_form = dual_saddle_value(moment, direction, operator @ direction)

    assert primal_form.item() == pytest.approx(quadratic.item(), rel=1.0e-12)
    assert saddle_form.item() == pytest.approx(quadratic.item(), rel=1.0e-12)


def test_envelope_weight_matches_finite_difference() -> None:
    edge_features, margins, targets, operator = _problem()
    moment = empirical_moment(edge_features, margins, targets)
    direction = torch.linalg.solve(operator, moment)
    weights = envelope_weights(edge_features, direction)

    perturbation = torch.linspace(-0.7, 0.9, margins.numel(), dtype=torch.float64)
    epsilon = 1.0e-6

    def exact_loss(candidate_margins: torch.Tensor) -> torch.Tensor:
        candidate_moment = empirical_moment(edge_features, candidate_margins, targets)
        return 0.5 * torch.dot(candidate_moment, torch.linalg.solve(operator, candidate_moment))

    finite_difference = (
        exact_loss(margins + epsilon * perturbation)
        - exact_loss(margins - epsilon * perturbation)
    ) / (2.0 * epsilon)
    envelope_derivative = torch.mean(weights * perturbation)

    assert finite_difference.item() == pytest.approx(
        envelope_derivative.item(), rel=2.0e-8, abs=2.0e-10
    )

    differentiable_margins = margins.clone().requires_grad_(True)
    surrogate = envelope_surrogate(differentiable_margins, targets, weights)
    (gradient,) = torch.autograd.grad(surrogate, differentiable_margins)
    assert torch.allclose(gradient, weights / margins.numel(), atol=1.0e-14, rtol=0.0)
