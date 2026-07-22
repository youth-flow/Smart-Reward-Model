import pytest
import torch

from smart_reward.linear import DampedEmpiricalFisher, fisher_diagonal, fisher_matvec
from smart_reward.pcg import PCGBreakdownError, pcg


def test_pcg_matches_direct_damped_fisher_solve() -> None:
    generator = torch.Generator().manual_seed(22)
    scores = torch.randn(48, 7, generator=generator, dtype=torch.float64)
    rhs = torch.randn(7, generator=generator, dtype=torch.float64)
    damping = 0.17
    fisher = scores.mT @ scores / scores.shape[0]
    matrix = fisher + damping * torch.eye(7, dtype=torch.float64)
    expected = torch.linalg.solve(matrix, rhs)

    operator = DampedEmpiricalFisher(scores, damping)
    result = pcg(
        operator.matvec,
        rhs,
        inverse_diagonal=operator.inverse_diagonal(),
        max_iterations=50,
        tolerance=1.0e-12,
    )

    assert result.converged
    assert result.reason == "converged"
    assert result.relative_residual < 1.0e-11
    assert torch.allclose(result.solution, expected, rtol=1.0e-10, atol=1.0e-11)
    assert torch.allclose(fisher_matvec(rhs, scores, damping), matrix @ rhs)
    assert torch.allclose(fisher_diagonal(scores, damping), torch.diagonal(matrix))


def test_pcg_handles_zero_rhs_exactly() -> None:
    rhs = torch.zeros(4, dtype=torch.float64)
    result = pcg(lambda vector: 3.0 * vector, rhs, x0=torch.ones_like(rhs))

    assert result.converged
    assert result.reason == "zero_rhs"
    assert result.iterations == 0
    assert result.relative_residual == 0.0
    assert torch.equal(result.solution, rhs)


def test_pcg_rejects_observed_non_spd_curvature() -> None:
    rhs = torch.tensor([1.0, -2.0], dtype=torch.float64)
    with pytest.raises(PCGBreakdownError, match="not SPD"):
        pcg(lambda vector: -vector, rhs)
