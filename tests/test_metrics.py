import torch

import smart_reward.metrics as metrics_module
from smart_reward.metrics import (
    gauge_center,
    local_regret,
    natural_direction,
    natural_direction_metrics,
    policy_reward_moment,
)


def test_default_score_fisher_solve_is_unpreconditioned(monkeypatch) -> None:
    original_pcg = metrics_module.pcg
    observed_preconditioners: list[torch.Tensor | None] = []

    def recording_pcg(*args, **kwargs):
        observed_preconditioners.append(kwargs.get("inverse_diagonal"))
        return original_pcg(*args, **kwargs)

    monkeypatch.setattr(metrics_module, "pcg", recording_pcg)
    scores = torch.tensor([[[1.0, 0.0], [-1.0, 1.0], [0.5, -0.5]]], dtype=torch.float64)
    rewards = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)
    natural_direction(scores, rewards, damping=0.2)

    assert observed_preconditioners == [None]


def test_gauge_centering_removes_per_prompt_constants() -> None:
    rewards = torch.tensor([[1.0, -1.0, 2.0], [3.0, 4.0, -2.0]], dtype=torch.float64)
    offsets = torch.tensor([[8.0], [-13.0]], dtype=torch.float64)

    assert torch.allclose(gauge_center(rewards + offsets), gauge_center(rewards))
    assert torch.allclose(gauge_center(rewards).mean(dim=-1), torch.zeros(2, dtype=torch.float64))


def test_local_metrics_are_gauge_invariant() -> None:
    generator = torch.Generator().manual_seed(77)
    scores = torch.randn(5, 4, 3, generator=generator, dtype=torch.float64)
    # Exact candidate score centering is the finite-policy score identity.
    scores = scores - scores.mean(dim=1, keepdim=True)
    predicted = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    target = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    predicted_shift = torch.randn(5, 1, generator=generator, dtype=torch.float64)
    target_shift = torch.randn(5, 1, generator=generator, dtype=torch.float64)

    base_regret = local_regret(scores, predicted, target, damping=0.2)
    shifted_regret = local_regret(
        scores,
        predicted + predicted_shift,
        target + target_shift,
        damping=0.2,
    )
    assert torch.allclose(base_regret, shifted_regret, rtol=1.0e-12, atol=1.0e-13)

    base_metrics = natural_direction_metrics(scores, predicted, target, damping=0.2)
    shifted_metrics = natural_direction_metrics(
        scores,
        predicted + predicted_shift,
        target + target_shift,
        damping=0.2,
    )
    assert torch.allclose(
        base_metrics.predicted_direction,
        shifted_metrics.predicted_direction,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    assert torch.allclose(
        base_metrics.target_direction,
        shifted_metrics.target_direction,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    assert torch.allclose(
        base_metrics.squared_fisher_error,
        shifted_metrics.squared_fisher_error,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    assert torch.allclose(
        base_metrics.fisher_cosine,
        shifted_metrics.fisher_cosine,
        rtol=1.0e-12,
        atol=1.0e-13,
    )


def test_moment_uses_m_minus_one_covariance_but_fisher_uses_node_mean() -> None:
    scores = torch.tensor(
        [
            [[1.0, 0.0], [2.0, -1.0], [-1.0, 2.0]],
            [[0.0, 2.0], [3.0, 1.0], [1.0, -2.0]],
        ],
        dtype=torch.float64,
    )
    rewards = torch.tensor([[2.0, -1.0, 4.0], [0.5, 3.0, -2.0]], dtype=torch.float64)
    num_prompts, num_candidates = rewards.shape
    centered_scores = scores - scores.mean(dim=1, keepdim=True)
    centered_rewards = rewards - rewards.mean(dim=1, keepdim=True)
    expected_moment = (centered_scores.reshape(-1, 2).mT @ centered_rewards.reshape(-1)) / (
        num_prompts * (num_candidates - 1)
    )

    actual_moment = policy_reward_moment(scores, rewards)
    assert torch.allclose(actual_moment, expected_moment, rtol=1.0e-13, atol=1.0e-13)

    damping = 0.3
    flat_scores = scores.reshape(-1, 2)
    node_fisher = flat_scores.mT @ flat_scores / flat_scores.shape[0]
    expected_direction = torch.linalg.solve(
        node_fisher + damping * torch.eye(2, dtype=torch.float64),
        expected_moment,
    )
    actual_direction = natural_direction(scores, rewards, damping=damping)
    assert torch.allclose(actual_direction, expected_direction, rtol=1.0e-12, atol=1.0e-13)


def test_metrics_accept_external_fisher_matrix() -> None:
    scores = torch.tensor([[[1.0, 0.0], [-1.0, 2.0], [0.5, -0.5]]], dtype=torch.float64)
    rewards = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    external_fisher = torch.tensor([[2.0, 0.3], [0.3, 1.4]], dtype=torch.float64)
    damping = 0.2
    moment = policy_reward_moment(scores, rewards)
    expected = torch.linalg.solve(
        external_fisher + damping * torch.eye(2, dtype=torch.float64),
        moment,
    )

    actual = natural_direction(
        scores,
        rewards,
        damping=damping,
        fisher_matrix=external_fisher,
    )
    assert torch.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-13)


def test_perfect_rewards_have_zero_regret_and_direction_error() -> None:
    generator = torch.Generator().manual_seed(88)
    scores = torch.randn(3, 3, 2, generator=generator, dtype=torch.float64)
    rewards = torch.randn(3, 3, generator=generator, dtype=torch.float64)

    regret = local_regret(scores, rewards, rewards, damping=0.1)
    metrics = natural_direction_metrics(scores, rewards, rewards, damping=0.1)

    assert regret.item() == 0.0
    assert metrics.squared_fisher_error.item() == 0.0
    assert torch.allclose(metrics.fisher_cosine, torch.ones((), dtype=torch.float64), atol=1.0e-12)
