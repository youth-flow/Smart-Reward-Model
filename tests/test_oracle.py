import math

import pytest
import torch

from smart_reward.oracle import (
    RobustOracleTransform,
    btl_probabilities,
    fit_robust_oracle_transform,
    pair_margins,
)


def test_fit_uses_train_scores_and_transform_is_frozen() -> None:
    train_scores = torch.tensor([-4.0, -1.0, 2.0, 3.0, 20.0], dtype=torch.float64)
    held_out_scores = torch.tensor([-100.0, 100.0], dtype=torch.float64)

    transform = fit_robust_oracle_transform(train_scores)

    assert transform.b == pytest.approx(2.0)
    assert transform.tau == pytest.approx(1.4826 * 3.0)
    expected = (math.log(3.0) / 2.0) * torch.tanh((held_out_scores - transform.b) / transform.tau)
    torch.testing.assert_close(transform.transform(held_out_scores), expected)
    assert transform.b == pytest.approx(2.0)
    assert transform.tau == pytest.approx(1.4826 * 3.0)


def test_transform_range_dtype_device_and_affine_equivariance() -> None:
    train_scores = torch.tensor([-7.0, -2.0, 0.5, 4.0, 11.0], dtype=torch.float64)
    scores = torch.tensor([-1.0e6, -3.0, 0.5, 8.0, 1.0e6], dtype=torch.float64)
    transform = fit_robust_oracle_transform(train_scores)
    transformed = transform(scores)

    bound = math.log(3.0) / 2.0
    assert transformed.dtype == scores.dtype
    assert transformed.device == scores.device
    assert bool((transformed.abs() <= bound).all())

    scale = 3.25
    shift = -19.0
    affine_transform = fit_robust_oracle_transform(scale * train_scores + shift)
    affine_result = affine_transform.transform(scale * scores + shift)
    torch.testing.assert_close(affine_result, transformed, rtol=1.0e-13, atol=1.0e-13)


def test_pair_margin_orientation_reverses_on_swap() -> None:
    left = torch.tensor([0.4, -0.2, 0.0])
    right = torch.tensor([-0.1, 0.3, 0.0])

    margins = pair_margins(left, right)
    torch.testing.assert_close(margins, left - right)
    torch.testing.assert_close(pair_margins(right, left), -margins)


def test_btl_probabilities_obey_controlled_bounds() -> None:
    raw_scores = torch.tensor([-1.0e30, -2.0, 0.0, 3.0, 1.0e30], dtype=torch.float64)
    transform = RobustOracleTransform(b=0.0, tau=1.0)
    rewards = transform(raw_scores)
    margins = pair_margins(rewards.flip(0), rewards)

    probabilities = btl_probabilities(margins)
    assert bool((probabilities >= 0.25).all())
    assert bool((probabilities <= 0.75).all())
    torch.testing.assert_close(
        probabilities + probabilities.flip(0),
        torch.ones_like(probabilities),
    )

    with pytest.raises(AssertionError, match=r"\[0\.25, 0\.75\]"):
        btl_probabilities(torch.tensor([math.log(3.0) + 0.1]))


def test_constant_training_scores_use_scale_floor() -> None:
    train_scores = torch.full((7,), 4.25, dtype=torch.float32)
    transform = fit_robust_oracle_transform(train_scores)

    assert transform.b == pytest.approx(4.25)
    assert transform.tau == pytest.approx(1.0e-6)
    torch.testing.assert_close(transform(train_scores), torch.zeros_like(train_scores))


@pytest.mark.parametrize(
    "scores",
    [
        torch.tensor([], dtype=torch.float32),
        torch.tensor([0.0, float("nan")]),
        torch.tensor([0.0, float("inf")]),
    ],
)
def test_fit_rejects_empty_or_nonfinite_training_scores(scores: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        fit_robust_oracle_transform(scores)
