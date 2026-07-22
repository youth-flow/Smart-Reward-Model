import math

import pytest
import torch
import torch.nn.functional as functional

from smart_reward.baseline import repeated_btl_nll


def test_repeated_btl_matches_expanded_labels() -> None:
    margins = torch.tensor([1.25, -0.7], dtype=torch.float64)
    wins = torch.tensor([2, 1], dtype=torch.int64)
    totals = torch.tensor([3, 2], dtype=torch.int64)

    expanded_margin = torch.tensor([1.25, 1.25, 1.25, -0.7, -0.7], dtype=torch.float64)
    expanded_label = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    expected = functional.softplus(expanded_margin).sub(
        expanded_label * expanded_margin
    ).mean()

    assert torch.allclose(repeated_btl_nll(margins, wins, totals), expected)


def test_zero_margin_is_log_two() -> None:
    loss = repeated_btl_nll(
        torch.zeros(2),
        torch.tensor([0, 5]),
        torch.tensor([3, 5]),
    )
    assert loss.item() == pytest.approx(math.log(2.0))


def test_extreme_margins_are_finite() -> None:
    margins = torch.tensor([1000.0, -1000.0], requires_grad=True)
    loss = repeated_btl_nll(
        margins,
        torch.tensor([1, 0]),
        torch.tensor([1, 1]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(margins.grad).all()


def test_invalid_counts_fail() -> None:
    with pytest.raises(ValueError):
        repeated_btl_nll(
            torch.zeros(1),
            torch.tensor([2]),
            torch.tensor([1]),
        )
