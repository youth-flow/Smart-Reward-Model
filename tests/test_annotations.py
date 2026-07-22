import math

import pytest
import torch

from smart_reward.annotations import (
    geometric_annotation_counts,
    randomized_truncation_u_statistic_from_counts,
    repeated_labels_to_h,
    sample_geometric_repeated_labels,
)


@pytest.mark.parametrize("probability", [0.25, 0.4, 0.5, 0.6, 0.75])
def test_randomized_u_statistic_is_monte_carlo_unbiased(probability: float) -> None:
    trials = 40_000
    gamma = 0.9
    generator = torch.Generator().manual_seed(7_000 + round(100 * probability))
    totals = geometric_annotation_counts(trials, gamma, generator=generator)
    probabilities = torch.full((trials,), probability, dtype=torch.float64)
    wins = torch.binomial(totals.to(torch.float64), probabilities, generator=generator).to(
        torch.int64
    )

    estimates = randomized_truncation_u_statistic_from_counts(wins, totals, gamma)
    truth = math.log(probability / (1.0 - probability))

    assert abs(float(estimates.mean()) - truth) < 0.02


def test_stable_recurrence_and_fixed_order_special_case() -> None:
    expected_two_wins = 1.0 + 1.0 / (2.0 * 0.9)
    assert repeated_labels_to_h([1, 1], gamma=0.9).item() == pytest.approx(
        expected_two_wins
    )
    assert repeated_labels_to_h([1, 0], gamma=0.9).item() == pytest.approx(0.0)
    assert repeated_labels_to_h([1, 1, 1], gamma=1.0).item() == pytest.approx(
        1.0 + 0.5 + 1.0 / 3.0
    )


def test_geometric_repeated_label_batch_is_consistent() -> None:
    generator = torch.Generator().manual_seed(123)
    probabilities = torch.tensor([[0.0, 1.0], [0.25, 0.75]], dtype=torch.float64)
    batch = sample_geometric_repeated_labels(
        probabilities,
        gamma=0.8,
        generator=generator,
        max_total_annotations=10_000,
    )

    assert batch.counts.shape == probabilities.shape
    assert batch.labels.numel() == int(batch.counts.sum())
    assert batch.wins[0, 0].item() == 0
    assert batch.wins[0, 1].item() == batch.counts[0, 1].item()
    assert torch.isfinite(batch.logit_estimates(gamma=0.8)).all()


@pytest.mark.parametrize(
    ("labels", "gamma"),
    [([], 0.9), ([0, 2], 0.9), ([0, 1], 0.0), ([0, 1], 1.1)],
)
def test_repeated_label_input_validation(labels: list[int], gamma: float) -> None:
    with pytest.raises(ValueError):
        repeated_labels_to_h(labels, gamma=gamma)
