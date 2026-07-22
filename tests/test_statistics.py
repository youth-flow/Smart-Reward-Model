from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from smart_reward.statistics import (
    MetricDirection,
    aggregate_paired_metrics,
    atomic_write_aggregate_json,
    paired_bootstrap_ci,
)


def _five_seed_tables() -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    bt = {
        11: {"test_local_regret": 5.0, "fisher_cosine": 0.2},
        12: {"test_local_regret": 6.0, "fisher_cosine": 0.3},
        13: {"test_local_regret": 7.0, "fisher_cosine": 0.4},
        14: {"test_local_regret": 8.0, "fisher_cosine": 0.5},
        15: {"test_local_regret": 9.0, "fisher_cosine": 0.6},
    }
    prorm_plus = {
        11: {"test_local_regret": 4.0, "fisher_cosine": 0.3},
        12: {"test_local_regret": 5.0, "fisher_cosine": 0.5},
        13: {"test_local_regret": 8.0, "fisher_cosine": 0.3},
        14: {"test_local_regret": 11.0, "fisher_cosine": 0.9},
        15: {"test_local_regret": 10.0, "fisher_cosine": 0.6},
    }
    return bt, prorm_plus


def test_known_paired_arithmetic_and_direction_metadata() -> None:
    bt, prorm_plus = _five_seed_tables()

    aggregate = aggregate_paired_metrics(
        bt,
        prorm_plus,
        bootstrap_seed=20260722,
        num_resamples=2_000,
    )
    summaries = {summary.metric: summary for summary in aggregate.metrics}
    regret = summaries["test_local_regret"]

    # Paired differences are [-1, -1, 1, 3, 1].
    assert [item.prorm_plus_minus_bt for item in regret.per_seed] == [
        -1.0,
        -1.0,
        1.0,
        3.0,
        1.0,
    ]
    assert regret.paired_mean == pytest.approx(0.6)
    assert regret.sample_std == pytest.approx(math.sqrt(2.8))
    assert regret.standard_error == pytest.approx(math.sqrt(2.8 / 5.0))
    assert regret.direction is MetricDirection.LOWER_IS_BETTER
    assert regret.favorable_prorm_plus_minus_bt_sign == "negative"
    assert summaries["fisher_cosine"].direction is MetricDirection.HIGHER_IS_BETTER
    assert summaries["fisher_cosine"].favorable_prorm_plus_minus_bt_sign == "positive"

    payload = aggregate.to_dict()
    assert payload["schema_version"] == "paired-seed-aggregate/v2"
    assert payload["num_seeds"] == 5
    assert payload["metrics"]["test_local_regret"]["paired_mean"] == pytest.approx(0.6)
    first_seed = payload["metrics"]["test_local_regret"]["per_seed"][0]
    assert "prorm_plus" in first_seed and "prorm_plus_minus_bt" in first_seed
    assert "srm" not in first_seed and "srm_minus_bt" not in first_seed
    assert "significant" not in json.dumps(payload)


def test_bootstrap_is_deterministic_and_does_not_touch_global_rng() -> None:
    differences = [-2.0, -1.0, 0.5, 1.0, 4.0]
    torch.manual_seed(12345)
    global_state = torch.random.get_rng_state().clone()

    first = paired_bootstrap_ci(
        differences,
        bootstrap_seed=99,
        num_resamples=5_000,
    )
    second = paired_bootstrap_ci(
        differences,
        bootstrap_seed=99,
        num_resamples=5_000,
    )

    assert first == second
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_explicit_generators_are_reproducible() -> None:
    left = torch.Generator(device="cpu").manual_seed(17)
    right = torch.Generator(device="cpu").manual_seed(17)

    first = paired_bootstrap_ci([1.0, 2.0, 5.0], generator=left, num_resamples=500)
    second = paired_bootstrap_ci([1.0, 2.0, 5.0], generator=right, num_resamples=500)

    assert first == second


def test_constant_difference_has_exact_percentile_interval() -> None:
    interval = paired_bootstrap_ci(
        [-0.75, -0.75, -0.75, -0.75, -0.75],
        bootstrap_seed=3,
        num_resamples=100,
    )

    assert interval.lower == pytest.approx(-0.75)
    assert interval.upper == pytest.approx(-0.75)


@pytest.mark.parametrize(
    ("bt", "prorm_plus", "message"),
    [
        (
            {1: {"local_regret": 1.0}, 2: {"local_regret": 2.0}},
            {1: {"local_regret": 1.0}, 3: {"local_regret": 2.0}},
            "seed sets must match exactly",
        ),
        (
            {1: {"local_regret": 1.0}},
            {1: {"local_regret": 1.0}},
            "at least two",
        ),
        (
            {1: {"local_regret": 1.0}, 2: {"local_regret": 2.0}},
            {1: {"local_regret": 1.0}, 2: {"fisher_cosine": 0.2}},
            "same metric names",
        ),
    ],
)
def test_pairing_and_metric_sets_fail_closed(
    bt: dict[int, dict[str, float]],
    prorm_plus: dict[int, dict[str, float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_paired_metrics(bt, prorm_plus, bootstrap_seed=1, num_resamples=10)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_metrics_are_rejected(invalid: float) -> None:
    bt = {1: {"local_regret": 1.0}, 2: {"local_regret": 2.0}}
    prorm_plus = {1: {"local_regret": 0.5}, 2: {"local_regret": invalid}}

    with pytest.raises(ValueError, match="finite"):
        aggregate_paired_metrics(bt, prorm_plus, bootstrap_seed=1, num_resamples=10)


def test_unknown_metric_requires_explicit_direction() -> None:
    bt = {1: {"custom_score": 1.0}, 2: {"custom_score": 2.0}}
    prorm_plus = {1: {"custom_score": 2.0}, 2: {"custom_score": 4.0}}

    with pytest.raises(ValueError, match="no locked direction"):
        aggregate_paired_metrics(bt, prorm_plus, bootstrap_seed=1, num_resamples=10)
    aggregate = aggregate_paired_metrics(
        bt,
        prorm_plus,
        directions={"custom_score": "higher_is_better"},
        bootstrap_seed=1,
        num_resamples=10,
    )
    assert aggregate.metrics[0].direction is MetricDirection.HIGHER_IS_BETTER


def test_atomic_writer_replaces_json_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    bt, prorm_plus = _five_seed_tables()
    aggregate = aggregate_paired_metrics(
        bt,
        prorm_plus,
        bootstrap_seed=8,
        num_resamples=100,
    )
    destination = tmp_path / "five-seed.json"
    destination.write_text("old\n", encoding="utf-8")

    atomic_write_aggregate_json(destination, aggregate)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == aggregate.to_dict()
    assert list(tmp_path.glob(".five-seed.json.*.tmp")) == []


def test_random_source_must_be_explicit_and_unambiguous() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        paired_bootstrap_ci([1.0, 2.0])
    with pytest.raises(ValueError, match="exactly one"):
        paired_bootstrap_ci(
            [1.0, 2.0],
            bootstrap_seed=1,
            generator=torch.Generator(device="cpu"),
        )
