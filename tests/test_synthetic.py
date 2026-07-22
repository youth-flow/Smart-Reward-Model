import math
from dataclasses import replace

import pytest
import torch

import smart_reward.synthetic as synthetic_module
from smart_reward.annotations import randomized_truncation_u_statistic_from_counts
from smart_reward.synthetic import SyntheticExperimentConfig, run_synthetic_experiment


@pytest.fixture(scope="module")
def result():
    return run_synthetic_experiment(314159)


def test_synthetic_experiment_is_exactly_reproducible(result) -> None:
    repeated = run_synthetic_experiment(result.seed, result.config)

    assert repeated == result
    assert result != run_synthetic_experiment(result.seed + 1, result.config)


def test_result_is_finite_and_contains_valid_real_diagnostics(result) -> None:
    assert result.train_misspecification_rmse > 0.0
    for learner in (result.bt, result.prorm_plus):
        scalars = (
            learner.initial_train_objective,
            learner.final_train_objective,
            learner.test_local_regret,
            learner.direction_fisher_cosine,
            learner.direction_squared_fisher_error,
            learner.predicted_direction_fisher_norm,
            learner.target_direction_fisher_norm,
            *learner.final_weight,
        )
        assert all(math.isfinite(value) for value in scalars)
        assert learner.test_local_regret >= 0.0
        assert learner.direction_squared_fisher_error >= 0.0
        assert -1.0 <= learner.direction_fisher_cosine <= 1.0
        assert learner.predicted_direction_fisher_norm > 0.0
        assert learner.target_direction_fisher_norm > 0.0

    evidence = result.prorm_plus_pcg
    assert evidence.last_train_converged
    assert evidence.evaluation_converged
    assert evidence.last_train_relative_residual <= result.config.pcg_tolerance
    assert evidence.evaluation_relative_residual <= result.config.pcg_tolerance
    assert evidence.last_train_dual_loss == pytest.approx(
        evidence.last_train_dual_saddle_value,
        rel=1.0e-10,
        abs=1.0e-14,
    )
    assert evidence.evaluation_dual_loss == pytest.approx(
        evidence.evaluation_dual_saddle_value,
        rel=1.0e-10,
        abs=1.0e-14,
    )
    assert result.prorm_plus.final_train_objective == evidence.evaluation_dual_loss


def test_repeated_label_evidence_is_the_geometric_u_statistic(result) -> None:
    counts = torch.tensor(result.annotation_counts, dtype=torch.int64)
    wins = torch.tensor(result.left_wins, dtype=torch.int64)
    recorded_h = torch.tensor(result.h_values, dtype=torch.float64)
    reconstructed_h = randomized_truncation_u_statistic_from_counts(
        wins,
        counts,
        gamma=result.config.annotation_gamma,
        dtype=torch.float64,
    )

    assert torch.equal(recorded_h, reconstructed_h)
    assert int(counts.sum()) >= result.config.num_train_prompts
    # This fixed-seed draw is genuinely ragged rather than a disguised fixed-N batch.
    assert torch.unique(counts).numel() > 1
    assert bool((counts > 1).any())
    assert bool(((wins >= 0) & (wins <= counts)).all())


def test_train_and_test_are_disjoint_and_test_size_cannot_change_training(result) -> None:
    assert set(result.train_prompt_ids).isdisjoint(result.test_prompt_ids)
    assert len(result.train_prompt_ids) == result.config.num_train_prompts
    assert len(result.test_prompt_ids) == result.config.num_test_prompts

    changed_test_config = replace(
        result.config,
        num_test_prompts=result.config.num_test_prompts + 3,
    )
    changed_test = run_synthetic_experiment(result.seed, changed_test_config)

    assert changed_test.annotation_counts == result.annotation_counts
    assert changed_test.left_wins == result.left_wins
    assert changed_test.h_values == result.h_values
    assert changed_test.oracle_center == result.oracle_center
    assert changed_test.oracle_scale == result.oracle_scale
    assert changed_test.train_misspecification_rmse == result.train_misspecification_rmse
    assert changed_test.bt.initial_train_objective == result.bt.initial_train_objective
    assert changed_test.bt.final_train_objective == result.bt.final_train_objective
    assert changed_test.bt.final_weight == result.bt.final_weight
    assert (
        changed_test.prorm_plus.initial_train_objective == result.prorm_plus.initial_train_objective
    )
    assert changed_test.prorm_plus.final_train_objective == result.prorm_plus.final_train_objective
    assert changed_test.prorm_plus.final_weight == result.prorm_plus.final_weight


def test_benchmark_uses_cpu_float64_labels_and_held_out_covariance_metrics(
    monkeypatch,
) -> None:
    observed_probabilities: list[torch.Tensor] = []
    observed_metric_shapes: list[tuple[int, ...]] = []
    original_sampler = synthetic_module.sample_geometric_repeated_labels
    original_regret = synthetic_module.local_regret
    original_directions = synthetic_module.natural_direction_metrics

    def checked_sampler(probabilities, *args, **kwargs):
        assert probabilities.dtype == torch.float64
        assert probabilities.device.type == "cpu"
        assert bool(((probabilities >= 0.25) & (probabilities <= 0.75)).all())
        observed_probabilities.append(probabilities.detach().clone())
        return original_sampler(probabilities, *args, **kwargs)

    def checked_regret(scores, predicted, target, *args, **kwargs):
        assert scores.dtype == predicted.dtype == target.dtype == torch.float64
        assert scores.device.type == predicted.device.type == target.device.type == "cpu"
        assert scores.shape[:-1] == predicted.shape == target.shape
        observed_metric_shapes.append(tuple(scores.shape))
        return original_regret(scores, predicted, target, *args, **kwargs)

    def checked_directions(scores, predicted, target, *args, **kwargs):
        assert scores.dtype == predicted.dtype == target.dtype == torch.float64
        assert scores.shape[:-1] == predicted.shape == target.shape
        observed_metric_shapes.append(tuple(scores.shape))
        return original_directions(scores, predicted, target, *args, **kwargs)

    monkeypatch.setattr(
        synthetic_module,
        "sample_geometric_repeated_labels",
        checked_sampler,
    )
    monkeypatch.setattr(synthetic_module, "local_regret", checked_regret)
    monkeypatch.setattr(
        synthetic_module,
        "natural_direction_metrics",
        checked_directions,
    )
    config = SyntheticExperimentConfig(
        num_train_prompts=12,
        num_test_prompts=8,
        bt_steps=5,
        prorm_plus_steps=5,
        microbatch_size=4,
    )
    run_synthetic_experiment(7, config)

    assert len(observed_probabilities) == 1
    assert len(observed_metric_shapes) == 4
    assert all(
        shape == (config.num_test_prompts, config.num_candidates, config.policy_dimension)
        for shape in observed_metric_shapes
    )


def test_benchmark_does_not_mutate_global_torch_rng() -> None:
    torch.manual_seed(8675309)
    expected = torch.rand(5)
    torch.manual_seed(8675309)
    config = SyntheticExperimentConfig(
        num_train_prompts=8,
        num_test_prompts=6,
        bt_steps=2,
        prorm_plus_steps=2,
        microbatch_size=4,
    )
    run_synthetic_experiment(11, config)
    actual = torch.rand(5)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_candidates": 1},
        {"annotation_gamma": 1.0},
        {"training_damping": 0.0},
        {"evaluation_damping": -0.1},
        {"bt_steps": 0},
        {"pcg_tolerance": float("nan")},
        {"num_train_prompts": 1, "num_candidates": 2, "reward_dimension": 1},
    ],
)
def test_config_rejects_invalid_controlled_benchmark_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        SyntheticExperimentConfig(**kwargs)
