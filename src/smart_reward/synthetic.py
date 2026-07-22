"""Fast deterministic end-to-end benchmark for misspecified SRM+ training.

This module is a CPU-only numerical integration benchmark, not evidence that
SRM+ must outperform BT-MLE.  It deliberately runs the real randomized
geometric repeated-label estimator, both production feature trainers, and the
held-out prompt-covariance policy metrics.  Train labels and test nodes are
generated from independent local random streams so changing the test split
cannot alter a training result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .annotations import sample_geometric_repeated_labels
from .metrics import local_regret, natural_direction_metrics
from .oracle import btl_probabilities, fit_robust_oracle_transform, pair_margins
from .training import (
    BTMLETrainer,
    BTMLETrainingConfig,
    FeatureTrainingBatch,
    FrozenFeatureLinearReward,
    SRMPlusTrainer,
    SRMPlusTrainingConfig,
    evaluate_bt_mle,
    evaluate_srm_plus,
)

_MAX_SEED = 2**63 - 1


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


@dataclass(frozen=True)
class SyntheticExperimentConfig:
    """Fixed-size CPU/float64 controlled-benchmark configuration."""

    num_train_prompts: int = 32
    num_test_prompts: int = 20
    num_candidates: int = 4
    reward_dimension: int = 2
    policy_dimension: int = 3
    annotation_gamma: float = 0.8
    bt_steps: int = 24
    srm_steps: int = 24
    bt_learning_rate: float = 0.04
    srm_learning_rate: float = 0.04
    beta: float = 1.0
    training_damping: float = 0.15
    evaluation_damping: float = 0.15
    microbatch_size: int = 8
    pcg_tolerance: float = 1.0e-10
    pcg_max_iterations: int = 30

    def __post_init__(self) -> None:
        for name in (
            "num_train_prompts",
            "num_test_prompts",
            "reward_dimension",
            "policy_dimension",
            "bt_steps",
            "srm_steps",
            "microbatch_size",
            "pcg_max_iterations",
        ):
            _positive_integer(name, getattr(self, name))
        if (
            isinstance(self.num_candidates, bool)
            or not isinstance(self.num_candidates, int)
            or self.num_candidates < 2
        ):
            raise ValueError("num_candidates must be an integer of at least two")
        identifiable_reward_rank = self.num_train_prompts * (self.num_candidates - 1)
        if self.reward_dimension >= identifiable_reward_rank:
            raise ValueError(
                "reward_dimension must be smaller than the centered train-node rank bound"
            )
        gamma = float(self.annotation_gamma)
        if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
            raise ValueError("annotation_gamma must be finite and lie in (0, 1)")
        for name in (
            "bt_learning_rate",
            "srm_learning_rate",
            "beta",
            "training_damping",
            "evaluation_damping",
            "pcg_tolerance",
        ):
            _finite_positive(name, getattr(self, name))


@dataclass(frozen=True)
class SyntheticLearnerResult:
    """Training and held-out policy metrics for one reward learner."""

    initial_train_objective: float
    final_train_objective: float
    test_local_regret: float
    direction_fisher_cosine: float
    direction_squared_fisher_error: float
    predicted_direction_fisher_norm: float
    target_direction_fisher_norm: float
    final_weight: tuple[float, ...]

    def __post_init__(self) -> None:
        scalar_names = (
            "initial_train_objective",
            "final_train_objective",
            "test_local_regret",
            "direction_fisher_cosine",
            "direction_squared_fisher_error",
            "predicted_direction_fisher_norm",
            "target_direction_fisher_norm",
        )
        for name in scalar_names:
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not -1.0 <= self.direction_fisher_cosine <= 1.0:
            raise ValueError("direction_fisher_cosine must lie in [-1, 1]")
        for name in (
            "initial_train_objective",
            "final_train_objective",
            "test_local_regret",
            "direction_squared_fisher_error",
            "predicted_direction_fisher_norm",
            "target_direction_fisher_norm",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not self.final_weight or not all(math.isfinite(value) for value in self.final_weight):
            raise ValueError("final_weight must be non-empty and finite")


@dataclass(frozen=True)
class SyntheticPCGEvidence:
    """Numerical evidence from the last training solve and final evaluation solve."""

    last_train_iterations: int
    last_train_residual_norm: float
    last_train_relative_residual: float
    last_train_converged: bool
    last_train_dual_loss: float
    last_train_dual_saddle_value: float
    evaluation_iterations: int
    evaluation_residual_norm: float
    evaluation_relative_residual: float
    evaluation_converged: bool
    evaluation_dual_loss: float
    evaluation_dual_saddle_value: float

    def __post_init__(self) -> None:
        for name in ("last_train_iterations", "evaluation_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "last_train_residual_norm",
            "last_train_relative_residual",
            "last_train_dual_loss",
            "last_train_dual_saddle_value",
            "evaluation_residual_norm",
            "evaluation_relative_residual",
            "evaluation_dual_loss",
            "evaluation_dual_saddle_value",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in (
            "last_train_residual_norm",
            "last_train_relative_residual",
            "last_train_dual_loss",
            "evaluation_residual_norm",
            "evaluation_relative_residual",
            "evaluation_dual_loss",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.last_train_converged, bool) or not isinstance(
            self.evaluation_converged, bool
        ):
            raise TypeError("PCG convergence fields must be bool")


@dataclass(frozen=True)
class SyntheticExperimentResult:
    """Auditable output of :func:`run_synthetic_experiment`."""

    seed: int
    config: SyntheticExperimentConfig
    train_prompt_ids: tuple[int, ...]
    test_prompt_ids: tuple[int, ...]
    annotation_counts: tuple[int, ...]
    left_wins: tuple[int, ...]
    h_values: tuple[float, ...]
    oracle_center: float
    oracle_scale: float
    train_misspecification_rmse: float
    bt: SyntheticLearnerResult
    srm: SyntheticLearnerResult
    srm_pcg: SyntheticPCGEvidence

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.config, SyntheticExperimentConfig):
            raise TypeError("config must be a SyntheticExperimentConfig")
        if len(self.train_prompt_ids) != self.config.num_train_prompts:
            raise ValueError("train_prompt_ids has the wrong length")
        if len(self.test_prompt_ids) != self.config.num_test_prompts:
            raise ValueError("test_prompt_ids has the wrong length")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (*self.train_prompt_ids, *self.test_prompt_ids)
        ):
            raise TypeError("prompt IDs must be integers")
        if len(set(self.train_prompt_ids)) != len(self.train_prompt_ids) or len(
            set(self.test_prompt_ids)
        ) != len(self.test_prompt_ids):
            raise ValueError("prompt IDs must be unique within each split")
        if set(self.train_prompt_ids).intersection(self.test_prompt_ids):
            raise ValueError("train and test prompt IDs must be disjoint")
        num_edges = self.config.num_train_prompts
        if not (
            len(self.annotation_counts) == len(self.left_wins) == len(self.h_values) == num_edges
        ):
            raise ValueError("annotation evidence must contain one entry per train edge")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (*self.annotation_counts, *self.left_wins)
        ):
            raise TypeError("annotation counts and wins must be integers")
        if any(total < 1 for total in self.annotation_counts):
            raise ValueError("annotation counts must be positive")
        if any(
            win < 0 or win > total
            for win, total in zip(self.left_wins, self.annotation_counts, strict=True)
        ):
            raise ValueError("left-win evidence is inconsistent with annotation counts")
        for name in (
            "oracle_center",
            "oracle_scale",
            "train_misspecification_rmse",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.oracle_scale <= 0.0:
            raise ValueError("oracle_scale must be positive")
        if self.train_misspecification_rmse <= 0.0:
            raise ValueError("the controlled reward class must be genuinely misspecified")
        if not all(math.isfinite(value) for value in self.h_values):
            raise ValueError("h_values must be finite")
        if not isinstance(self.bt, SyntheticLearnerResult) or not isinstance(
            self.srm, SyntheticLearnerResult
        ):
            raise TypeError("bt and srm must be SyntheticLearnerResult objects")
        if not isinstance(self.srm_pcg, SyntheticPCGEvidence):
            raise TypeError("srm_pcg must be SyntheticPCGEvidence")
        if (
            len(self.bt.final_weight) != self.config.reward_dimension
            or len(self.srm.final_weight) != self.config.reward_dimension
        ):
            raise ValueError("learner weight dimensions do not match the configuration")


@dataclass(frozen=True)
class _SyntheticWorld:
    feature_basis: torch.Tensor
    score_projection: torch.Tensor
    omitted_direction: torch.Tensor
    reward_weight: torch.Tensor


@dataclass(frozen=True)
class _SyntheticNodes:
    features: torch.Tensor
    scores: torch.Tensor
    raw_rewards: torch.Tensor


def _generator(seed: int, stream: int) -> torch.Generator:
    derived_seed = (seed + 1_000_003 * stream) % _MAX_SEED
    return torch.Generator(device="cpu").manual_seed(derived_seed)


def _make_world(config: SyntheticExperimentConfig, seed: int) -> _SyntheticWorld:
    generator = _generator(seed, 1)
    latent_dimension = config.reward_dimension + config.policy_dimension + 2
    raw_basis = torch.randn(
        latent_dimension,
        config.reward_dimension,
        generator=generator,
        dtype=torch.float64,
    )
    feature_basis = torch.linalg.qr(raw_basis, mode="reduced").Q
    omitted = torch.randn(latent_dimension, generator=generator, dtype=torch.float64)
    omitted = omitted - feature_basis @ (feature_basis.mT @ omitted)
    omitted = omitted / torch.linalg.vector_norm(omitted)

    extra_scores = torch.randn(
        latent_dimension,
        config.policy_dimension - 1,
        generator=generator,
        dtype=torch.float64,
    )
    score_projection = torch.cat((omitted[:, None], extra_scores), dim=1)
    score_projection = score_projection / torch.linalg.vector_norm(
        score_projection, dim=0, keepdim=True
    )
    reward_weight = torch.randn(
        config.reward_dimension,
        generator=generator,
        dtype=torch.float64,
    )
    reward_weight = reward_weight / torch.linalg.vector_norm(reward_weight)
    return _SyntheticWorld(
        feature_basis=feature_basis,
        score_projection=score_projection,
        omitted_direction=omitted,
        reward_weight=reward_weight,
    )


def _sample_nodes(
    config: SyntheticExperimentConfig,
    world: _SyntheticWorld,
    *,
    num_prompts: int,
    generator: torch.Generator,
) -> _SyntheticNodes:
    latent_dimension = world.feature_basis.shape[0]
    latent = torch.randn(
        num_prompts,
        config.num_candidates,
        latent_dimension,
        generator=generator,
        dtype=torch.float64,
    )
    features = latent @ world.feature_basis
    raw_scores = latent @ world.score_projection
    scores = raw_scores - raw_scores.mean(dim=1, keepdim=True)
    omitted_signal = latent @ world.omitted_direction
    represented_signal = features @ world.reward_weight
    nonlinear_signal = omitted_signal * features[..., 0]
    raw_rewards = 0.65 * represented_signal + omitted_signal + 0.25 * nonlinear_signal
    return _SyntheticNodes(features=features, scores=scores, raw_rewards=raw_rewards)


def _misspecification_rmse(features: torch.Tensor, rewards: torch.Tensor) -> float:
    centered_features = features - features.mean(dim=1, keepdim=True)
    centered_rewards = rewards - rewards.mean(dim=1, keepdim=True)
    flat_features = centered_features.reshape(-1, centered_features.shape[-1])
    flat_rewards = centered_rewards.reshape(-1)
    fitted_weight = torch.linalg.lstsq(flat_features, flat_rewards).solution
    residual = flat_features @ fitted_weight - flat_rewards
    return float(torch.sqrt(torch.mean(residual.square())).item())


def _learner_result(
    model: FrozenFeatureLinearReward,
    test_nodes: _SyntheticNodes,
    target_rewards: torch.Tensor,
    *,
    initial_train_objective: float,
    final_train_objective: float,
    config: SyntheticExperimentConfig,
) -> SyntheticLearnerResult:
    with torch.no_grad():
        predicted_rewards = model(test_nodes.features)
    regret = local_regret(
        test_nodes.scores,
        predicted_rewards,
        target_rewards,
        damping=config.evaluation_damping,
        beta=config.beta,
        pcg_tolerance=config.pcg_tolerance,
        pcg_max_iterations=config.pcg_max_iterations,
    )
    direction = natural_direction_metrics(
        test_nodes.scores,
        predicted_rewards,
        target_rewards,
        damping=config.evaluation_damping,
        pcg_tolerance=config.pcg_tolerance,
        pcg_max_iterations=config.pcg_max_iterations,
    )
    return SyntheticLearnerResult(
        initial_train_objective=initial_train_objective,
        final_train_objective=final_train_objective,
        test_local_regret=float(regret.item()),
        direction_fisher_cosine=float(direction.fisher_cosine.item()),
        direction_squared_fisher_error=float(direction.squared_fisher_error.item()),
        predicted_direction_fisher_norm=float(direction.predicted_fisher_norm.item()),
        target_direction_fisher_norm=float(direction.target_fisher_norm.item()),
        final_weight=tuple(float(value) for value in model.weight.detach().tolist()),
    )


def run_synthetic_experiment(
    seed: int,
    config: SyntheticExperimentConfig | None = None,
) -> SyntheticExperimentResult:
    """Run the real controlled training/evaluation path on CPU in float64.

    The benchmark uses one canonical edge (candidate 0 minus candidate 1) per
    training prompt, while all candidates enter the train Fisher and held-out
    prompt-covariance metrics.  No assertion or data construction forces SRM+
    to win; the returned values are the observed outcome for ``seed``.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed >= _MAX_SEED:
        raise ValueError(f"seed must lie in [0, {_MAX_SEED})")
    effective_config = SyntheticExperimentConfig() if config is None else config
    if not isinstance(effective_config, SyntheticExperimentConfig):
        raise TypeError("config must be a SyntheticExperimentConfig")

    world = _make_world(effective_config, seed)
    train_nodes = _sample_nodes(
        effective_config,
        world,
        num_prompts=effective_config.num_train_prompts,
        generator=_generator(seed, 2),
    )
    test_nodes = _sample_nodes(
        effective_config,
        world,
        num_prompts=effective_config.num_test_prompts,
        generator=_generator(seed, 3),
    )

    oracle_transform = fit_robust_oracle_transform(train_nodes.raw_rewards)
    train_rewards = oracle_transform(train_nodes.raw_rewards)
    test_rewards = oracle_transform(test_nodes.raw_rewards)
    true_train_margins = pair_margins(train_rewards[:, 0], train_rewards[:, 1])
    probabilities = btl_probabilities(true_train_margins)
    repeated_labels = sample_geometric_repeated_labels(
        probabilities,
        gamma=effective_config.annotation_gamma,
        generator=_generator(seed, 4),
    )
    h = repeated_labels.logit_estimates(gamma=effective_config.annotation_gamma)

    training_batch = FeatureTrainingBatch(
        left_features=train_nodes.features[:, 0],
        right_features=train_nodes.features[:, 1],
        edge_scores=train_nodes.scores[:, 0] - train_nodes.scores[:, 1],
        node_scores=train_nodes.scores.reshape(-1, effective_config.policy_dimension),
        h=h,
        left_wins=repeated_labels.wins,
        num_annotations=repeated_labels.counts,
    )
    initial_weight = torch.zeros(effective_config.reward_dimension, dtype=torch.float64)
    bt_model = FrozenFeatureLinearReward(
        effective_config.reward_dimension,
        initial_weight,
    )
    srm_model = FrozenFeatureLinearReward(
        effective_config.reward_dimension,
        initial_weight,
    )

    bt_config = BTMLETrainingConfig(
        learning_rate=effective_config.bt_learning_rate,
        optimizer="adamw",
        weight_decay=0.0,
        microbatch_size=effective_config.microbatch_size,
    )
    bt_initial = evaluate_bt_mle(bt_model, training_batch)
    bt_trainer = BTMLETrainer(bt_model, training_batch, bt_config)
    bt_trainer.fit(effective_config.bt_steps)
    bt_final = bt_trainer.evaluate()

    srm_config = SRMPlusTrainingConfig(
        learning_rate=effective_config.srm_learning_rate,
        optimizer="adamw",
        weight_decay=0.0,
        microbatch_size=effective_config.microbatch_size,
        beta=effective_config.beta,
        damping=effective_config.training_damping,
        pcg_max_iterations=effective_config.pcg_max_iterations,
        pcg_tolerance=effective_config.pcg_tolerance,
        require_pcg_convergence=True,
    )
    srm_initial = evaluate_srm_plus(srm_model, training_batch, srm_config).dual_loss
    srm_trainer = SRMPlusTrainer(srm_model, training_batch, srm_config)
    srm_history = srm_trainer.fit(effective_config.srm_steps)
    srm_final_evaluation = srm_trainer.evaluate()
    last_srm_step = srm_history[-1]
    srm_diagnostics = (
        last_srm_step.pcg_iterations,
        last_srm_step.pcg_residual_norm,
        last_srm_step.pcg_relative_residual,
        last_srm_step.pcg_converged,
        last_srm_step.dual_loss,
        last_srm_step.dual_saddle_value,
    )
    if any(value is None for value in srm_diagnostics):
        raise RuntimeError("SRM+ trainer omitted required PCG/dual diagnostics")

    train_prompt_ids = tuple(range(effective_config.num_train_prompts))
    test_prompt_ids = tuple(
        range(
            effective_config.num_train_prompts,
            effective_config.num_train_prompts + effective_config.num_test_prompts,
        )
    )
    return SyntheticExperimentResult(
        seed=seed,
        config=effective_config,
        train_prompt_ids=train_prompt_ids,
        test_prompt_ids=test_prompt_ids,
        annotation_counts=tuple(int(value) for value in repeated_labels.counts.tolist()),
        left_wins=tuple(int(value) for value in repeated_labels.wins.tolist()),
        h_values=tuple(float(value) for value in h.tolist()),
        oracle_center=oracle_transform.b,
        oracle_scale=oracle_transform.tau,
        train_misspecification_rmse=_misspecification_rmse(
            train_nodes.features,
            train_rewards,
        ),
        bt=_learner_result(
            bt_model,
            test_nodes,
            test_rewards,
            initial_train_objective=bt_initial,
            final_train_objective=bt_final,
            config=effective_config,
        ),
        srm=_learner_result(
            srm_model,
            test_nodes,
            test_rewards,
            initial_train_objective=srm_initial,
            final_train_objective=srm_final_evaluation.dual_loss,
            config=effective_config,
        ),
        srm_pcg=SyntheticPCGEvidence(
            last_train_iterations=last_srm_step.pcg_iterations,
            last_train_residual_norm=last_srm_step.pcg_residual_norm,
            last_train_relative_residual=last_srm_step.pcg_relative_residual,
            last_train_converged=last_srm_step.pcg_converged,
            last_train_dual_loss=last_srm_step.dual_loss,
            last_train_dual_saddle_value=last_srm_step.dual_saddle_value,
            evaluation_iterations=srm_final_evaluation.pcg_iterations,
            evaluation_residual_norm=srm_final_evaluation.pcg_residual_norm,
            evaluation_relative_residual=srm_final_evaluation.pcg_relative_residual,
            evaluation_converged=srm_final_evaluation.pcg_converged,
            evaluation_dual_loss=srm_final_evaluation.dual_loss,
            evaluation_dual_saddle_value=srm_final_evaluation.dual_saddle_value,
        ),
    )


__all__ = [
    "SyntheticExperimentConfig",
    "SyntheticExperimentResult",
    "SyntheticLearnerResult",
    "SyntheticPCGEvidence",
    "run_synthetic_experiment",
]
