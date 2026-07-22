"""Numerical core for Smart Reward Model (SRM+) experiments."""

from .annotations import (
    RepeatedLabelBatch,
    geometric_annotation_counts,
    randomized_truncation_u_statistic_from_counts,
    repeated_labels_to_h,
    sample_geometric_repeated_labels,
)
from .artifacts import (
    artifact_metadata_sha256,
    load_controlled_feature_artifact,
    save_controlled_feature_artifact,
)
from .baseline import repeated_btl_nll
from .config import config_hash, load_config, validate_config
from .data import (
    CandidateNode,
    EvaluationEdgeRecord,
    EvaluationLeakageError,
    SchemaError,
    TrainingEdgeRecord,
    load_jsonl,
    save_jsonl,
    swap_edge_orientation,
    validate_disjoint_prompt_splits,
)
from .experiment import (
    ControlledFeatureExperiment,
    EvaluationTensorData,
    FeatureExperimentConfig,
    TrainingTensorData,
    compile_feature_experiment_config,
    run_feature_experiment,
)
from .linear import (
    DampedEmpiricalFisher,
    damped_fisher_diagonal,
    damped_fisher_matvec,
    fisher_diagonal,
    fisher_matvec,
)
from .metrics import (
    NaturalDirectionMetrics,
    empirical_fisher_matrix,
    gauge_center,
    local_regret,
    natural_direction,
    natural_direction_metrics,
    policy_reward_moment,
)
from .objective import (
    dual_loss,
    dual_saddle_value,
    empirical_moment,
    envelope_surrogate,
    envelope_weights,
    srm_dual_loss,
)
from .oracle import (
    RobustOracleTransform,
    btl_probabilities,
    fit_robust_oracle_transform,
    pair_margins,
)
from .pcg import PCGBreakdownError, PCGResult, pcg
from .phase1 import assemble_controlled_experiment, materialize_phase1
from .phase1_rollout import evaluate_matched_kl_rollouts
from .policy_update import (
    line_search_measured_kl,
    masked_causal_forward_kl,
    select_causal_response_logits,
    selected_causal_forward_kl,
    set_tangent_update_,
    step_size_for_kl_budget,
)
from .rollout import (
    match_fixed_a_measured_kl,
    oracle_rollout_improvement,
    policy_direction_from_head,
)
from .scores import (
    ParameterLayout,
    ParameterLayoutEntry,
    ScoreDiagnostics,
    edge_score_differences,
    empirical_score_diagnostics,
    per_sample_scores,
    select_named_tangent_parameters,
    sequence_log_probs,
)
from .statistics import aggregate_paired_metrics
from .synthetic import run_synthetic_experiment
from .training import (
    BTMLETrainer,
    FeatureTrainingBatch,
    FrozenFeatureLinearReward,
    SRMPlusTrainer,
)

__all__ = [
    "BTMLETrainer",
    "ControlledFeatureExperiment",
    "DampedEmpiricalFisher",
    "CandidateNode",
    "EvaluationEdgeRecord",
    "EvaluationLeakageError",
    "EvaluationTensorData",
    "FeatureExperimentConfig",
    "FeatureTrainingBatch",
    "FrozenFeatureLinearReward",
    "NaturalDirectionMetrics",
    "ParameterLayout",
    "ParameterLayoutEntry",
    "PCGBreakdownError",
    "PCGResult",
    "RepeatedLabelBatch",
    "RobustOracleTransform",
    "SchemaError",
    "ScoreDiagnostics",
    "SRMPlusTrainer",
    "TrainingTensorData",
    "TrainingEdgeRecord",
    "btl_probabilities",
    "artifact_metadata_sha256",
    "compile_feature_experiment_config",
    "config_hash",
    "damped_fisher_diagonal",
    "damped_fisher_matvec",
    "dual_loss",
    "dual_saddle_value",
    "edge_score_differences",
    "empirical_fisher_matrix",
    "empirical_moment",
    "empirical_score_diagnostics",
    "evaluate_matched_kl_rollouts",
    "envelope_surrogate",
    "envelope_weights",
    "fisher_diagonal",
    "fisher_matvec",
    "fit_robust_oracle_transform",
    "gauge_center",
    "geometric_annotation_counts",
    "line_search_measured_kl",
    "local_regret",
    "load_config",
    "load_controlled_feature_artifact",
    "match_fixed_a_measured_kl",
    "masked_causal_forward_kl",
    "materialize_phase1",
    "natural_direction",
    "natural_direction_metrics",
    "load_jsonl",
    "pair_margins",
    "pcg",
    "per_sample_scores",
    "policy_reward_moment",
    "policy_direction_from_head",
    "oracle_rollout_improvement",
    "randomized_truncation_u_statistic_from_counts",
    "repeated_labels_to_h",
    "repeated_btl_nll",
    "save_jsonl",
    "sample_geometric_repeated_labels",
    "save_controlled_feature_artifact",
    "select_causal_response_logits",
    "selected_causal_forward_kl",
    "set_tangent_update_",
    "select_named_tangent_parameters",
    "sequence_log_probs",
    "srm_dual_loss",
    "step_size_for_kl_budget",
    "swap_edge_orientation",
    "validate_disjoint_prompt_splits",
    "aggregate_paired_metrics",
    "assemble_controlled_experiment",
    "run_feature_experiment",
    "run_synthetic_experiment",
    "validate_config",
]
