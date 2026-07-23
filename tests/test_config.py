from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import smart_reward.config as config_module
from smart_reward.config import (
    ConfigError,
    MissingConfigDependencyError,
    canonical_json,
    config_hash,
    load_config,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def valid_config() -> dict[str, object]:
    return load_config(ROOT / "configs" / "smoke.yaml")


@pytest.mark.parametrize("name", ["smoke.yaml", "main.yaml"])
def test_checked_in_configs_satisfy_strict_schema(name: str) -> None:
    config = load_config(ROOT / "configs" / name)

    assert len(config_hash(config)) == 64
    assert config["policy"]["lora_dropout"] == 0.0
    assert config["policy"]["trainable_tangent_parameters"] == "lora_B_only"


def test_validation_is_recursive_and_does_not_mutate_input(valid_config: dict[str, object]) -> None:
    candidate = copy.deepcopy(valid_config)
    candidate["policy"]["sampling"]["typo"] = 1
    before = copy.deepcopy(candidate)

    with pytest.raises(ConfigError, match="policy.sampling.*unknown keys.*typo"):
        validate_config(candidate)

    assert candidate == before


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("run", "seed"), 2**63, "at most"),
        (("data", "num_candidates"), 3, "must equal 4"),
        (("policy", "dtype"), "bfloat16", "must equal float32"),
        (("policy", "lora_alpha"), 3, "lora_alpha == policy.lora_rank"),
        (("policy", "lora_dropout"), 0.1, "lora_dropout == 0"),
        (("oracle", "robust_scale_floor"), 1.0e-5, "locked value 1e-6"),
        (("oracle", "probability_floor"), 0.2, "locked value 0.25"),
        (("annotations", "gamma"), 0.85, "must equal.*0.9"),
        (
            ("objective", "damping_relative_to_mean_fisher_diagonal"),
            0.0,
            "must be > 0.0",
        ),
        (("reward_model", "refresh_dual_every_steps"), 2, "must equal 1"),
        (("reward_model", "optimizer"), "sgd", "must equal 'adamw'"),
        (("reward_model", "weight_decay"), 0.01, "must equal 0"),
        (("evaluation", "kl_budget"), 0.0, "must be > 0.0"),
    ],
)
def test_theory_and_training_invariants_are_fail_closed(
    valid_config: dict[str, object],
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    candidate = copy.deepcopy(valid_config)
    candidate[path[0]][path[1]] = value

    with pytest.raises(ConfigError, match=message):
        validate_config(candidate)


def test_split_names_and_sum_are_exact(valid_config: dict[str, object]) -> None:
    bad_name = copy.deepcopy(valid_config)
    bad_name["run"]["split_sizes"]["val"] = bad_name["run"]["split_sizes"].pop("validation")
    with pytest.raises(ConfigError, match="missing keys.*validation.*unknown keys.*val"):
        validate_config(bad_name)

    bad_sum = copy.deepcopy(valid_config)
    bad_sum["run"]["split_sizes"]["train"] -= 1
    with pytest.raises(ConfigError, match="sum exactly"):
        validate_config(bad_sum)

    too_small_test = copy.deepcopy(valid_config)
    too_small_test["run"]["split_sizes"]["test"] = 1
    too_small_test["run"]["split_sizes"]["train"] += 7
    with pytest.raises(ConfigError, match="test must be at least 2"):
        validate_config(too_small_test)


def test_run_name_must_be_a_safe_artifact_identifier(
    valid_config: dict[str, object],
) -> None:
    candidate = copy.deepcopy(valid_config)
    candidate["run"]["name"] = "../escaped"
    with pytest.raises(ConfigError, match="filesystem-safe"):
        validate_config(candidate)


def test_policy_sampling_is_the_exact_unwarped_distribution(
    valid_config: dict[str, object],
) -> None:
    candidate = copy.deepcopy(valid_config)
    candidate["policy"]["sampling"]["top_p"] = 0.95
    with pytest.raises(ConfigError, match="unwarped on-policy contract"):
        validate_config(candidate)


def test_evaluation_seed_and_probe_count_match_runtime_domain(
    valid_config: dict[str, object],
) -> None:
    invalid_seed = copy.deepcopy(valid_config)
    invalid_seed["evaluation"]["paired_bootstrap_seed"] = 2**63
    with pytest.raises(ConfigError, match="at most"):
        validate_config(invalid_seed)

    too_many_probes = copy.deepcopy(valid_config)
    too_many_probes["evaluation"]["kl_probe_candidates"] = 10_000
    with pytest.raises(ConfigError, match="cannot exceed"):
        validate_config(too_many_probes)


def test_pcg_cap_covers_the_empirical_fisher_rank_bound(
    valid_config: dict[str, object],
) -> None:
    candidate = copy.deepcopy(valid_config)
    train_nodes = candidate["run"]["split_sizes"]["train"] * candidate["data"]["num_candidates"]
    candidate["objective"]["pcg_max_iterations"] = train_nodes

    with pytest.raises(ConfigError, match="train Fisher nodes plus one"):
        validate_config(candidate)


def test_pcg_dtype_is_locked_to_float64(valid_config: dict[str, object]) -> None:
    candidate = copy.deepcopy(valid_config)
    candidate["objective"]["pcg_dtype"] = "float32"

    with pytest.raises(ConfigError, match="pcg_dtype must be 'float64'"):
        validate_config(candidate)


def test_frozen_reward_backbone_must_equal_policy_revision(
    valid_config: dict[str, object],
) -> None:
    candidate = copy.deepcopy(valid_config)
    candidate["reward_model"]["revision"] = "f" * 40

    with pytest.raises(ConfigError, match=r"reward_model\.revision.*policy\.revision"):
        validate_config(candidate)


def test_canonical_hash_is_key_order_independent_and_uses_compact_json() -> None:
    left = {"b": [2, {"z": 1}], "a": "值"}
    right = {"a": "值", "b": [2, {"z": 1}]}

    assert canonical_json(left) == '{"a":"值","b":[2,{"z":1}]}'
    assert config_hash(left) == config_hash(right)
    assert json.loads(canonical_json(left)) == left


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("run: {}\nrun: {}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate YAML key 'run'"):
        load_config(path)


def test_missing_pyyaml_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = config_module.importlib.import_module

    def fake_import(name: str) -> object:
        if name == "yaml":
            error = ModuleNotFoundError("No module named 'yaml'")
            error.name = "yaml"
            raise error
        return real_import(name)

    monkeypatch.setattr(config_module.importlib, "import_module", fake_import)

    with pytest.raises(MissingConfigDependencyError, match="install.*pyyaml"):
        load_config(ROOT / "configs" / "smoke.yaml")
