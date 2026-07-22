"""Command-line control plane for Smart Reward Model experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from types import ModuleType

from .config import ConfigError, config_hash, load_config
from .data import SchemaError, TrainingEdgeRecord, load_jsonl
from .repro import atomic_write_json, build_run_manifest, collect_execution_identity


def _resolve_run_seed(config: dict[str, object], requested_seed: int | None) -> int:
    run = config["run"]
    if not isinstance(run, dict):  # load_config already guarantees this.
        raise ConfigError("run must be a mapping")
    allowed = [int(run["seed"])] if "seed" in run else [int(seed) for seed in run["seeds"]]
    if requested_seed is None:
        if len(allowed) != 1:
            raise ConfigError("--seed is required when run.seeds contains multiple seeds")
        return allowed[0]
    if requested_seed not in allowed:
        raise ConfigError(f"seed {requested_seed} is not declared by the configuration")
    return requested_seed


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_result_float(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _read_json_object(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"JSON input exceeds 64 MiB: {source}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def _lower_hex_digest(value: object, *, path: str, lengths: set[int]) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase hexadecimal digest")
    return value


def _run_environment_identity(
    manifest_path: str | Path,
    *,
    expected_config_hash: str,
    expected_seed: int,
    require_formal: bool,
    match_current_environment: bool = False,
) -> tuple[str, dict[str, object]]:
    path = Path(manifest_path)
    value = _read_json_object(path)
    if value.get("schema_version") != "smart-reward-run/v1":
        raise ValueError(f"{path} is not a smart-reward-run/v1 manifest")
    if value.get("config_hash") != expected_config_hash:
        raise ValueError(f"{path} config hash does not match the controlled run")
    if value.get("selected_seed") != expected_seed:
        raise ValueError(f"{path} selected_seed does not match the controlled run")

    git = value.get("git")
    slurm = value.get("slurm")
    torch_state = value.get("torch")
    if not all(isinstance(item, dict) for item in (git, slurm, torch_state)):
        raise ValueError(f"{path} has invalid git/slurm/torch evidence")
    commit = _lower_hex_digest(git.get("commit"), path=f"{path}:git.commit", lengths={40, 64})
    dirty = git.get("dirty")
    if not isinstance(dirty, bool):
        raise ValueError(f"{path}:git.dirty must be boolean")

    image = slurm.get("SRM_IMAGE_SHA256")
    environment_commit = slurm.get("SRM_GIT_COMMIT")
    account = slurm.get("SLURM_JOB_ACCOUNT")
    partition = slurm.get("SLURM_JOB_PARTITION")
    gpus = torch_state.get("gpus")
    gpu_names = (
        [gpu.get("name") for gpu in gpus if isinstance(gpu, dict)] if isinstance(gpus, list) else []
    )
    complete = (
        dirty is False
        and isinstance(image, str)
        and isinstance(environment_commit, str)
        and environment_commit == commit
        and isinstance(partition, str)
        and bool(partition)
        and account == "sigroup"
        and torch_state.get("cuda_available") is True
        and torch_state.get("gpu_count") == 1
        and len(gpu_names) == 1
        and isinstance(gpu_names[0], str)
        and bool(gpu_names[0])
    )
    if require_formal and not complete:
        raise ValueError(f"{path} lacks a clean, single-GPU Slurm/image/Git environment identity")
    if complete:
        image = _lower_hex_digest(image, path=f"{path}:slurm.SRM_IMAGE_SHA256", lengths={64})
    else:
        commit = None
        image = None
        account = None
        partition = None
        gpu_names = []
    identity = {
        "formal": complete,
        "git_commit": commit,
        "image_sha256": image,
        "account": account,
        "partition": partition,
        "gpu_models": gpu_names,
    }
    if match_current_environment and collect_execution_identity() != identity:
        raise ValueError(f"{path} environment identity does not match the current process")
    return _sha256_file(path), identity


def _formal_execution_requested() -> bool:
    """Return whether the caller explicitly entered the formal Slurm protocol."""

    return any(
        bool(os.environ.get(name))
        for name in ("SLURM_JOB_ID", "SRM_GIT_COMMIT", "SRM_IMAGE_SHA256")
    )


def _start_cuda_memory_tracking() -> ModuleType:
    """Reset the single visible GPU's PyTorch peak-memory counters."""

    import importlib

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SRM_MEMORY_REPORT requires exactly one visible CUDA GPU")
    torch.cuda.reset_peak_memory_stats(0)
    return torch


def _write_cuda_memory_report(
    torch: ModuleType,
    destination: str | Path,
    *,
    command: str,
    status: str,
) -> None:
    """Persist allocator peak evidence after a model-stage CLI invocation."""

    torch.cuda.synchronize(0)
    atomic_write_json(
        destination,
        {
            "schema_version": "cuda-memory-peak/v1",
            "command": command,
            "status": status,
            "device_index": 0,
            "device_name": str(torch.cuda.get_device_name(0)),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        },
    )


def _config_check(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    _print_json(
        {
            "config_hash": config_hash(config),
            "path": str(Path(arguments.config)),
            "status": "ok",
        }
    )
    return 0


def _env_report(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    selected_seed = (
        None
        if arguments.seed is None and isinstance(config["run"].get("seeds"), list)
        else _resolve_run_seed(config, arguments.seed)
    )
    manifest = build_run_manifest(
        config,
        repo_path=arguments.repo_root,
        selected_seed=selected_seed,
    )
    if arguments.output is None:
        _print_json(manifest.to_dict())
    else:
        atomic_write_json(arguments.output, manifest)
        _print_json(
            {
                "config_hash": manifest.config_hash,
                "output": str(Path(arguments.output)),
                "status": "ok",
            }
        )
    return 0


def _data_check(arguments: argparse.Namespace) -> int:
    records = load_jsonl(arguments.jsonl, TrainingEdgeRecord)
    if not records:
        raise SchemaError("training JSONL must contain at least one record")
    seen: set[str | int] = set()
    duplicates: list[str | int] = []
    for record in records:
        if record.edge_id in seen:
            duplicates.append(record.edge_id)
        seen.add(record.edge_id)
    if duplicates:
        ordered = sorted({repr(item) for item in duplicates})
        raise SchemaError(f"duplicate edge_id values are forbidden: {ordered!r}")
    _print_json(
        {
            "annotations": sum(record.num_annotations for record in records),
            "edges": len(records),
            "path": str(Path(arguments.jsonl)),
            "prompts": len({record.prompt_id for record in records}),
            "schema_version": records[0].schema_version,
            "status": "ok",
        }
    )
    return 0


def _prepare_prompts(arguments: argparse.Namespace) -> int:
    from .prompts import load_multipref_prompts, save_prompt_jsonl
    from .seeding import SeedBundle

    config = load_config(arguments.config)
    seed = _resolve_run_seed(config, arguments.seed)
    split_seed = SeedBundle.from_base_seed(seed).prompt_split
    run = config["run"]
    data = config["data"]
    records = load_multipref_prompts(
        dataset_name=str(data["prompt_dataset"]),
        revision=str(data["prompt_revision"]),
        split_sizes=run["split_sizes"],
        seed=split_seed,
    )
    save_prompt_jsonl(arguments.output, records)
    _print_json(
        {
            "config_hash": config_hash(config),
            "output": str(Path(arguments.output)),
            "prompts": len(records),
            "seed": seed,
            "prompt_split_seed": split_seed,
            "status": "ok",
        }
    )
    return 0


def _synthetic_check(arguments: argparse.Namespace) -> int:
    from .synthetic import run_synthetic_experiment

    payload = asdict(run_synthetic_experiment(seed=arguments.seed))
    payload["benchmark_only"] = True
    payload["status"] = "ok"
    if arguments.output is None:
        _print_json(payload)
    else:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination, payload)
        _print_json({"output": str(destination), "seed": arguments.seed, "status": "ok"})
    return 0


def _controlled_materialize(arguments: argparse.Namespace) -> int:
    from .phase1 import materialize_phase1

    config = load_config(arguments.config)
    seed = _resolve_run_seed(config, arguments.seed)
    materialization = materialize_phase1(
        config,
        seed=seed,
        artifact_dir=arguments.artifact_dir,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    _print_json(
        {
            "artifact_dir": str(materialization.artifact_directory),
            "config_hash": config_hash(config),
            "seed": seed,
            "status": "ok",
        }
    )
    return 0


def _damping_multipliers(config: dict[str, object]) -> tuple[float, ...]:
    objective = config["objective"]
    if not isinstance(objective, dict):
        raise ConfigError("objective must be a mapping")
    raw = objective.get("damping_sensitivity_multipliers", [1.0])
    values = tuple(float(value) for value in raw)
    # The primary run is required by every downstream stage and must be
    # completed before an ill-conditioned sensitivity solve is attempted.
    return (1.0, *(value for value in values if value != 1.0))


def _controlled_compare(arguments: argparse.Namespace) -> int:
    from .artifacts import artifact_metadata_sha256, load_controlled_feature_artifact
    from .experiment import (
        ControlledFeatureExperiment,
        EvaluationTensorData,
        TrainingTensorData,
        compile_feature_experiment_config,
        run_feature_experiment,
    )

    config = load_config(arguments.config)
    seed = _resolve_run_seed(config, arguments.seed)
    digest = config_hash(config)
    destination = Path(arguments.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    formal_execution = _formal_execution_requested()
    manifest_sha256, environment_identity = _run_environment_identity(
        arguments.run_manifest,
        expected_config_hash=digest,
        expected_seed=seed,
        require_formal=formal_execution,
        # Always match the parsed manifest to this process.  An old formal
        # manifest cannot be replayed from an unrelated local workstation.
        match_current_environment=True,
    )
    experiment = load_controlled_feature_artifact(
        arguments.artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    artifact_identity = artifact_metadata_sha256(
        arguments.artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    artifact_metadata = _read_json_object(Path(arguments.artifact_dir) / "metadata.json")
    artifact_evidence = artifact_metadata.get("evidence")
    artifact_producer = (
        artifact_evidence.get("producer") if isinstance(artifact_evidence, dict) else None
    )
    if not isinstance(artifact_producer, dict):
        raise ValueError("artifact metadata is missing producer identity")
    if environment_identity["formal"] and artifact_producer != {
        "git_commit": environment_identity["git_commit"],
        "image_sha256": environment_identity["image_sha256"],
    }:
        raise ValueError("artifact producer does not match the run manifest environment")
    if arguments.device != "cpu":
        train = experiment.train
        validation = experiment.validation
        test = experiment.test
        experiment = ControlledFeatureExperiment(
            train=TrainingTensorData(
                prompt_ids=train.prompt_ids,
                policy_scores=train.policy_scores.to(arguments.device),
                reward_features=train.reward_features.to(arguments.device),
                h=train.h.to(arguments.device),
                left_wins=train.left_wins.to(arguments.device),
                num_annotations=train.num_annotations.to(arguments.device),
            ),
            validation=EvaluationTensorData(
                prompt_ids=validation.prompt_ids,
                policy_scores=validation.policy_scores.to(arguments.device),
                reward_features=validation.reward_features.to(arguments.device),
                true_rewards=validation.true_rewards.to(arguments.device),
            ),
            test=EvaluationTensorData(
                prompt_ids=test.prompt_ids,
                policy_scores=test.policy_scores.to(arguments.device),
                reward_features=test.reward_features.to(arguments.device),
                true_rewards=test.true_rewards.to(arguments.device),
            ),
        )
    damping_runs = []
    for multiplier in _damping_multipliers(config):
        runtime = compile_feature_experiment_config(
            config,
            damping_multiplier=multiplier,
        )
        try:
            result = run_feature_experiment(experiment, runtime).to_dict()
        except RuntimeError as error:
            if multiplier == 1.0 or "PCG did not converge" not in str(error):
                raise
            result = {
                "status": "failed",
                "failure_type": "pcg_nonconvergence",
                "message": str(error),
            }
        damping_runs.append({"damping_multiplier": multiplier, "result": result})
    payload = {
        "schema_version": "controlled-comparison/v1",
        "config_hash": digest,
        "seed": seed,
        "artifact_dir": str(Path(arguments.artifact_dir)),
        "artifact_metadata_sha256": artifact_identity,
        "run_manifest": str(Path(arguments.run_manifest)),
        "run_manifest_sha256": manifest_sha256,
        "environment_identity": environment_identity,
        "damping_runs": damping_runs,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload)
    _print_json(
        {
            "config_hash": digest,
            "damping_runs": len(damping_runs),
            "output": str(destination),
            "seed": seed,
            "status": "ok",
        }
    )
    return 0


def _controlled_rollout(arguments: argparse.Namespace) -> int:
    from .phase1_rollout import evaluate_matched_kl_rollouts

    config = load_config(arguments.config)
    seed = _resolve_run_seed(config, arguments.seed)
    payload = evaluate_matched_kl_rollouts(
        config,
        seed=seed,
        artifact_dir=arguments.artifact_dir,
        comparison_json=arguments.comparison,
        output_json=arguments.output,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
    )
    _print_json(
        {
            "config_hash": payload["config_hash"],
            "output": str(Path(arguments.output)),
            "seed": seed,
            "status": "ok",
            "updated_rollouts": str(Path(arguments.output).parent / "updated_rollouts.jsonl"),
        }
    )
    return 0


def _load_comparison_metrics(
    paths: Sequence[str],
    *,
    expected_config_hash: str,
    expected_damping_multipliers: tuple[float, ...],
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[str, float]],
    dict[int, dict[str, object]],
    dict[float, dict[int, dict[str, object]]],
]:
    bt_by_seed: dict[int, dict[str, float]] = {}
    srm_by_seed: dict[int, dict[str, float]] = {}
    sources: dict[int, dict[str, object]] = {}
    damping_evidence: dict[float, dict[int, dict[str, object]]] = {
        multiplier: {} for multiplier in expected_damping_multipliers
    }
    for raw_path in paths:
        path = Path(raw_path)
        value = _read_json_object(path)
        if not isinstance(value, dict) or value.get("schema_version") != "controlled-comparison/v1":
            raise ValueError(f"{path} is not a controlled-comparison/v1 result")
        if value.get("config_hash") != expected_config_hash:
            raise ValueError(f"{path} config_hash does not match the aggregation config")
        artifact_identity = value.get("artifact_metadata_sha256")
        if (
            not isinstance(artifact_identity, str)
            or len(artifact_identity) != 64
            or any(character not in "0123456789abcdef" for character in artifact_identity)
        ):
            raise ValueError(f"{path} has an invalid artifact_metadata_sha256")
        seed = value.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{path} has an invalid seed")
        if seed in bt_by_seed:
            raise ValueError(f"duplicate comparison result for seed {seed}")
        manifest_name = Path(str(value.get("run_manifest", ""))).name
        if manifest_name != "run-manifest.json":
            raise ValueError(f"{path} records an invalid run manifest filename")
        manifest_path = path.parent / manifest_name
        recorded_manifest_sha = _lower_hex_digest(
            value.get("run_manifest_sha256"),
            path=f"{path}:run_manifest_sha256",
            lengths={64},
        )
        manifest_sha, environment_identity = _run_environment_identity(
            manifest_path,
            expected_config_hash=expected_config_hash,
            expected_seed=seed,
            require_formal=True,
        )
        if manifest_sha != recorded_manifest_sha:
            raise ValueError(f"{path} run-manifest.json SHA256 mismatch")
        if value.get("environment_identity") != environment_identity:
            raise ValueError(f"{path} environment identity does not match its run manifest")
        raw_runs = value.get("damping_runs")
        if not isinstance(raw_runs, list):
            raise ValueError(f"{path} damping_runs must be a list")
        runs: dict[float, dict[str, object]] = {}
        for run_index, run in enumerate(raw_runs):
            if not isinstance(run, dict) or set(run) != {"damping_multiplier", "result"}:
                raise ValueError(f"{path} damping_runs[{run_index}] has an invalid schema")
            multiplier = _finite_result_float(
                run["damping_multiplier"], path=f"{path}:damping_runs[{run_index}]"
            )
            if multiplier in runs:
                raise ValueError(f"{path} repeats damping multiplier {multiplier}")
            if multiplier not in damping_evidence:
                raise ValueError(f"{path} contains undeclared damping multiplier {multiplier}")
            result_value = run["result"]
            if not isinstance(result_value, dict):
                raise ValueError(f"{path} damping result must be an object")
            runs[multiplier] = result_value
        if set(runs) != set(expected_damping_multipliers):
            raise ValueError(f"{path} damping multipliers do not exactly match the config")

        for multiplier, damping_result in runs.items():
            if damping_result.get("status") == "failed":
                if multiplier == 1.0:
                    raise ValueError(f"{path} primary damping run may not be failed")
                damping_evidence[multiplier][seed] = {
                    "status": "failed",
                    "failure_type": str(damping_result.get("failure_type", "unknown")),
                    "message": str(damping_result.get("message", "")),
                }
                continue
            local_regret: dict[str, float] = {}
            for learner_name in ("bt_mle", "srm_plus"):
                learner_value = damping_result.get(learner_name)
                test_value = learner_value.get("test") if isinstance(learner_value, dict) else None
                if not isinstance(test_value, dict):
                    raise ValueError(f"{path} damping={multiplier} is missing {learner_name}.test")
                local_regret[learner_name] = _finite_result_float(
                    test_value.get("local_regret"),
                    path=f"{path}:damping={multiplier}:{learner_name}.test.local_regret",
                )
            srm_value = damping_result.get("srm_plus")
            final_pcg = srm_value.get("final_pcg") if isinstance(srm_value, dict) else None
            pcg_converged = isinstance(final_pcg, dict) and final_pcg.get("converged") is True
            damping_evidence[multiplier][seed] = {
                "status": "ok",
                "bt_local_regret": local_regret["bt_mle"],
                "srm_local_regret": local_regret["srm_plus"],
                "pcg_converged": pcg_converged,
            }

        result = runs[1.0]
        if not isinstance(result, dict):
            raise ValueError(f"{path} contains an invalid main result")
        learners: dict[str, dict[str, float]] = {}
        for key in ("bt_mle", "srm_plus"):
            learner = result.get(key)
            test = learner.get("test") if isinstance(learner, dict) else None
            if not isinstance(test, dict):
                raise ValueError(f"{path} is missing {key}.test metrics")
            learners[key] = {
                "test_local_regret": _finite_result_float(
                    test.get("local_regret"), path=f"{path}:{key}.test.local_regret"
                ),
                "test_squared_fisher_error": _finite_result_float(
                    test.get("squared_fisher_error"),
                    path=f"{path}:{key}.test.squared_fisher_error",
                ),
                "test_fisher_cosine": _finite_result_float(
                    test.get("fisher_cosine"), path=f"{path}:{key}.test.fisher_cosine"
                ),
                "test_pairwise_accuracy": _finite_result_float(
                    test.get("pairwise_accuracy"),
                    path=f"{path}:{key}.test.pairwise_accuracy",
                ),
            }
        bt_by_seed[seed] = learners["bt_mle"]
        srm_by_seed[seed] = learners["srm_plus"]
        sources[seed] = {
            "comparison_path": str(path),
            "comparison_sha256": _sha256_file(path),
            "artifact_metadata_sha256": artifact_identity,
            "run_manifest_path": str(manifest_path),
            "run_manifest_sha256": manifest_sha,
            "environment_identity": environment_identity,
        }
    return bt_by_seed, srm_by_seed, sources, damping_evidence


def _load_rollout_metrics(
    paths: Sequence[str],
    *,
    expected_config_hash: str,
    comparison_sources: dict[int, dict[str, object]],
    expected_kl: float,
    kl_relative_tolerance: float,
) -> tuple[dict[int, float], dict[int, float], dict[int, dict[str, object]]]:
    bt_by_seed: dict[int, float] = {}
    srm_by_seed: dict[int, float] = {}
    sources: dict[int, dict[str, object]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        value = _read_json_object(path)
        if not isinstance(value, dict) or value.get("schema_version") != "matched-kl-rollout/v1":
            raise ValueError(f"{path} is not a matched-kl-rollout/v1 result")
        if value.get("config_hash") != expected_config_hash:
            raise ValueError(f"{path} config_hash does not match the aggregation config")
        seed = value.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{path} has an invalid seed")
        if seed in bt_by_seed:
            raise ValueError(f"duplicate rollout result for seed {seed}")
        comparison_source = comparison_sources.get(seed)
        if comparison_source is None:
            raise ValueError(f"{path} has no same-seed controlled comparison")
        if value.get("artifact_metadata_sha256") != comparison_source["artifact_metadata_sha256"]:
            raise ValueError(f"{path} is bound to a different artifact")
        if value.get("comparison_sha256") != comparison_source["comparison_sha256"]:
            raise ValueError(f"{path} is bound to different comparison bytes")
        if value.get("run_manifest_sha256") != comparison_source["run_manifest_sha256"]:
            raise ValueError(f"{path} is bound to a different run manifest")
        if value.get("environment_identity") != comparison_source["environment_identity"]:
            raise ValueError(f"{path} is bound to a different execution environment")

        recorded_rollouts = value.get("updated_rollouts_sha256")
        if (
            not isinstance(recorded_rollouts, str)
            or len(recorded_rollouts) != 64
            or any(character not in "0123456789abcdef" for character in recorded_rollouts)
        ):
            raise ValueError(f"{path} has an invalid updated_rollouts_sha256")
        rollouts_name = Path(str(value.get("updated_rollouts_jsonl", ""))).name
        if rollouts_name != "updated_rollouts.jsonl":
            raise ValueError(f"{path} records an invalid updated rollouts filename")
        rollouts_path = path.parent / rollouts_name
        if _sha256_file(rollouts_path) != recorded_rollouts:
            raise ValueError(f"{path} updated_rollouts.jsonl SHA256 mismatch")

        reference = value.get("test_reference")
        if not isinstance(reference, dict) or reference.get("source") != (
            "zero_b_common_random_number_rollout"
        ):
            raise ValueError(f"{path} does not use the required zero-B CRN reference")
        num_prompts = reference.get("num_prompts")
        if isinstance(num_prompts, bool) or not isinstance(num_prompts, int) or num_prompts < 2:
            raise ValueError(f"{path} has an invalid test prompt count")
        learners = value.get("learners")
        if not isinstance(learners, dict) or set(learners) != {"bt_mle", "srm_plus"}:
            raise ValueError(f"{path} must contain exactly BT-MLE and SRM+ rollout results")
        parsed: dict[str, float] = {}
        for learner_name in ("bt_mle", "srm_plus"):
            learner = learners[learner_name]
            direction = learner.get("direction") if isinstance(learner, dict) else None
            direction_pcg = direction.get("pcg") if isinstance(direction, dict) else None
            if not isinstance(direction_pcg, dict) or direction_pcg.get("converged") is not True:
                raise ValueError(f"{path} {learner_name} policy-direction PCG did not converge")
            update = learner.get("measured_kl_update") if isinstance(learner, dict) else None
            if (
                not isinstance(update, dict)
                or update.get("converged") is not True
                or update.get("applied") is not True
            ):
                raise ValueError(f"{path} {learner_name} measured-KL update did not converge")
            target_kl = _finite_result_float(
                update.get("target_kl"), path=f"{path}:{learner_name}.target_kl"
            )
            applied_kl = _finite_result_float(
                update.get("applied_measured_kl"),
                path=f"{path}:{learner_name}.applied_measured_kl",
            )
            if target_kl != expected_kl:
                raise ValueError(f"{path} {learner_name} used the wrong KL target")
            if abs(applied_kl - expected_kl) / expected_kl > kl_relative_tolerance:
                raise ValueError(f"{path} {learner_name} did not meet the measured-KL tolerance")
            paired = (
                learner.get("paired_improvement_over_zero_b_reference")
                if isinstance(learner, dict)
                else None
            )
            if not isinstance(paired, dict) or paired.get("schema_version") != (
                "oracle-rollout-improvement/v1"
            ):
                raise ValueError(f"{path} has invalid {learner_name} paired improvement")
            if paired.get("num_pairs") != num_prompts:
                raise ValueError(f"{path} {learner_name} uncertainty is not prompt-level")
            if paired.get("significance_claimed") is not False:
                raise ValueError(f"{path} {learner_name} must not claim significance")
            parsed[learner_name] = _finite_result_float(
                paired.get("mean_difference"),
                path=f"{path}:{learner_name}.paired_improvement.mean_difference",
            )
        bt_by_seed[seed] = parsed["bt_mle"]
        srm_by_seed[seed] = parsed["srm_plus"]
        sources[seed] = {
            **comparison_source,
            "rollout_path": str(path),
            "rollout_sha256": _sha256_file(path),
            "updated_rollouts_path": str(rollouts_path),
            "updated_rollouts_sha256": recorded_rollouts,
        }
        if value.get("train_oracle_values_accessed") is not False:
            raise ValueError(f"{path} accessed train oracle values")
        if value.get("raw_oracle_values_serialized") is not False:
            raise ValueError(f"{path} serialized raw oracle values")
    return bt_by_seed, srm_by_seed, sources


def _aggregate_damping_evidence(
    damping_evidence: dict[float, dict[int, dict[str, object]]],
    *,
    declared_seeds: set[int],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> tuple[list[dict[str, object]], bool, bool]:
    from .statistics import aggregate_paired_metrics

    rows: list[dict[str, object]] = []
    all_pcg_converged = True
    sensitivity_nonreversal = True
    for multiplier in sorted(damping_evidence):
        per_seed = damping_evidence[multiplier]
        if set(per_seed) != declared_seeds:
            raise ValueError(f"damping={multiplier} seed set does not match config run.seeds")
        failures: list[dict[str, object]] = []
        bt: dict[int, dict[str, float]] = {}
        srm: dict[int, dict[str, float]] = {}
        for seed in sorted(declared_seeds):
            record = per_seed[seed]
            if record.get("status") != "ok" or record.get("pcg_converged") is not True:
                all_pcg_converged = False
                failures.append({"seed": seed, **record})
                continue
            bt[seed] = {"test_local_regret": float(record["bt_local_regret"])}
            srm[seed] = {"test_local_regret": float(record["srm_local_regret"])}
        if failures:
            if multiplier != 1.0:
                sensitivity_nonreversal = False
            rows.append(
                {
                    "damping_multiplier": multiplier,
                    "status": "incomplete",
                    "all_pcg_converged": False,
                    "local_regret_nonreversal": False,
                    "failures": failures,
                }
            )
            continue
        aggregate = aggregate_paired_metrics(
            bt,
            srm,
            bootstrap_seed=bootstrap_seed,
            num_resamples=bootstrap_resamples,
        ).to_dict()
        summary = aggregate["metrics"]["test_local_regret"]
        # The preregistration uses the strict negative sign.  An exact zero is
        # inconclusive rather than counted as robustness evidence.
        nonreversal = float(summary["paired_mean"]) < 0.0
        if multiplier != 1.0 and not nonreversal:
            sensitivity_nonreversal = False
        rows.append(
            {
                "damping_multiplier": multiplier,
                "status": "ok",
                "all_pcg_converged": True,
                "local_regret_nonreversal": nonreversal,
                "paired_local_regret": summary,
                "failures": [],
            }
        )
    return rows, all_pcg_converged, sensitivity_nonreversal


def _pre_registered_evidence_status(
    paired_metrics: dict[str, object],
    *,
    all_pcg_converged: bool,
    sensitivity_nonreversal: bool,
) -> dict[str, object]:
    metrics = paired_metrics["metrics"]

    def summary(name: str) -> dict[str, object]:
        value = metrics[name]
        if not isinstance(value, dict):
            raise ValueError(f"aggregate metric {name!r} has an invalid schema")
        return value

    local = summary("test_local_regret")
    error = summary("test_squared_fisher_error")
    cosine = summary("test_fisher_cosine")
    rollout = summary("test_rollout_improvement")
    criteria = {
        "main_local_regret_negative_with_ci": (
            float(local["paired_mean"]) < 0.0 and float(local["bootstrap_ci"]["upper"]) < 0.0
        ),
        "main_direction_error_negative_with_ci": (
            float(error["paired_mean"]) < 0.0 and float(error["bootstrap_ci"]["upper"]) < 0.0
        ),
        "main_fisher_cosine_positive": float(cosine["paired_mean"]) > 0.0,
        "matched_kl_rollout_positive_with_ci": (
            float(rollout["paired_mean"]) > 0.0 and float(rollout["bootstrap_ci"]["lower"]) > 0.0
        ),
        "sensitivity_local_regret_nonreversal": sensitivity_nonreversal,
        "all_pcg_converged": all_pcg_converged,
        # Non-converged/out-of-tolerance KL runs are rejected by the rollout
        # loader before an aggregate can be written.
        "all_measured_kl_updates_converged": True,
    }
    passed = all(criteria.values())
    return {
        "status": "passed" if passed else "not_passed",
        "supports_pre_registered_claim": passed,
        "criteria": criteria,
    }


def _aggregate_results(arguments: argparse.Namespace) -> int:
    from .statistics import aggregate_paired_metrics

    config = load_config(arguments.config)
    declared = config["run"].get("seeds")
    if not isinstance(declared, list):
        raise ConfigError("aggregate-results requires a config with run.seeds")
    digest = config_hash(config)
    damping_multipliers = _damping_multipliers(config)
    bt_by_seed, srm_by_seed, comparison_sources, damping_evidence = _load_comparison_metrics(
        arguments.results,
        expected_config_hash=digest,
        expected_damping_multipliers=damping_multipliers,
    )
    if set(bt_by_seed) != set(int(seed) for seed in declared):
        raise ValueError("comparison result seeds must exactly match config run.seeds")
    bt_rollout, srm_rollout, sources = _load_rollout_metrics(
        arguments.rollouts,
        expected_config_hash=digest,
        comparison_sources=comparison_sources,
        expected_kl=float(config["evaluation"]["kl_budget"]),
        kl_relative_tolerance=float(config["evaluation"]["kl_relative_tolerance"]),
    )
    if set(bt_rollout) != set(bt_by_seed):
        raise ValueError("rollout result seeds must exactly match comparison seeds")
    ordered_seeds = sorted(sources)
    shared_environment = sources[ordered_seeds[0]].get("environment_identity")
    if not isinstance(shared_environment, dict) or shared_environment.get("formal") is not True:
        raise ValueError("aggregate-results requires complete formal environment identities")
    if any(
        sources[seed].get("environment_identity") != shared_environment
        for seed in ordered_seeds[1:]
    ):
        raise ValueError(
            "all paired seeds must use the same Git commit, image, account, "
            "partition, and GPU model"
        )
    for seed in bt_by_seed:
        bt_by_seed[seed]["test_rollout_improvement"] = bt_rollout[seed]
        srm_by_seed[seed]["test_rollout_improvement"] = srm_rollout[seed]
    evaluation = config["evaluation"]
    aggregate = aggregate_paired_metrics(
        bt_by_seed,
        srm_by_seed,
        directions={
            "test_pairwise_accuracy": "higher_is_better",
            "test_rollout_improvement": "higher_is_better",
        },
        bootstrap_seed=int(evaluation["paired_bootstrap_seed"]),
        num_resamples=int(evaluation["paired_bootstrap_resamples"]),
    )
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = aggregate.to_dict()
    sensitivity, all_pcg, nonreversal = _aggregate_damping_evidence(
        damping_evidence,
        declared_seeds=set(int(seed) for seed in declared),
        bootstrap_seed=int(evaluation["paired_bootstrap_seed"]),
        bootstrap_resamples=int(evaluation["paired_bootstrap_resamples"]),
    )
    payload["config_hash"] = digest
    payload["environment_identity"] = shared_environment
    payload["damping_evidence"] = sensitivity
    payload["pre_registered_evidence"] = _pre_registered_evidence_status(
        payload,
        all_pcg_converged=all_pcg,
        sensitivity_nonreversal=nonreversal,
    )
    payload["sources"] = [{"seed": seed, **sources[seed]} for seed in sorted(sources)]
    atomic_write_json(destination, payload)
    _print_json(
        {
            "evidence_status": payload["pre_registered_evidence"]["status"],
            "num_seeds": len(bt_by_seed),
            "output": str(destination),
            "status": "ok",
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser without touching network or GPU state."""

    parser = argparse.ArgumentParser(
        prog="smart-reward",
        description="Validated controls and experiment entry points for Smart Reward Model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser(
        "config-check",
        help="strictly validate a YAML config and print its canonical SHA-256 hash",
    )
    config_parser.add_argument("config", help="path to configs/smoke.yaml or configs/main.yaml")
    config_parser.set_defaults(handler=_config_check)

    environment_parser = subparsers.add_parser(
        "env-report",
        help="emit an offline run manifest for a validated config",
    )
    environment_parser.add_argument("config", help="path to a YAML experiment config")
    environment_parser.add_argument(
        "--repo-root",
        default=".",
        help="Git repository root (default: current directory)",
    )
    environment_parser.add_argument(
        "--seed",
        type=int,
        help="selected seed for this run (required later by controlled-compare)",
    )
    environment_parser.add_argument(
        "--output",
        "-o",
        help="atomically write JSON here instead of printing the manifest",
    )
    environment_parser.set_defaults(handler=_env_report)

    data_parser = subparsers.add_parser(
        "data-check",
        help="validate an exact TrainingEdgeRecord JSONL file",
    )
    data_parser.add_argument("jsonl", help="path to training-edge/v1 JSONL")
    data_parser.set_defaults(handler=_data_check)

    prompt_parser = subparsers.add_parser(
        "prepare-prompts",
        help="download a pinned prompt dataset and write deterministic prompt splits",
    )
    prompt_parser.add_argument("config", help="path to a validated YAML experiment config")
    prompt_parser.add_argument("output", help="destination prompt/v1 JSONL")
    prompt_parser.add_argument("--seed", type=int, help="one seed declared by run.seed(s)")
    prompt_parser.set_defaults(handler=_prepare_prompts)

    synthetic_parser = subparsers.add_parser(
        "synthetic-check",
        help="run the CPU-only end-to-end numerical integration benchmark",
    )
    synthetic_parser.add_argument("--seed", type=int, default=0)
    synthetic_parser.add_argument("--output", "-o", help="atomically write JSON here")
    synthetic_parser.set_defaults(handler=_synthetic_check)

    materialize_parser = subparsers.add_parser(
        "controlled-materialize",
        help="materialize pinned Phase-1 candidates, geometry, oracle labels, and features",
    )
    materialize_parser.add_argument("config")
    materialize_parser.add_argument("artifact_dir")
    materialize_parser.add_argument("--seed", type=int, required=True)
    materialize_parser.add_argument("--device", default="cuda")
    materialize_parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face network access (formal HPC jobs remain offline by default)",
    )
    materialize_parser.set_defaults(handler=_controlled_materialize)

    compare_parser = subparsers.add_parser(
        "controlled-compare",
        help="run paired fixed-step BT/SRM+ training over an integrity-checked artifact",
    )
    compare_parser.add_argument("config")
    compare_parser.add_argument("artifact_dir")
    compare_parser.add_argument("output")
    compare_parser.add_argument("--seed", type=int, required=True)
    compare_parser.add_argument("--device", default="cpu")
    compare_parser.add_argument(
        "--run-manifest",
        required=True,
        help="selected-seed smart-reward-run/v1 manifest to bind into the result",
    )
    compare_parser.set_defaults(handler=_controlled_compare)

    rollout_parser = subparsers.add_parser(
        "controlled-rollout",
        help="match BT/SRM policy updates to measured KL and run paired oracle rollouts",
    )
    rollout_parser.add_argument("config")
    rollout_parser.add_argument("artifact_dir")
    rollout_parser.add_argument("comparison")
    rollout_parser.add_argument("output")
    rollout_parser.add_argument("--seed", type=int, required=True)
    rollout_parser.add_argument("--device", default="cuda")
    rollout_parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face network access (formal HPC jobs remain offline by default)",
    )
    rollout_parser.set_defaults(handler=_controlled_rollout)

    aggregate_parser = subparsers.add_parser(
        "aggregate-results",
        help="aggregate all declared paired seeds with a deterministic bootstrap CI",
    )
    aggregate_parser.add_argument("config")
    aggregate_parser.add_argument("output")
    aggregate_parser.add_argument("results", nargs="+")
    aggregate_parser.add_argument(
        "--rollouts",
        nargs="+",
        required=True,
        help="matched-kl-rollout/v1 JSON files, one for every declared seed",
    )
    aggregate_parser.set_defaults(handler=_aggregate_results)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command, returning zero on success and two on invalid input."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    memory_destination = os.environ.get("SRM_MEMORY_REPORT")
    torch: ModuleType | None = None
    status = "ok"
    try:
        torch = _start_cuda_memory_tracking() if memory_destination else None
        exit_code = int(arguments.handler(arguments))
        status = "ok" if exit_code == 0 else "error"
    except (
        ConfigError,
        ImportError,
        RuntimeError,
        SchemaError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        status = "error"
        print(f"error: {error}", file=sys.stderr)
        exit_code = 2
    if memory_destination and torch is not None:
        try:
            _write_cuda_memory_report(
                torch,
                memory_destination,
                command=str(arguments.command),
                status=status,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            print(f"error: failed to write CUDA memory report: {error}", file=sys.stderr)
            return 2
    return exit_code


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess entry points.
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
