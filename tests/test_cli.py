from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import smart_reward.cli as cli_module
from smart_reward.cli import main
from smart_reward.config import config_hash, load_config
from smart_reward.data import TrainingEdgeRecord, save_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_formal_execution_is_explicit_not_implied_by_local_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("SLURM_JOB_ID", "SRM_GIT_COMMIT", "SRM_IMAGE_SHA256"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert cli_module._formal_execution_requested() is False

    monkeypatch.setenv("SLURM_JOB_ID", "230642")
    assert cli_module._formal_execution_requested() is True


def test_config_check_success(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["config-check", str(ROOT / "configs" / "smoke.yaml")])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert len(payload["config_hash"]) == 64


def test_config_check_failure_has_nonzero_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("unknown: true\n", encoding="utf-8")

    assert main(["config-check", str(invalid)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err


def test_data_check_validates_training_edge_jsonl(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "training.jsonl"
    record = TrainingEdgeRecord(
        edge_id="edge-0",
        prompt_id="prompt-0",
        left_id="candidate-0",
        right_id="candidate-1",
        raw_labels=(1,),
        num_annotations=1,
        left_wins=1,
        h=1.0,
    )
    save_jsonl(path, [record])

    assert main(["data-check", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "annotations": 1,
        "edges": 1,
        "path": str(path),
        "prompts": 1,
        "schema_version": "training-edge/v1",
        "status": "ok",
    }


def test_data_check_rejects_wrong_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"chosen":"leak"}\n', encoding="utf-8")

    assert main(["data-check", str(path)]) == 2
    assert "chosen/rejected" in capsys.readouterr().err


def test_env_report_atomically_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "run-manifest.json"
    real_builder = cli_module.build_run_manifest

    def controlled_builder(
        config: object,
        *,
        repo_path: object,
        selected_seed: int | None,
    ) -> object:
        return real_builder(
            config,
            repo_path=repo_path,
            environ={"SLURM_JOB_ID": "42"},
            selected_seed=selected_seed,
        )

    monkeypatch.setattr(cli_module, "build_run_manifest", controlled_builder)

    exit_code = main(
        [
            "env-report",
            str(ROOT / "configs" / "smoke.yaml"),
            "--repo-root",
            str(ROOT),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    announcement = json.loads(capsys.readouterr().out)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert announcement["status"] == "ok"
    assert announcement["config_hash"] == manifest["config_hash"]
    assert manifest["selected_seed"] == 20260722
    assert manifest["slurm"] == {"SLURM_JOB_ID": "42"}


def test_synthetic_check_writes_benchmark_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "nested" / "synthetic.json"

    assert main(["synthetic-check", "--seed", "7", "--output", str(output)]) == 0

    announcement = json.loads(capsys.readouterr().out)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert announcement["status"] == "ok"
    assert payload["status"] == "ok"
    assert payload["benchmark_only"] is True
    assert payload["seed"] == 7
    assert set(payload) >= {"bt", "srm", "srm_pcg"}


def test_aggregate_results_requires_and_uses_all_declared_paired_seeds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seeds = [20260722, 20260723, 20260724, 20260725, 20260726]
    paths: list[str] = []
    rollout_paths: list[str] = []
    digest = config_hash(load_config(ROOT / "configs" / "main.yaml"))
    for index, seed in enumerate(seeds):
        seed_dir = tmp_path / str(seed)
        seed_dir.mkdir()
        path = seed_dir / "comparison.json"
        manifest_path = seed_dir / "run-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "smart-reward-run/v1",
                    "config_hash": digest,
                    "selected_seed": seed,
                    "git": {"commit": "a" * 40, "dirty": False},
                    "slurm": {
                        "SRM_IMAGE_SHA256": "b" * 64,
                        "SRM_GIT_COMMIT": "a" * 40,
                        "SLURM_JOB_ACCOUNT": "sigroup",
                        "SLURM_JOB_PARTITION": "gpu-l20",
                    },
                    "torch": {
                        "cuda_available": True,
                        "gpu_count": 1,
                        "gpus": [{"name": "NVIDIA L20"}],
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        environment_identity = {
            "formal": True,
            "git_commit": "a" * 40,
            "image_sha256": "b" * 64,
            "account": "sigroup",
            "partition": "gpu-l20",
            "gpu_models": ["NVIDIA L20"],
        }

        def learner(offset: float) -> dict[str, object]:
            return {
                "final_pcg": {"converged": True},
                "test": {
                    "local_regret": 1.0 + offset,
                    "squared_fisher_error": 2.0 + offset,
                    "fisher_cosine": 0.5 + 0.01 * offset,
                    "pairwise_accuracy": 0.6 + 0.01 * offset,
                },
            }

        result = {
            "bt_mle": learner(float(index)),
            "srm_plus": learner(float(index) - 0.1),
        }
        payload = {
            "schema_version": "controlled-comparison/v1",
            "config_hash": digest,
            "seed": seed,
            "artifact_metadata_sha256": "d" * 64,
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": manifest_sha,
            "environment_identity": environment_identity,
            "damping_runs": [
                {
                    "damping_multiplier": 1.0,
                    "result": result,
                },
                {"damping_multiplier": 0.1, "result": result},
                {"damping_multiplier": 10.0, "result": result},
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(str(path))
        comparison_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        rollouts_jsonl = seed_dir / "updated_rollouts.jsonl"
        rollouts_jsonl.write_text('{"safe":true}\n', encoding="utf-8")
        rollouts_sha = hashlib.sha256(rollouts_jsonl.read_bytes()).hexdigest()
        rollout_path = seed_dir / "rollout.json"

        def rollout_learner(improvement: float) -> dict[str, object]:
            return {
                "direction": {"pcg": {"converged": True}},
                "measured_kl_update": {
                    "converged": True,
                    "applied": True,
                    "target_kl": 0.01,
                    "applied_measured_kl": 0.01,
                },
                "paired_improvement_over_zero_b_reference": {
                    "schema_version": "oracle-rollout-improvement/v1",
                    "num_pairs": 2,
                    "mean_difference": improvement,
                    "significance_claimed": False,
                },
            }

        rollout_path.write_text(
            json.dumps(
                {
                    "schema_version": "matched-kl-rollout/v1",
                    "config_hash": digest,
                    "seed": seed,
                    "artifact_metadata_sha256": "d" * 64,
                    "comparison_sha256": comparison_sha,
                    "run_manifest_sha256": manifest_sha,
                    "environment_identity": environment_identity,
                    "updated_rollouts_sha256": rollouts_sha,
                    "updated_rollouts_jsonl": str(rollouts_jsonl),
                    "test_reference": {
                        "source": "zero_b_common_random_number_rollout",
                        "num_prompts": 2,
                    },
                    "learners": {
                        "bt_mle": rollout_learner(0.05 + index * 0.01),
                        "srm_plus": rollout_learner(0.15 + index * 0.01),
                    },
                    "train_oracle_values_accessed": False,
                    "raw_oracle_values_serialized": False,
                }
            ),
            encoding="utf-8",
        )
        rollout_paths.append(str(rollout_path))
    output = tmp_path / "aggregate.json"

    assert (
        main(
            [
                "aggregate-results",
                str(ROOT / "configs" / "main.yaml"),
                str(output),
                *paths,
                "--rollouts",
                *rollout_paths,
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["num_seeds"] == 5
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    assert aggregate["num_seeds"] == 5
    assert aggregate["metrics"]["test_local_regret"]["paired_mean"] == pytest.approx(-0.1)
    assert aggregate["metrics"]["test_rollout_improvement"]["paired_mean"] == (pytest.approx(0.1))
    assert aggregate["config_hash"] == digest
    assert aggregate["environment_identity"]["gpu_models"] == ["NVIDIA L20"]
    assert len(aggregate["sources"]) == 5
    assert len(aggregate["damping_evidence"]) == 3
    assert aggregate["pre_registered_evidence"]["status"] == "not_passed"


def test_damping_failure_is_preserved_as_failed_evidence() -> None:
    seeds = {1, 2, 3, 4, 5}

    def ok(seed: int) -> dict[str, object]:
        return {
            "status": "ok",
            "bt_local_regret": 1.0 + seed,
            "srm_local_regret": 0.5 + seed,
            "pcg_converged": True,
        }

    evidence = {
        1.0: {seed: ok(seed) for seed in seeds},
        0.1: {
            **{seed: ok(seed) for seed in seeds if seed != 3},
            3: {
                "status": "failed",
                "failure_type": "pcg_nonconvergence",
                "message": "residual too large",
            },
        },
    }
    rows, all_pcg, nonreversal = cli_module._aggregate_damping_evidence(
        evidence,
        declared_seeds=seeds,
        bootstrap_seed=7,
        bootstrap_resamples=100,
    )

    failed = next(row for row in rows if row["damping_multiplier"] == 0.1)
    assert failed["status"] == "incomplete"
    assert failed["failures"][0]["seed"] == 3
    assert all_pcg is False
    assert nonreversal is False

    tied = {
        0.1: {
            seed: {
                "status": "ok",
                "bt_local_regret": float(seed),
                "srm_local_regret": float(seed),
                "pcg_converged": True,
            }
            for seed in seeds
        }
    }
    tied_rows, tied_pcg, tied_nonreversal = cli_module._aggregate_damping_evidence(
        tied,
        declared_seeds=seeds,
        bootstrap_seed=7,
        bootstrap_resamples=100,
    )
    assert tied_rows[0]["local_regret_nonreversal"] is False
    assert tied_pcg is True
    assert tied_nonreversal is False
