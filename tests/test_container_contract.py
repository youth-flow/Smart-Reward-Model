from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from smart_reward.config import config_hash, load_config

ROOT = Path(__file__).parents[1]
BASE_DIGEST = "sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3"


def _locked_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in (
        (ROOT / "containers" / "requirements-hpc4.lock").read_text(encoding="utf-8").splitlines()
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", line), line
        name, version = line.split("==", maxsplit=1)
        key = name.lower().replace("_", "-")
        assert key not in versions
        versions[key] = version
    return versions


def test_hpc4_lock_has_exact_research_critical_versions() -> None:
    versions = _locked_versions()

    assert versions["transformers"] == "4.52.3"
    assert versions["peft"] == "0.15.2"
    assert versions["accelerate"] == "1.7.0"
    assert versions["datasets"] == "3.6.0"
    assert versions["huggingface-hub"] == "0.31.4"
    assert versions["tokenizers"] == "0.21.1"
    assert versions["numpy"] == "1.26.4"
    assert versions["pyarrow"] == "17.0.0"
    assert versions["fsspec"] == "2025.3.0"
    assert "torch" not in versions


def test_definition_locks_base_and_forbids_dependency_resolution() -> None:
    definition = (ROOT / "containers" / "prorm-hpc4.def").read_text(encoding="utf-8")

    assert BASE_DIGEST in definition
    assert f"From: docker.io/pytorch/pytorch@{BASE_DIGEST}" in definition
    assert "--only-binary=:all: --no-deps" in definition
    assert "--no-deps --no-build-isolation /opt/prorm" in definition
    assert "python -m pip check" in definition


def test_image_workflow_publishes_raw_sif_and_checks_public_access() -> None:
    workflow = (ROOT / ".github" / "workflows" / "build-hpc4-image.yml").read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n  packages: write" in workflow
    assert "apptainer build \\" in workflow
    assert "--reproducible" in workflow
    assert "apptainer push --authfile" in workflow
    assert "oras://${GHCR_PACKAGE}:${tag}" in workflow
    assert "Prove anonymous GHCR pull authorization" in workflow
    assert "sif_sha256" in workflow


def test_tracked_config_identities_match_exact_bytes_and_semantics() -> None:
    identity_path = ROOT / "configs" / "identities.json"
    payload = json.loads(identity_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "prorm-config-identities/v1"
    assert set(payload["configs"]) == {"configs/main.yaml", "configs/smoke.yaml"}
    for relative, entry in payload["configs"].items():
        path = ROOT / relative
        config = load_config(path)
        run = config["run"]
        expected_seed_count = 1 if "seed" in run else len(run["seeds"])
        assert entry == {
            "config_hash": config_hash(config),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed_count": expected_seed_count,
        }


def test_hpc4_staging_and_submission_do_not_execute_images_on_login() -> None:
    stage_submit = (ROOT / "scripts" / "hpc4" / "submit_hf_stage.sh").read_text(encoding="utf-8")
    controlled_submit = (ROOT / "scripts" / "hpc4" / "submit_controlled.sh").read_text(
        encoding="utf-8"
    )
    stage_job = (ROOT / "scripts" / "hpc4" / "hf_stage.sbatch").read_text(encoding="utf-8")
    controlled_job = (ROOT / "scripts" / "hpc4" / "controlled.sbatch").read_text(encoding="utf-8")

    assert "apptainer exec" not in stage_submit
    assert "apptainer exec" not in controlled_submit
    for submit_script in (stage_submit, controlled_submit):
        assert 'identity_relative="configs/identities.json"' in submit_script
        assert "python3 -I -S -" in submit_script
        assert '"cat-file", "blob"' in submit_script
        assert '--export="ALL,' not in submit_script
    assert "apptainer exec --cleanenv" in stage_job
    assert "--no-mount home,cwd" in stage_job
    assert "PRORM_CONFIG_HASH" in stage_job
    assert "stage result inventory_sha256 is invalid" in stage_job
    assert "another Hugging Face staging job is writing the shared cache" in stage_job
    assert "--no-mount home,cwd" in controlled_job
    assert "PRORM_IMAGE escaped PRORM_PROJECT_ROOT" in controlled_job
    assert "PRORM_HF_CACHE escaped PRORM_PROJECT_ROOT" in controlled_job
