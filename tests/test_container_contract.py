from __future__ import annotations

import re
from pathlib import Path

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
