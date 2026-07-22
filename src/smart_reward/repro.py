"""Reproducibility metadata for ProRM/ProRM+ experiments.

The manifest collector is intentionally offline and uses an explicit Slurm
environment allowlist.  It never serializes the process environment wholesale,
which prevents credentials such as Hugging Face or Weights & Biases tokens from
being copied into experiment artifacts.
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata as importlib_metadata
import json
import os
import platform as platform_module
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config_hash, validate_config
from .contracts import compatibility_value
from .seeding import SeedBundle

RUN_MANIFEST_SCHEMA_VERSION = "smart-reward-run/v1"

_SLURM_ENVIRONMENT_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_ACCOUNT",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NAME",
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_SUBMIT_DIR",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "PRORM_IMAGE",
    "PRORM_IMAGE_SHA256",
    "PRORM_GIT_COMMIT",
    "SRM_IMAGE",
    "SRM_IMAGE_SHA256",
    "SRM_GIT_COMMIT",
)

_PACKAGE_NAMES = (
    "accelerate",
    "datasets",
    "peft",
    "pyyaml",
    "safetensors",
    "torch",
    "transformers",
)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """JSON-compatible, immutable top-level run manifest."""

    schema_version: str
    created_at_utc: str
    config_hash: str
    normalized_config: dict[str, object]
    seed: int | list[int]
    selected_seed: int | None
    named_seeds: dict[str, dict[str, int]]
    git: dict[str, object]
    python: dict[str, object]
    platform: dict[str, object]
    torch: dict[str, object]
    revisions: dict[str, object]
    packages: dict[str, str | None]
    slurm: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Return an independent JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "config_hash": self.config_hash,
            "normalized_config": copy.deepcopy(self.normalized_config),
            "seed": copy.deepcopy(self.seed),
            "selected_seed": self.selected_seed,
            "named_seeds": copy.deepcopy(self.named_seeds),
            "git": copy.deepcopy(self.git),
            "python": copy.deepcopy(self.python),
            "platform": copy.deepcopy(self.platform),
            "torch": copy.deepcopy(self.torch),
            "revisions": copy.deepcopy(self.revisions),
            "packages": copy.deepcopy(self.packages),
            "slurm": copy.deepcopy(self.slurm),
        }


def _git_output(repo_path: Path, arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def collect_git_state(repo_path: str | os.PathLike[str] = ".") -> dict[str, object]:
    """Collect the current commit and tracked/untracked dirty state."""

    root = Path(repo_path).resolve()
    commit = _git_output(root, ["rev-parse", "--verify", "HEAD"])
    status = _git_output(root, ["status", "--porcelain", "--untracked-files=normal"])
    if commit is None or status is None:
        return {"commit": commit, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def collect_slurm_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return only reproducibility-relevant Slurm variables from ``environ``."""

    source = os.environ if environ is None else environ
    return {key: source[key] for key in _SLURM_ENVIRONMENT_KEYS if key in source}


def _safe_call(function: Any, default: Any = None) -> Any:
    try:
        return function()
    except Exception:  # Hardware probes are best-effort metadata collection.
        return default


def collect_torch_state() -> dict[str, object]:
    """Collect Torch, CUDA, cuDNN, and visible GPU metadata without model loading."""

    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError):
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "cudnn_version": None,
            "gpu_count": 0,
            "gpus": [],
        }

    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(_safe_call(cuda.is_available, False)) if cuda is not None else False
    gpu_count = int(_safe_call(cuda.device_count, 0) or 0) if cuda_available else 0
    gpus: list[dict[str, object]] = []
    for index in range(gpu_count):
        name = _safe_call(lambda index=index: cuda.get_device_name(index), None)
        properties = _safe_call(lambda index=index: cuda.get_device_properties(index), None)
        capability = _safe_call(lambda index=index: cuda.get_device_capability(index), None)
        gpu: dict[str, object] = {"index": index, "name": name}
        if properties is not None:
            total_memory = getattr(properties, "total_memory", None)
            if isinstance(total_memory, int):
                gpu["total_memory_bytes"] = total_memory
        if isinstance(capability, (tuple, list)) and len(capability) == 2:
            gpu["compute_capability"] = f"{capability[0]}.{capability[1]}"
        gpus.append(gpu)

    version_namespace = getattr(torch, "version", None)
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None) if backends is not None else None
    return {
        "installed": True,
        "version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": cuda_available,
        "cuda_version": getattr(version_namespace, "cuda", None),
        "cudnn_version": _safe_call(cudnn.version, None) if cudnn is not None else None,
        "gpu_count": gpu_count,
        "gpus": gpus,
    }


def collect_execution_identity(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Collect the current formal identity used for cross-stage matching."""

    source = os.environ if environ is None else environ
    torch_state = collect_torch_state()
    gpus = torch_state.get("gpus")
    gpu_names = (
        [gpu.get("name") for gpu in gpus if isinstance(gpu, dict)] if isinstance(gpus, list) else []
    )
    commit = compatibility_value(source, "PRORM_GIT_COMMIT", "SRM_GIT_COMMIT")
    image = compatibility_value(source, "PRORM_IMAGE_SHA256", "SRM_IMAGE_SHA256")
    account = source.get("SLURM_JOB_ACCOUNT")
    partition = source.get("SLURM_JOB_PARTITION")
    complete = (
        isinstance(commit, str)
        and len(commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in commit)
        and isinstance(image, str)
        and len(image) == 64
        and all(character in "0123456789abcdef" for character in image)
        and isinstance(partition, str)
        and bool(partition)
        and account == "sigroup"
        and torch_state.get("cuda_available") is True
        and torch_state.get("gpu_count") == 1
        and len(gpu_names) == 1
        and isinstance(gpu_names[0], str)
        and bool(gpu_names[0])
    )
    return {
        "formal": complete,
        "git_commit": commit if complete else None,
        "image_sha256": image if complete else None,
        "account": account if complete else None,
        "partition": partition if complete else None,
        "gpu_models": gpu_names if complete else [],
    }


def _collect_python_state() -> dict[str, object]:
    return {
        "version": platform_module.python_version(),
        "implementation": platform_module.python_implementation(),
        "version_info": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
    }


def _collect_platform_state() -> dict[str, object]:
    return {
        "system": platform_module.system(),
        "release": platform_module.release(),
        "machine": platform_module.machine(),
        "architecture": platform_module.architecture()[0],
    }


def _collect_revisions(config: Mapping[str, Any]) -> dict[str, object]:
    data = config["data"]
    policy = config["policy"]
    oracle = config["oracle"]
    reward_model = config["reward_model"]
    return {
        "prompt_dataset": {
            "id": data["prompt_dataset"],
            "revision": data["prompt_revision"],
        },
        "policy_model": {"id": policy["model"], "revision": policy["revision"]},
        "oracle_model": {"id": oracle["model"], "revision": oracle["revision"]},
        "reward_model": {
            "id": reward_model["model"],
            "revision": reward_model["revision"],
        },
    }


def _collect_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _utc_timestamp(now: datetime | None) -> str:
    timestamp = datetime.now(timezone.utc) if now is None else now
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("manifest timestamp must be timezone-aware")
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_run_manifest(
    config: Mapping[str, object],
    *,
    repo_path: str | os.PathLike[str] = ".",
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
    selected_seed: int | None = None,
) -> RunManifest:
    """Validate ``config`` and collect a complete offline run manifest."""

    normalized = validate_config(config)
    run = normalized["run"]
    seed: int | list[int] = int(run["seed"]) if "seed" in run else list(run["seeds"])
    seed_values = [seed] if isinstance(seed, int) else seed
    if selected_seed is None:
        resolved_selected_seed = seed if isinstance(seed, int) else None
    else:
        if (
            isinstance(selected_seed, bool)
            or not isinstance(selected_seed, int)
            or selected_seed not in seed_values
        ):
            raise ValueError("selected_seed must be one of the seeds declared by the config")
        resolved_selected_seed = selected_seed
    return RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        created_at_utc=_utc_timestamp(now),
        config_hash=config_hash(normalized),
        normalized_config=copy.deepcopy(normalized),
        seed=seed,
        selected_seed=resolved_selected_seed,
        named_seeds={
            str(value): SeedBundle.from_base_seed(value).to_dict() for value in seed_values
        },
        git=collect_git_state(repo_path),
        python=_collect_python_state(),
        platform=_collect_platform_state(),
        torch=collect_torch_state(),
        revisions=_collect_revisions(normalized),
        packages=_collect_package_versions(),
        slurm=collect_slurm_environment(environ),
    )


create_run_manifest = build_run_manifest


def atomic_write_json(
    path: str | os.PathLike[str],
    value: RunManifest | Mapping[str, object],
) -> None:
    """Atomically replace ``path`` with deterministic UTF-8 JSON."""

    destination = Path(path)
    if not destination.parent.exists():
        raise FileNotFoundError(f"destination directory does not exist: {destination.parent}")
    payload: Mapping[str, object]
    payload = value.to_dict() if isinstance(value, RunManifest) else value

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)


def write_run_manifest(
    path: str | os.PathLike[str],
    manifest: RunManifest,
) -> None:
    """Atomically write one :class:`RunManifest`."""

    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    atomic_write_json(path, manifest)


__all__ = [
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "atomic_write_json",
    "build_run_manifest",
    "collect_git_state",
    "collect_execution_identity",
    "collect_slurm_environment",
    "collect_torch_state",
    "create_run_manifest",
    "write_run_manifest",
]
