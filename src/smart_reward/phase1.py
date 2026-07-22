"""Phase-1 on-policy feature materialization and leakage-safe assembly.

The tensor-only :func:`assemble_controlled_experiment` is the authoritative
join boundary.  Prompt order defines the first tensor axis, all four candidate
nodes are retained for the Fisher geometry, and only candidate ``0 - 1`` is an
annotated edge.  Oracle calibration is fit exclusively on training nodes.

The Hugging Face entry point is intentionally lazy and fail closed.  It loads
only immutable revisions, defaults to the local cache, generates and scores
with one in-memory zero-B policy, then releases that policy before loading the
oracle.  The resulting artifact contains no training true-reward tensor.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .annotations import sample_geometric_repeated_labels
from .config import config_hash, validate_config
from .data import (
    DEFAULT_CONTINUATION_PROBABILITY,
    CandidateNode,
    EvaluationEdgeRecord,
    TrainingEdgeRecord,
    repeated_labels_to_h,
    save_jsonl,
)
from .experiment import (
    ControlledFeatureExperiment,
    EvaluationTensorData,
    TrainingTensorData,
)
from .hf import (
    assert_noop_logits,
    configure_fixed_a_lora,
    generate_exact_candidates,
    pool_final_response_hidden_state,
    score_exact_candidates,
    score_oracle_chats,
)
from .oracle import (
    RobustOracleTransform,
    btl_probabilities,
    fit_robust_oracle_transform,
)
from .prompts import PromptRecord, prepare_multipref_prompts, save_prompt_jsonl
from .scores import per_sample_scores
from .seeding import SeedBundle, derive_seed

_SPLITS = ("train", "validation", "test")
_NUM_CANDIDATES = 4
_ASSEMBLY_SCHEMA = "phase1-assembly/v1"
_MATERIALIZATION_SCHEMA = "phase1-materialization/v1"
_LOWER_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class Phase1Assembly:
    """Complete tensor/edge result of the pure Phase-1 join."""

    experiment: ControlledFeatureExperiment
    training_edges: tuple[TrainingEdgeRecord, ...]
    validation_edges: tuple[EvaluationEdgeRecord, ...]
    test_edges: tuple[EvaluationEdgeRecord, ...]
    oracle_transform: RobustOracleTransform
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, ControlledFeatureExperiment):
            raise TypeError("experiment must be ControlledFeatureExperiment")
        if not all(isinstance(edge, TrainingEdgeRecord) for edge in self.training_edges):
            raise TypeError("training_edges must contain TrainingEdgeRecord objects")
        for name, edges in (
            ("validation_edges", self.validation_edges),
            ("test_edges", self.test_edges),
        ):
            if not all(isinstance(edge, EvaluationEdgeRecord) for edge in edges):
                raise TypeError(f"{name} must contain EvaluationEdgeRecord objects")
        if not isinstance(self.oracle_transform, RobustOracleTransform):
            raise TypeError("oracle_transform must be RobustOracleTransform")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("evidence must be a mapping")


@dataclass(frozen=True)
class Phase1Materialization:
    """Materialized candidate graph plus its integrity-audited tensor assembly."""

    assembly: Phase1Assembly
    candidates: tuple[CandidateNode, ...]
    artifact_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, Phase1Assembly):
            raise TypeError("assembly must be Phase1Assembly")
        if not self.candidates or not all(
            isinstance(candidate, CandidateNode) for candidate in self.candidates
        ):
            raise TypeError("candidates must be a non-empty CandidateNode tuple")
        if (
            len(self.candidates)
            != (
                self.assembly.experiment.train.num_prompts
                + self.assembly.experiment.validation.num_prompts
                + self.assembly.experiment.test.num_prompts
            )
            * _NUM_CANDIDATES
        ):
            raise ValueError("candidate count does not match the assembled prompt graph")
        if not isinstance(self.artifact_directory, Path):
            raise TypeError("artifact_directory must be a pathlib.Path")


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    return seed


def _candidate_id(prompt_id: str, index: int) -> str:
    return f"{prompt_id}::candidate::{index}"


def _edge_id(prompt_id: str) -> str:
    return f"{prompt_id}::edge::0-1"


def _validate_assembler_inputs(
    prompt_records: Sequence[PromptRecord],
    policy_scores: torch.Tensor,
    reward_features: torch.Tensor,
    raw_oracle_scores: torch.Tensor,
) -> tuple[tuple[PromptRecord, ...], int, int, int]:
    if isinstance(prompt_records, (str, bytes)) or not isinstance(prompt_records, Sequence):
        raise TypeError("prompt_records must be a sequence of PromptRecord objects")
    records = tuple(prompt_records)
    if not records or not all(isinstance(record, PromptRecord) for record in records):
        raise TypeError("prompt_records must contain at least one PromptRecord")
    prompt_ids = [record.prompt_id for record in records]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt_records must have unique prompt_id values")
    split_counts = {split: 0 for split in _SPLITS}
    for record in records:
        split_counts[record.split] += 1
    empty = [split for split, count in split_counts.items() if count == 0]
    if empty:
        raise ValueError(f"every split must be non-empty; empty={empty!r}")

    for name, tensor in (
        ("policy_scores", policy_scores),
        ("reward_features", reward_features),
        ("raw_oracle_scores", raw_oracle_scores),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if tensor.requires_grad:
            raise ValueError(f"{name} must be detached")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")

    if policy_scores.ndim != 3:
        raise ValueError("policy_scores must have shape (P, 4, D)")
    num_prompts, num_candidates, policy_dimension = policy_scores.shape
    if num_prompts != len(records):
        raise ValueError(
            "prompt_records order is the tensor join key: len(prompt_records) must equal P"
        )
    if num_candidates != _NUM_CANDIDATES:
        raise ValueError("the controlled Phase-1 graph requires exactly four candidates")
    if policy_dimension < 1:
        raise ValueError("policy score dimension must be positive")
    if reward_features.ndim != 3 or reward_features.shape[:2] != (
        num_prompts,
        _NUM_CANDIDATES,
    ):
        raise ValueError("reward_features must have shape (P, 4, H)")
    if reward_features.shape[2] < 1:
        raise ValueError("reward feature dimension must be positive")
    if raw_oracle_scores.shape != (num_prompts, _NUM_CANDIDATES):
        raise ValueError("raw_oracle_scores must have shape (P, 4)")
    for tensor in (reward_features, raw_oracle_scores):
        if tensor.dtype != policy_scores.dtype or tensor.device != policy_scores.device:
            raise ValueError("all assembler tensors must share dtype and device")
    return records, num_prompts, policy_dimension, reward_features.shape[2]


def _local_annotation_batch(
    probabilities: torch.Tensor,
    *,
    seed: int,
    gamma: float,
) -> tuple[list[tuple[int, ...]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample on CPU with a private generator, independent of global RNG state."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batch = sample_geometric_repeated_labels(
        probabilities.detach().to(device="cpu", dtype=torch.float64),
        gamma=gamma,
        generator=generator,
    )
    labels_by_edge: list[tuple[int, ...]] = []
    offset = 0
    for count in batch.counts.reshape(-1).tolist():
        labels = tuple(int(value) for value in batch.labels[offset : offset + count].tolist())
        labels_by_edge.append(labels)
        offset += count
    if offset != batch.labels.numel():
        raise RuntimeError("internal repeated-label offset mismatch")
    wins = batch.wins.reshape(-1)
    totals = batch.counts.reshape(-1)
    h = torch.tensor(
        [repeated_labels_to_h(labels, gamma) for labels in labels_by_edge],
        dtype=torch.float64,
    )
    return labels_by_edge, wins, totals, h


def assemble_controlled_experiment(
    prompt_records: Sequence[PromptRecord],
    policy_scores: torch.Tensor,
    reward_features: torch.Tensor,
    raw_oracle_scores: torch.Tensor,
    *,
    seed: int,
    gamma: float = DEFAULT_CONTINUATION_PROBABILITY,
) -> Phase1Assembly:
    """Join ordered node tensors into the locked controlled experiment.

    ``prompt_records[i]`` is the explicit join key for tensor row ``i``.  The
    robust oracle transform is fit on all and only the ``train`` rows.  Raw or
    transformed rewards are never supplied to :class:`TrainingTensorData`.
    A private named CPU generator samples iid BTL labels and does not mutate
    Python, CPU-Torch, or CUDA global RNG state.
    """

    validated_seed = _validate_seed(seed)
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value != DEFAULT_CONTINUATION_PROBABILITY:
        raise ValueError(
            "gamma must equal the locked TrainingEdgeRecord value "
            f"{DEFAULT_CONTINUATION_PROBABILITY}"
        )
    records, _, _, _ = _validate_assembler_inputs(
        prompt_records,
        policy_scores,
        reward_features,
        raw_oracle_scores,
    )

    split_indices = {
        split: [index for index, record in enumerate(records) if record.split == split]
        for split in _SPLITS
    }
    train_index = torch.tensor(
        split_indices["train"], dtype=torch.int64, device=raw_oracle_scores.device
    )
    transform = fit_robust_oracle_transform(
        raw_oracle_scores.index_select(0, train_index).reshape(-1)
    )
    true_rewards = transform(raw_oracle_scores)
    margins = true_rewards[:, 0] - true_rewards[:, 1]
    probabilities = btl_probabilities(margins)

    seeds = SeedBundle.from_base_seed(validated_seed)
    labels_by_edge: list[tuple[int, ...]] = [tuple() for _ in records]
    all_wins_cpu = torch.empty(len(records), dtype=torch.int64)
    all_totals_cpu = torch.empty(len(records), dtype=torch.int64)
    all_h_cpu = torch.empty(len(records), dtype=torch.float64)
    annotation_split_seeds: dict[str, int] = {}
    for split in _SPLITS:
        indices = split_indices[split]
        index = torch.tensor(indices, dtype=torch.int64, device=probabilities.device)
        split_seed = derive_seed(seeds.annotations, f"annotations:{split}")
        annotation_split_seeds[split] = split_seed
        split_labels, split_wins, split_totals, split_h = _local_annotation_batch(
            probabilities.index_select(0, index),
            seed=split_seed,
            gamma=gamma_value,
        )
        for local_index, global_index in enumerate(indices):
            labels_by_edge[global_index] = split_labels[local_index]
            all_wins_cpu[global_index] = split_wins[local_index]
            all_totals_cpu[global_index] = split_totals[local_index]
            all_h_cpu[global_index] = split_h[local_index]

    training_edges: list[TrainingEdgeRecord] = []
    validation_edges: list[EvaluationEdgeRecord] = []
    test_edges: list[EvaluationEdgeRecord] = []
    for index, record in enumerate(records):
        common: dict[str, Any] = {
            "edge_id": _edge_id(record.prompt_id),
            "prompt_id": record.prompt_id,
            "left_id": _candidate_id(record.prompt_id, 0),
            "right_id": _candidate_id(record.prompt_id, 1),
            "raw_labels": labels_by_edge[index],
            "num_annotations": int(all_totals_cpu[index].item()),
            "left_wins": int(all_wins_cpu[index].item()),
            "h": float(all_h_cpu[index].item()),
        }
        if record.split == "train":
            training_edges.append(TrainingEdgeRecord(**common))
        else:
            edge = EvaluationEdgeRecord(
                **common,
                true_margin=float(margins[index].item()),
            )
            if record.split == "validation":
                validation_edges.append(edge)
            else:
                test_edges.append(edge)

    def frozen_index(split: str) -> torch.Tensor:
        return torch.tensor(split_indices[split], dtype=torch.int64, device=policy_scores.device)

    train_rows = frozen_index("train")
    validation_rows = frozen_index("validation")
    test_rows = frozen_index("test")
    train_edge_positions = [
        index for index, record in enumerate(records) if record.split == "train"
    ]
    train_edge_index_cpu = torch.tensor(train_edge_positions, dtype=torch.int64)

    # Clone every slice: the immutable data object cannot be changed by later
    # mutation of a caller-owned materialization buffer.
    train = TrainingTensorData(
        prompt_ids=tuple(record.prompt_id for record in records if record.split == "train"),
        policy_scores=policy_scores.index_select(0, train_rows).detach().clone(),
        reward_features=reward_features.index_select(0, train_rows).detach().clone(),
        h=all_h_cpu.index_select(0, train_edge_index_cpu)
        .to(device=policy_scores.device, dtype=policy_scores.dtype)
        .clone(),
        left_wins=all_wins_cpu.index_select(0, train_edge_index_cpu)
        .to(device=policy_scores.device)
        .clone(),
        num_annotations=all_totals_cpu.index_select(0, train_edge_index_cpu)
        .to(device=policy_scores.device)
        .clone(),
    )

    def evaluation(split: str, rows: torch.Tensor) -> EvaluationTensorData:
        return EvaluationTensorData(
            prompt_ids=tuple(record.prompt_id for record in records if record.split == split),
            policy_scores=policy_scores.index_select(0, rows).detach().clone(),
            reward_features=reward_features.index_select(0, rows).detach().clone(),
            true_rewards=true_rewards.index_select(0, rows).detach().clone(),
        )

    experiment = ControlledFeatureExperiment(
        train=train,
        validation=evaluation("validation", validation_rows),
        test=evaluation("test", test_rows),
    )
    evidence: dict[str, Any] = {
        "schema": _ASSEMBLY_SCHEMA,
        "seed": validated_seed,
        "named_seeds": seeds.to_dict(),
        "annotation_split_seeds": annotation_split_seeds,
        "gamma": gamma_value,
        "num_candidates": _NUM_CANDIDATES,
        "split_sizes": {split: len(split_indices[split]) for split in _SPLITS},
        "oracle_transform": {"b": transform.b, "tau": transform.tau},
        "oracle_fit_split": "train",
        "edge_orientation": {"left_candidate": 0, "right_candidate": 1},
    }
    return Phase1Assembly(
        experiment=experiment,
        training_edges=tuple(training_edges),
        validation_edges=tuple(validation_edges),
        test_edges=tuple(test_edges),
        oracle_transform=transform,
        evidence=evidence,
    )


def _require_module(name: str, *, extra: str = "llm") -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            f"Phase-1 materialization requires optional dependency {name!r}; "
            f"install smart-reward-model[{extra}]"
        ) from error


def _local_snapshot_error(
    kind: str,
    identifier: str,
    revision: str,
) -> RuntimeError:
    return RuntimeError(
        f"pinned {kind} snapshot is unavailable from the local cache: "
        f"{identifier}@{revision}. Pre-stage this exact revision on the HPC login node; "
        "formal compute-node runs must not download from the network."
    )


def _model_inputs(encoded: object, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(encoded, torch.Tensor):
        inputs = {"input_ids": encoded}
    elif isinstance(encoded, Mapping):
        inputs = {name: value for name, value in encoded.items() if isinstance(value, torch.Tensor)}
    else:
        raise TypeError("apply_chat_template must return a tensor or tensor mapping")
    if "input_ids" not in inputs:
        raise ValueError("chat template output must contain input_ids")
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return {name: value.to(device) for name, value in inputs.items()}


def _prompt_text(record: PromptRecord) -> str:
    if len(record.messages) != 1 or record.messages[0].role != "user":
        raise ValueError(
            "controlled MultiPref prompts must contain exactly one user message and no system role"
        )
    return record.messages[0].content


@contextmanager
def _fork_torch_seed(seed: int, device: torch.device) -> Iterator[None]:
    cuda_devices: list[int] = []
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        cuda_devices = [index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed(seed)
        yield


def _jsonl_sha256(records: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for record in records:
        payload = record.to_dict()
        line = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _producer_identity_from_environment() -> dict[str, str]:
    """Read only formal-run producer digests; never snapshot the environment."""

    result: dict[str, str] = {}
    for environment_name, evidence_name, lengths in (
        ("SRM_GIT_COMMIT", "git_commit", {40, 64}),
        ("SRM_IMAGE_SHA256", "image_sha256", {64}),
    ):
        value = os.environ.get(environment_name)
        if value is None:
            continue
        if len(value) not in lengths or any(character not in _LOWER_HEX for character in value):
            raise ValueError(f"{environment_name} must be a lowercase hexadecimal producer digest")
        result[evidence_name] = value
    return result


def _decode_response(tokenizer: object, token_ids: torch.Tensor) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("policy tokenizer must expose callable decode")
    return str(decode(token_ids.tolist(), skip_special_tokens=True))


def _load_prompts(
    datasets: Any,
    config: Mapping[str, Any],
    *,
    split_seed: int,
    local_files_only: bool,
) -> list[PromptRecord]:
    data_config = config["data"]
    run_config = config["run"]
    kwargs: dict[str, Any] = {
        "revision": data_config["prompt_revision"],
        "split": "train",
    }
    if local_files_only:
        download_config_type = getattr(datasets, "DownloadConfig", None)
        if download_config_type is None:
            raise RuntimeError("installed datasets package does not expose DownloadConfig")
        kwargs["download_config"] = download_config_type(local_files_only=True)
    try:
        rows = datasets.load_dataset(data_config["prompt_dataset"], **kwargs)
    except (OSError, FileNotFoundError, ConnectionError) as error:
        if local_files_only:
            raise _local_snapshot_error(
                "dataset",
                data_config["prompt_dataset"],
                data_config["prompt_revision"],
            ) from error
        raise
    return prepare_multipref_prompts(
        rows,
        split_sizes=run_config["split_sizes"],
        seed=split_seed,
    )


def _load_pretrained(
    factory: Any,
    identifier: str,
    revision: str,
    *,
    local_files_only: bool,
    kind: str,
    **kwargs: Any,
) -> Any:
    try:
        return factory.from_pretrained(
            identifier,
            revision=revision,
            local_files_only=local_files_only,
            **kwargs,
        )
    except (OSError, FileNotFoundError, ConnectionError) as error:
        if local_files_only:
            raise _local_snapshot_error(kind, identifier, revision) from error
        raise


def _preflight_artifact_target(destination: Path) -> None:
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"artifact target is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact directory: {destination}")


def materialize_phase1(
    config: Mapping[str, object],
    *,
    seed: int,
    artifact_dir: str | os.PathLike[str],
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> Phase1Materialization:
    """Materialize the pinned four-candidate Phase-1 experiment.

    This operation never overwrites an existing file.  ``local_files_only`` is
    true by default so a formal HPC run cannot silently change its inputs or
    spend allocation time downloading a new snapshot.
    """

    validated_seed = _validate_seed(seed)
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be bool")
    destination = Path(artifact_dir)
    _preflight_artifact_target(destination)
    normalized = validate_config(config)
    configured_seeds = (
        [normalized["run"]["seed"]] if "seed" in normalized["run"] else normalized["run"]["seeds"]
    )
    if validated_seed not in configured_seeds:
        raise ValueError("seed must be one of the explicitly configured run seeds")
    if normalized["data"]["num_candidates"] != _NUM_CANDIDATES:
        raise ValueError("Phase-1 materialization requires data.num_candidates == 4")
    if any(
        normalized[section]["dtype"] != "float32"
        for section in ("policy", "reward_model", "oracle")
    ):
        raise ValueError("Phase-1 policy, reward features, and oracle must all use float32")
    if (
        normalized["reward_model"]["model"] != normalized["policy"]["model"]
        or normalized["reward_model"]["revision"] != normalized["policy"]["revision"]
    ):
        raise ValueError(
            "reward_model model/revision must match policy exactly because frozen reward "
            "features are extracted from the zero-B policy forward"
        )

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    # Dependency checks are lazy and occur only after cheap configuration and
    # destination guards have succeeded.
    datasets = _require_module("datasets")
    transformers = _require_module("transformers")
    peft = _require_module("peft")
    _require_module("safetensors")
    from .artifacts import save_controlled_feature_artifact

    seeds = SeedBundle.from_base_seed(validated_seed)
    prompts = _load_prompts(
        datasets,
        normalized,
        split_seed=seeds.prompt_split,
        local_files_only=local_files_only,
    )

    policy_config = normalized["policy"]
    tokenizer = _load_pretrained(
        transformers.AutoTokenizer,
        policy_config["model"],
        policy_config["revision"],
        local_files_only=local_files_only,
        kind="policy tokenizer",
        use_fast=True,
    )
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("policy tokenizer must provide a non-empty chat_template")
    tokenizer.truncation_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with _fork_torch_seed(seeds.policy_lora_a, target_device):
        policy_model = _load_pretrained(
            transformers.AutoModelForCausalLM,
            policy_config["model"],
            policy_config["revision"],
            local_files_only=local_files_only,
            kind="policy model",
            torch_dtype=torch.float32,
        )
    policy_model.to(target_device)
    policy_model.eval()

    first_encoded = tokenizer.apply_chat_template(
        [message.to_dict() for message in prompts[0].messages],
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=policy_config["max_prompt_tokens"],
        return_tensors="pt",
        return_dict=True,
    )
    probe_inputs = _model_inputs(first_encoded, target_device)
    with torch.inference_mode():
        reference_logits = policy_model(**probe_inputs, use_cache=False).logits.detach().clone()

    lora_config = peft.LoraConfig(
        r=policy_config["lora_rank"],
        lora_alpha=policy_config["lora_alpha"],
        lora_dropout=policy_config["lora_dropout"],
        target_modules=list(policy_config["lora_modules"]),
        layers_to_transform=list(policy_config["lora_layers"]),
        bias="none",
        init_lora_weights=True,
        task_type="CAUSAL_LM",
    )
    with _fork_torch_seed(seeds.policy_lora_a, target_device):
        setup = configure_fixed_a_lora(policy_model, lora_config)
    policy_model = setup.model
    policy_model.eval()
    with torch.inference_mode():
        adapted_logits = policy_model(**probe_inputs, use_cache=False).logits
    zero_b_error = assert_noop_logits(reference_logits, adapted_logits)
    del reference_logits, adapted_logits

    candidate_nodes: list[CandidateNode] = []
    policy_score_rows: list[torch.Tensor] = []
    reward_feature_rows: list[torch.Tensor] = []
    generation_kwargs = {
        **dict(policy_config["sampling"]),
        "num_return_sequences": _NUM_CANDIDATES,
        "max_new_tokens": policy_config["max_response_tokens"],
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    with _fork_torch_seed(seeds.candidate_generation, target_device):
        for prompt in prompts:
            encoded = tokenizer.apply_chat_template(
                [message.to_dict() for message in prompt.messages],
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=policy_config["max_prompt_tokens"],
                return_tensors="pt",
                return_dict=True,
            )
            prompt_inputs = _model_inputs(encoded, target_device)
            candidates = generate_exact_candidates(
                policy_model,
                prompt_inputs["input_ids"],
                prompt_attention_mask=prompt_inputs["attention_mask"],
                generation_kwargs=generation_kwargs,
            )
            log_probabilities = score_exact_candidates(policy_model, candidates)
            scores = per_sample_scores(
                log_probabilities,
                setup.named_tangent_parameters(),
                layout=setup.layout,
            )
            if scores.shape[0] != _NUM_CANDIDATES:
                raise RuntimeError("policy did not return exactly four candidate scores")
            policy_score_rows.append(scores.to(device="cpu", dtype=torch.float32))

            with torch.inference_mode():
                hidden_output = policy_model(
                    input_ids=candidates.input_ids,
                    attention_mask=candidates.attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                features = pool_final_response_hidden_state(
                    hidden_output.hidden_states,
                    candidates.response_mask,
                )
            reward_feature_rows.append(features.detach().to(device="cpu", dtype=torch.float32))

            prompt_text = _prompt_text(prompt)
            for candidate_index in range(_NUM_CANDIDATES):
                active_response_ids = candidates.input_ids[candidate_index][
                    candidates.response_mask[candidate_index].bool()
                ]
                candidate_nodes.append(
                    CandidateNode(
                        prompt_id=prompt.prompt_id,
                        candidate_id=_candidate_id(prompt.prompt_id, candidate_index),
                        prompt=prompt_text,
                        response=_decode_response(tokenizer, active_response_ids),
                        token_ids=tuple(
                            int(value) for value in candidates.input_ids[candidate_index].tolist()
                        ),
                        response_mask=tuple(
                            int(value)
                            for value in candidates.response_mask[candidate_index].tolist()
                        ),
                        terminated_by_eos=bool(
                            candidates.terminated_by_eos[candidate_index].item()
                        ),
                        reached_max_length=bool(
                            candidates.reached_max_length[candidate_index].item()
                        ),
                    )
                )
            del log_probabilities, scores, hidden_output, features, candidates

    policy_scores = torch.stack(policy_score_rows, dim=0)
    reward_features = torch.stack(reward_feature_rows, dim=0)
    layout_metadata = setup.layout.to_metadata()
    a_state_sha256 = setup.a_state_sha256
    del setup, policy_model, policy_score_rows, reward_feature_rows, probe_inputs
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    oracle_config = normalized["oracle"]
    oracle_tokenizer = _load_pretrained(
        transformers.AutoTokenizer,
        oracle_config["model"],
        oracle_config["revision"],
        local_files_only=local_files_only,
        kind="oracle tokenizer",
        use_fast=True,
    )
    oracle_chat_template = getattr(oracle_tokenizer, "chat_template", None)
    if oracle_chat_template in (None, ""):
        raise ValueError("oracle tokenizer must provide a non-empty chat_template")
    oracle_chat_template_sha256 = hashlib.sha256(
        str(oracle_chat_template).encode("utf-8")
    ).hexdigest()
    oracle_model = _load_pretrained(
        transformers.AutoModelForSequenceClassification,
        oracle_config["model"],
        oracle_config["revision"],
        local_files_only=local_files_only,
        kind="oracle model",
        torch_dtype=torch.float32,
        num_labels=1,
    )
    oracle_model.to(target_device)
    oracle_model.eval()
    flat_prompts = [candidate.prompt for candidate in candidate_nodes]
    flat_responses = [candidate.response for candidate in candidate_nodes]
    oracle_batch_size = min(16, int(normalized["reward_model"]["microbatch_size"]))
    raw_batches: list[torch.Tensor] = []
    for start in range(0, len(candidate_nodes), oracle_batch_size):
        stop = min(start + oracle_batch_size, len(candidate_nodes))
        raw_batches.append(
            score_oracle_chats(
                oracle_model,
                oracle_tokenizer,
                flat_prompts[start:stop],
                flat_responses[start:stop],
                device=target_device,
            ).to(device="cpu", dtype=torch.float32)
        )
    raw_oracle_scores = torch.cat(raw_batches).reshape(len(prompts), _NUM_CANDIDATES)
    del oracle_model, oracle_tokenizer, raw_batches
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    assembly = assemble_controlled_experiment(
        prompts,
        policy_scores,
        reward_features,
        raw_oracle_scores,
        seed=validated_seed,
        gamma=float(normalized["annotations"]["gamma"]),
    )

    evaluation_edges = (*assembly.validation_edges, *assembly.test_edges)
    json_hashes = {
        "prompts.jsonl": _jsonl_sha256(prompts),
        "candidates.jsonl": _jsonl_sha256(candidate_nodes),
        "training_edges.jsonl": _jsonl_sha256(assembly.training_edges),
        "evaluation_edges.jsonl": _jsonl_sha256(evaluation_edges),
    }
    full_evidence = {
        **dict(assembly.evidence),
        "schema": _MATERIALIZATION_SCHEMA,
        "config_sha256": config_hash(normalized),
        "policy_a_sha256": a_state_sha256,
        "policy_layout": layout_metadata,
        "policy_zero_b_max_absolute_error": zero_b_error,
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")
        ).hexdigest(),
        "oracle_chat_template_sha256": oracle_chat_template_sha256,
        "jsonl_sha256": json_hashes,
        "revisions": {
            "prompt_dataset": normalized["data"]["prompt_revision"],
            "policy_model": policy_config["revision"],
            "reward_feature_model": normalized["reward_model"]["revision"],
            "oracle_model": oracle_config["revision"],
        },
        "local_files_only": local_files_only,
        "producer": _producer_identity_from_environment(),
    }
    assembly = replace(assembly, evidence=full_evidence)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.phase1-", dir=destination.parent))
    try:
        save_controlled_feature_artifact(
            assembly.experiment,
            staging,
            config_hash=config_hash(normalized),
            seed=validated_seed,
            evidence=full_evidence,
            overwrite=False,
        )
        save_prompt_jsonl(staging / "prompts.jsonl", prompts)
        save_jsonl(staging / "candidates.jsonl", candidate_nodes)
        save_jsonl(staging / "training_edges.jsonl", assembly.training_edges)
        save_jsonl(staging / "evaluation_edges.jsonl", evaluation_edges)
        for filename, expected_digest in json_hashes.items():
            if _sha256_file(staging / filename) != expected_digest:
                raise RuntimeError(f"serialized digest mismatch for {filename}")
        if destination.exists():
            # Preflight proved this exact target directory was empty.
            destination.rmdir()
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return Phase1Materialization(
        assembly=assembly,
        candidates=tuple(candidate_nodes),
        artifact_directory=destination,
    )


__all__ = [
    "Phase1Assembly",
    "Phase1Materialization",
    "assemble_controlled_experiment",
    "materialize_phase1",
]
