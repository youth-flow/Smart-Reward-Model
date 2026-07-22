"""Strict, orientation-aware data contracts for Smart Reward Model runs.

The training edge format deliberately stores an edge in its pre-declared
``(left_id, right_id)`` direction.  It never converts an observation to a
``(chosen, rejected)`` pair: doing so would silently make the score difference
``z``, the reward margin, and the repeated-label statistic disagree about
orientation.

All ``from_dict`` methods require an exact schema.  In particular,
evaluation-only oracle quantities are rejected before a training record is
constructed.  JSONL writes use a temporary file in the destination directory
and ``os.replace`` so an interrupted write cannot leave a partial dataset at
the requested path.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, ClassVar, TypeVar

Identifier = str | int

CANDIDATE_SCHEMA_VERSION = "candidate-node/v1"
REPEATED_EDGE_SCHEMA_VERSION = "repeated-edge/v1"
TRAINING_EDGE_SCHEMA_VERSION = "training-edge/v1"
EVALUATION_EDGE_SCHEMA_VERSION = "evaluation-edge/v1"
DEFAULT_CONTINUATION_PROBABILITY = 0.9


class SchemaError(ValueError):
    """Raised when serialized data does not satisfy an exact schema."""


class EvaluationLeakageError(SchemaError):
    """Raised when evaluation-only information enters a training schema."""


def _validate_identifier(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{name} must be a string or integer identifier")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must not be empty")
    if isinstance(value, int) and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    actual = set(value)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)!r}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)!r}")
        raise SchemaError(f"invalid {context} schema: {', '.join(details)}")


def _tuple_of_integers(value: object, name: str, *, binary: bool) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    result = tuple(value)
    for item in result:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} must contain integers, not {item!r}")
        if binary and item not in (0, 1):
            raise ValueError(f"{name} must contain only 0 and 1")
        if not binary and item < 0:
            raise ValueError(f"{name} must contain non-negative token ids")
    return result


def repeated_labels_to_h(
    raw_labels: Sequence[int],
    continuation_probability: float = DEFAULT_CONTINUATION_PROBABILITY,
) -> float:
    """Return the randomized-truncation U-statistic for one repeated edge.

    ``continuation_probability`` is the geometric survival probability
    ``P(N >= k + 1 | N >= k)``.  The default is the locked Phase-1 protocol
    value.  The recurrence is algebraically identical to the implementation
    in :mod:`smart_reward.annotations`, but this data-contract module has no
    dependency on PyTorch.
    """

    labels = _tuple_of_integers(raw_labels, "raw_labels", binary=True)
    if not labels:
        raise ValueError("raw_labels must contain at least one annotation")
    gamma = float(continuation_probability)
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("continuation_probability must be finite and lie in (0, 1]")

    total = len(labels)
    wins = sum(labels)
    losses = total - wins
    weighted_positive = 1.0
    weighted_negative = 1.0
    estimate = 0.0

    for order in range(1, total + 1):
        denominator = float(total - order + 1)
        continuation = 1.0 if order == 1 else gamma
        positive_ratio = max(wins - order + 1, 0) / denominator
        negative_ratio = max(losses - order + 1, 0) / denominator
        weighted_positive *= positive_ratio / continuation
        weighted_negative *= negative_ratio / continuation
        estimate += (weighted_positive - weighted_negative) / order

    if not math.isfinite(estimate):
        raise FloatingPointError("the repeated-label h statistic is non-finite")
    return estimate


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateNode:
    """One exactly-tokenized response sampled from the reference policy."""

    prompt_id: Identifier
    candidate_id: Identifier
    prompt: str
    response: str
    token_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    terminated_by_eos: bool
    reached_max_length: bool
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    _EXPECTED_SCHEMA_VERSION: ClassVar[str] = CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_identifier(self.prompt_id, "prompt_id")
        _validate_identifier(self.candidate_id, "candidate_id")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(self.response, str):
            raise TypeError("response must be a string")

        token_ids = _tuple_of_integers(self.token_ids, "token_ids", binary=False)
        response_mask = _tuple_of_integers(self.response_mask, "response_mask", binary=True)
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "response_mask", response_mask)
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        if len(response_mask) != len(token_ids):
            raise ValueError("response_mask must have the same length as token_ids")
        if not any(response_mask):
            raise ValueError("response_mask must select at least one response token")

        if not isinstance(self.terminated_by_eos, bool):
            raise TypeError("terminated_by_eos must be a boolean")
        if not isinstance(self.reached_max_length, bool):
            raise TypeError("reached_max_length must be a boolean")
        if self.terminated_by_eos and self.reached_max_length:
            raise ValueError("a candidate cannot both terminate by EOS and reach the length limit")
        if self.schema_version != self._EXPECTED_SCHEMA_VERSION:
            raise SchemaError(
                f"schema_version must be {self._EXPECTED_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CandidateNode:
        """Construct a node from an exact JSON-compatible mapping."""

        expected = {field.name for field in fields(cls)}
        _validate_exact_keys(value, expected, context=cls.__name__)
        payload = dict(value)
        payload["token_ids"] = _tuple_of_integers(payload["token_ids"], "token_ids", binary=False)
        payload["response_mask"] = _tuple_of_integers(
            payload["response_mask"], "response_mask", binary=True
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation with no implicit fields."""

        return {
            "prompt_id": self.prompt_id,
            "candidate_id": self.candidate_id,
            "prompt": self.prompt,
            "response": self.response,
            "token_ids": list(self.token_ids),
            "response_mask": list(self.response_mask),
            "terminated_by_eos": self.terminated_by_eos,
            "reached_max_length": self.reached_max_length,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedEdgeRecord:
    """Repeated binary annotations in a fixed left/right orientation.

    ``left_id`` and ``right_id`` are the caller's canonical endpoint ids.
    They are intentionally not sorted here: lexicographic sorting would, for
    example, put candidate ``"10"`` before ``"2"`` and could change a
    previously declared orientation.  Chosen/rejected keys are rejected by
    :meth:`from_dict` rather than being converted.
    """

    edge_id: Identifier
    prompt_id: Identifier
    left_id: Identifier
    right_id: Identifier
    raw_labels: tuple[int, ...]
    num_annotations: int
    left_wins: int
    h: float
    schema_version: str = REPEATED_EDGE_SCHEMA_VERSION

    _EXPECTED_SCHEMA_VERSION: ClassVar[str] = REPEATED_EDGE_SCHEMA_VERSION
    _FORBIDDEN_ORIENTATION_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"chosen", "rejected", "chosen_id", "rejected_id"}
    )

    def __post_init__(self) -> None:
        for name in ("edge_id", "prompt_id", "left_id", "right_id"):
            _validate_identifier(getattr(self, name), name)
        if self.left_id == self.right_id:
            raise ValueError("left_id and right_id must identify distinct candidates")

        labels = _tuple_of_integers(self.raw_labels, "raw_labels", binary=True)
        object.__setattr__(self, "raw_labels", labels)
        if isinstance(self.num_annotations, bool) or not isinstance(self.num_annotations, int):
            raise TypeError("num_annotations must be an integer")
        if self.num_annotations < 1:
            raise ValueError("num_annotations must be at least one")
        if self.num_annotations != len(labels):
            raise ValueError("num_annotations must equal len(raw_labels)")
        if isinstance(self.left_wins, bool) or not isinstance(self.left_wins, int):
            raise TypeError("left_wins must be an integer")
        if self.left_wins != sum(labels):
            raise ValueError("left_wins must equal sum(raw_labels)")

        if isinstance(self.h, bool) or not isinstance(self.h, (int, float)):
            raise TypeError("h must be a real number")
        h_value = float(self.h)
        if not math.isfinite(h_value):
            raise ValueError("h must be finite")
        expected_h = repeated_labels_to_h(labels)
        if not math.isclose(h_value, expected_h, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "h is inconsistent with raw_labels under the locked "
                f"gamma={DEFAULT_CONTINUATION_PROBABILITY}: "
                f"expected {expected_h!r}, got {h_value!r}"
            )
        object.__setattr__(self, "h", h_value)

        if self.schema_version != self._EXPECTED_SCHEMA_VERSION:
            raise SchemaError(
                f"schema_version must be {self._EXPECTED_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RepeatedEdgeRecord:
        """Construct an edge without changing its declared orientation."""

        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__} must be a mapping")
        orientation_keys = set(value) & cls._FORBIDDEN_ORIENTATION_KEYS
        if orientation_keys:
            raise SchemaError(
                "chosen/rejected edge orientation is forbidden; provide canonical "
                f"left_id/right_id fields (found {sorted(orientation_keys)!r})"
            )
        expected = {field.name for field in fields(cls)}
        _validate_exact_keys(value, expected, context=cls.__name__)
        payload = dict(value)
        payload["raw_labels"] = _tuple_of_integers(payload["raw_labels"], "raw_labels", binary=True)
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        """Return the exact base edge schema as JSON-compatible values."""

        return {
            "edge_id": self.edge_id,
            "prompt_id": self.prompt_id,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "raw_labels": list(self.raw_labels),
            "num_annotations": self.num_annotations,
            "left_wins": self.left_wins,
            "h": self.h,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingEdgeRecord(RepeatedEdgeRecord):
    """Training edge schema that cannot carry oracle/evaluation fields."""

    schema_version: str = TRAINING_EDGE_SCHEMA_VERSION

    _EXPECTED_SCHEMA_VERSION: ClassVar[str] = TRAINING_EDGE_SCHEMA_VERSION
    _KNOWN_EVALUATION_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "true_margin",
            "oracle_score",
            "oracle_reward",
            "raw_oracle_score",
            "evaluation_only",
        }
    )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TrainingEdgeRecord:
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__} must be a mapping")
        leakage = {
            key
            for key in value
            if key in cls._KNOWN_EVALUATION_FIELDS
            or key.startswith("evaluation_")
            or key.startswith("oracle_")
        }
        if leakage:
            raise EvaluationLeakageError(
                f"evaluation-only fields are forbidden in TrainingEdgeRecord: {sorted(leakage)!r}"
            )
        # ``slots=True`` dataclasses are rebuilt by the dataclass decorator;
        # spelling out the class avoids CPython's zero-argument ``super``
        # closure referring to the pre-rebuild class object.
        return super(TrainingEdgeRecord, cls).from_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationEdgeRecord(RepeatedEdgeRecord):
    """Evaluation-only edge with the latent oriented reward margin."""

    true_margin: float
    schema_version: str = EVALUATION_EDGE_SCHEMA_VERSION

    _EXPECTED_SCHEMA_VERSION: ClassVar[str] = EVALUATION_EDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        RepeatedEdgeRecord.__post_init__(self)
        if isinstance(self.true_margin, bool) or not isinstance(self.true_margin, (int, float)):
            raise TypeError("true_margin must be a real number")
        margin = float(self.true_margin)
        if not math.isfinite(margin):
            raise ValueError("true_margin must be finite")
        object.__setattr__(self, "true_margin", margin)

    def to_dict(self) -> dict[str, object]:
        result = RepeatedEdgeRecord.to_dict(self)
        result["true_margin"] = self.true_margin
        return result


EdgeRecordT = TypeVar("EdgeRecordT", bound=RepeatedEdgeRecord)


def swap_edge_orientation(record: EdgeRecordT) -> EdgeRecordT:
    """Return the mathematically equivalent record in the reverse direction.

    Labels are complemented because they always mean "current left wins".
    Consequently ``left_wins`` becomes ``N-left_wins`` and both ``h`` and an
    evaluation ``true_margin`` change sign.  ``edge_id`` is retained because
    this is the same unordered edge; the helper is intended for invariance
    checks, not for redefining the canonical on-disk direction.
    """

    if not isinstance(record, RepeatedEdgeRecord):
        raise TypeError("record must be a RepeatedEdgeRecord")
    changes: dict[str, object] = {
        "left_id": record.right_id,
        "right_id": record.left_id,
        "raw_labels": tuple(1 - label for label in record.raw_labels),
        "left_wins": record.num_annotations - record.left_wins,
        "h": -record.h,
    }
    if isinstance(record, EvaluationEdgeRecord):
        changes["true_margin"] = -record.true_margin
    return replace(record, **changes)


PromptRecord = CandidateNode | RepeatedEdgeRecord | Identifier


def validate_disjoint_prompt_splits(
    splits: Mapping[str, Iterable[PromptRecord]],
) -> None:
    """Fail if any prompt id occurs in more than one named split.

    Values may be prompt identifiers directly or records exposing a
    ``prompt_id`` attribute.  Repeated records for one prompt inside a split
    are expected and therefore collapse to one id before comparison.
    """

    if not isinstance(splits, Mapping):
        raise TypeError("splits must be a mapping from split name to records")
    seen: dict[Identifier, str] = {}
    for split_name, records in splits.items():
        if not isinstance(split_name, str) or not split_name:
            raise ValueError("split names must be non-empty strings")
        if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Iterable):
            raise TypeError(f"split {split_name!r} must contain an iterable of records")
        current: set[Identifier] = set()
        for record in records:
            prompt_id = getattr(record, "prompt_id", record)
            _validate_identifier(prompt_id, f"prompt id in split {split_name!r}")
            current.add(prompt_id)
        conflicts: dict[str, list[Identifier]] = {}
        for prompt_id in current:
            prior_split = seen.get(prompt_id)
            if prior_split is not None:
                conflicts.setdefault(prior_split, []).append(prompt_id)
            else:
                seen[prompt_id] = split_name
        if conflicts:
            details = "; ".join(
                f"{prior!r} and {split_name!r}: {sorted(repr(item) for item in prompt_ids)}"
                for prior, prompt_ids in sorted(conflicts.items())
            )
            raise ValueError(f"prompt ids must be disjoint across splits ({details})")


def validate_disjoint_prompt_ids(splits: Mapping[str, Iterable[Identifier]]) -> None:
    """Identifier-only alias for :func:`validate_disjoint_prompt_splits`."""

    validate_disjoint_prompt_splits(splits)


RecordT = TypeVar("RecordT", CandidateNode, RepeatedEdgeRecord)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_jsonl(
    path: str | os.PathLike[str],
    record_type: type[RecordT],
) -> list[RecordT]:
    """Load an exact record type from a strict UTF-8 JSONL file."""

    if not isinstance(record_type, type) or not issubclass(
        record_type, (CandidateNode, RepeatedEdgeRecord)
    ):
        raise TypeError("record_type must be a CandidateNode or RepeatedEdgeRecord class")
    source = Path(path)
    result: list[RecordT] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise SchemaError(f"{source}:{line_number}: blank JSONL lines are forbidden")
            try:
                value = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
                if not isinstance(value, dict):
                    raise SchemaError("the JSON value must be an object")
                result.append(record_type.from_dict(value))
            except (json.JSONDecodeError, SchemaError, TypeError, ValueError) as error:
                if isinstance(error, EvaluationLeakageError):
                    raise EvaluationLeakageError(f"{source}:{line_number}: {error}") from error
                raise SchemaError(f"{source}:{line_number}: {error}") from error
    return result


def save_jsonl(
    path: str | os.PathLike[str],
    records: Iterable[CandidateNode | RepeatedEdgeRecord],
) -> None:
    """Atomically replace ``path`` with strictly serialized JSONL records."""

    destination = Path(path)
    if not destination.parent.exists():
        raise FileNotFoundError(f"destination directory does not exist: {destination.parent}")

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
            for record in records:
                if not isinstance(record, (CandidateNode, RepeatedEdgeRecord)):
                    raise TypeError(
                        "records must contain only CandidateNode or RepeatedEdgeRecord objects"
                    )
                json.dump(
                    record.to_dict(),
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
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


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "DEFAULT_CONTINUATION_PROBABILITY",
    "EVALUATION_EDGE_SCHEMA_VERSION",
    "REPEATED_EDGE_SCHEMA_VERSION",
    "TRAINING_EDGE_SCHEMA_VERSION",
    "CandidateNode",
    "EvaluationEdgeRecord",
    "EvaluationLeakageError",
    "Identifier",
    "RepeatedEdgeRecord",
    "SchemaError",
    "TrainingEdgeRecord",
    "load_jsonl",
    "repeated_labels_to_h",
    "save_jsonl",
    "swap_edge_orientation",
    "validate_disjoint_prompt_ids",
    "validate_disjoint_prompt_splits",
]
