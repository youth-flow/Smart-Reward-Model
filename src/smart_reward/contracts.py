"""Stable method identifiers and versioned serialization contracts.

Human-facing terminology uses ProRM/ProRM+.  The repository still accepts the
legacy ``srm_plus`` identifier at explicit v1 input boundaries so historical
artifacts remain readable; all new writers emit ``prorm_plus`` under v2
schemas.
"""

from __future__ import annotations

from collections.abc import Mapping

BT_MLE = "bt_mle"
PRORM_PLUS = "prorm_plus"
LEGACY_SRM_PLUS = "srm_plus"

CANONICAL_LEARNERS = (BT_MLE, PRORM_PLUS)
LEGACY_V1_LEARNERS = (BT_MLE, LEGACY_SRM_PLUS)

CONTROLLED_COMPARISON_SCHEMA_V1 = "controlled-comparison/v1"
CONTROLLED_COMPARISON_SCHEMA_V2 = "controlled-comparison/v2"
MATCHED_KL_ROLLOUT_SCHEMA_V1 = "matched-kl-rollout/v1"
MATCHED_KL_ROLLOUT_SCHEMA_V2 = "matched-kl-rollout/v2"
UPDATED_ROLLOUT_SCHEMA_V1 = "updated-rollout/v1"
UPDATED_ROLLOUT_SCHEMA_V2 = "updated-rollout/v2"
PAIRED_AGGREGATE_SCHEMA_V1 = "paired-seed-aggregate/v1"
PAIRED_AGGREGATE_SCHEMA_V2 = "paired-seed-aggregate/v2"

CHECKPOINT_FORMAT_V1 = 1
CHECKPOINT_FORMAT_V2 = 2


def compatibility_value(
    source: Mapping[str, object],
    canonical_name: str,
    legacy_name: str,
) -> object | None:
    """Resolve one canonical/legacy key pair and reject conflicting values."""

    canonical_present = canonical_name in source
    legacy_present = legacy_name in source
    canonical = source[canonical_name] if canonical_present else None
    legacy = source[legacy_name] if legacy_present else None
    if canonical_present and legacy_present and canonical != legacy:
        raise ValueError(f"conflicting {canonical_name} and legacy {legacy_name}")
    return canonical if canonical_present else legacy


def canonical_method(value: object, *, allow_legacy: bool = False) -> str:
    """Return a canonical learner identifier, optionally accepting v1 input."""

    if value in CANONICAL_LEARNERS:
        return str(value)
    if allow_legacy and value == LEGACY_SRM_PLUS:
        return PRORM_PLUS
    expected = repr(CANONICAL_LEARNERS)
    if allow_legacy:
        expected = f"{expected} or legacy {LEGACY_SRM_PLUS!r}"
    raise ValueError(f"method must be one of {expected}; got {value!r}")


def learner_key_for_schema(method: object, schema_version: str) -> str:
    """Return the serialized key for ``method`` under a supported schema."""

    canonical = canonical_method(method, allow_legacy=True)
    if schema_version in {
        CONTROLLED_COMPARISON_SCHEMA_V2,
        MATCHED_KL_ROLLOUT_SCHEMA_V2,
        UPDATED_ROLLOUT_SCHEMA_V2,
        PAIRED_AGGREGATE_SCHEMA_V2,
    }:
        return canonical
    if schema_version in {
        CONTROLLED_COMPARISON_SCHEMA_V1,
        MATCHED_KL_ROLLOUT_SCHEMA_V1,
        UPDATED_ROLLOUT_SCHEMA_V1,
        PAIRED_AGGREGATE_SCHEMA_V1,
    }:
        return LEGACY_SRM_PLUS if canonical == PRORM_PLUS else canonical
    raise ValueError(f"unsupported schema version: {schema_version!r}")


__all__ = [
    "BT_MLE",
    "CANONICAL_LEARNERS",
    "CHECKPOINT_FORMAT_V1",
    "CHECKPOINT_FORMAT_V2",
    "CONTROLLED_COMPARISON_SCHEMA_V1",
    "CONTROLLED_COMPARISON_SCHEMA_V2",
    "LEGACY_SRM_PLUS",
    "LEGACY_V1_LEARNERS",
    "MATCHED_KL_ROLLOUT_SCHEMA_V1",
    "MATCHED_KL_ROLLOUT_SCHEMA_V2",
    "PAIRED_AGGREGATE_SCHEMA_V1",
    "PAIRED_AGGREGATE_SCHEMA_V2",
    "PRORM_PLUS",
    "UPDATED_ROLLOUT_SCHEMA_V1",
    "UPDATED_ROLLOUT_SCHEMA_V2",
    "canonical_method",
    "compatibility_value",
    "learner_key_for_schema",
]
