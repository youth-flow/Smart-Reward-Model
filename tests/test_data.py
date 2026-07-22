from __future__ import annotations

import json

import pytest
import torch

from smart_reward.annotations import repeated_labels_to_h as tensor_labels_to_h
from smart_reward.data import (
    CandidateNode,
    EvaluationEdgeRecord,
    EvaluationLeakageError,
    TrainingEdgeRecord,
    load_jsonl,
    repeated_labels_to_h,
    save_jsonl,
    swap_edge_orientation,
    validate_disjoint_prompt_splits,
)


def _training_edge(**changes: object) -> TrainingEdgeRecord:
    values: dict[str, object] = {
        "edge_id": "prompt-1:0-1",
        "prompt_id": "prompt-1",
        "left_id": "prompt-1:0",
        "right_id": "prompt-1:1",
        "raw_labels": (1, 1, 0),
        "num_annotations": 3,
        "left_wins": 2,
        "h": repeated_labels_to_h((1, 1, 0)),
    }
    values.update(changes)
    return TrainingEdgeRecord(**values)


def test_swapping_edge_preserves_oriented_moment() -> None:
    record = _training_edge()
    swapped = swap_edge_orientation(record)

    assert swapped.left_id == record.right_id
    assert swapped.right_id == record.left_id
    assert swapped.raw_labels == tuple(1 - value for value in record.raw_labels)
    assert swapped.left_wins == record.num_annotations - record.left_wins
    assert swapped.h == -record.h
    assert swap_edge_orientation(swapped) == record

    # z, the model margin t, and h all use the same endpoint orientation.
    left_score, right_score = 2.5, -0.75
    left_reward, right_reward = 0.2, -0.4
    z = left_score - right_score
    t = left_reward - right_reward
    swapped_z = right_score - left_score
    swapped_t = right_reward - left_reward
    assert swapped_z * (swapped_t - swapped.h) == pytest.approx(z * (t - record.h))


def test_standard_library_and_tensor_h_implementations_agree() -> None:
    labels = (1, 0, 1, 1, 0, 0, 1)
    expected = repeated_labels_to_h(labels)
    actual = tensor_labels_to_h(torch.tensor(labels), gamma=0.9)
    assert actual.item() == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_evaluation_swap_negates_true_margin() -> None:
    base = _training_edge().to_dict()
    base["schema_version"] = "evaluation-edge/v1"
    base["true_margin"] = 0.625
    record = EvaluationEdgeRecord.from_dict(base)
    swapped = swap_edge_orientation(record)
    assert swapped.true_margin == -record.true_margin
    assert swap_edge_orientation(swapped) == record


def test_training_schema_hard_fails_on_evaluation_leakage(tmp_path) -> None:
    leaked = _training_edge().to_dict()
    leaked["true_margin"] = 0.5
    with pytest.raises(EvaluationLeakageError, match="true_margin"):
        TrainingEdgeRecord.from_dict(leaked)

    path = tmp_path / "leaked.jsonl"
    path.write_text(json.dumps(leaked) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationLeakageError, match="true_margin"):
        load_jsonl(path, TrainingEdgeRecord)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"right_id": "prompt-1:0"}, "distinct"),
        ({"num_annotations": 4}, "len"),
        ({"left_wins": 1}, "sum"),
        ({"raw_labels": (1, 2, 0)}, "only 0 and 1"),
        ({"h": 0.0}, "inconsistent"),
    ],
)
def test_repeated_edge_consistency_checks(changes: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _training_edge(**changes)


def test_candidate_and_edge_jsonl_roundtrip(tmp_path) -> None:
    candidate = CandidateNode(
        prompt_id="prompt-1",
        candidate_id="prompt-1:0",
        prompt="Give a short answer.",
        response="Done.",
        token_ids=(11, 12, 13, 2),
        response_mask=(0, 0, 1, 1),
        terminated_by_eos=True,
        reached_max_length=False,
    )
    candidate_path = tmp_path / "candidates.jsonl"
    save_jsonl(candidate_path, [candidate])
    assert load_jsonl(candidate_path, CandidateNode) == [candidate]
    assert CandidateNode.from_dict(candidate.to_dict()) == candidate

    edge = _training_edge()
    edge_path = tmp_path / "training-edges.jsonl"
    edge_path.write_text("old content that must be atomically replaced\n", encoding="utf-8")
    save_jsonl(edge_path, [edge])
    assert load_jsonl(edge_path, TrainingEdgeRecord) == [edge]
    assert not list(tmp_path.glob(".training-edges.jsonl.*.tmp"))


def test_prompt_split_ids_must_be_disjoint() -> None:
    train = [_training_edge(), _training_edge(edge_id="prompt-1:another")]
    validation = [_training_edge(edge_id="prompt-2:edge", prompt_id="prompt-2")]
    validate_disjoint_prompt_splits({"train": train, "validation": validation})

    test = [_training_edge(edge_id="prompt-1:test")]
    with pytest.raises(ValueError, match="disjoint"):
        validate_disjoint_prompt_splits({"train": train, "test": test})


def test_chosen_rejected_orientation_is_never_silently_converted() -> None:
    payload = _training_edge().to_dict()
    payload["chosen_id"] = payload.pop("left_id")
    payload["rejected_id"] = payload.pop("right_id")
    with pytest.raises(ValueError, match="chosen/rejected"):
        TrainingEdgeRecord.from_dict(payload)
