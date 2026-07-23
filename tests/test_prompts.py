import json
from types import SimpleNamespace

import pytest

from smart_reward.prompts import (
    PromptRecord,
    load_multipref_parquet_snapshot,
    load_prompt_jsonl,
    prepare_multipref_prompts,
    save_prompt_jsonl,
)


def _rows() -> list[dict[str, str]]:
    rows = [{"prompt_id": f"p-{index:02d}", "text": f"Question {index}"} for index in range(12)]
    return rows + [rows[3].copy(), rows[7].copy()]


def test_prompt_split_is_deterministic_and_input_order_invariant() -> None:
    sizes = {"train": 6, "validation": 2, "test": 2}
    first = prepare_multipref_prompts(_rows(), split_sizes=sizes, seed=17)
    second = prepare_multipref_prompts(reversed(_rows()), split_sizes=sizes, seed=17)
    different = prepare_multipref_prompts(_rows(), split_sizes=sizes, seed=18)

    assert first == second
    assert [record.prompt_id for record in first] != [record.prompt_id for record in different]
    assert {split: sum(record.split == split for record in first) for split in sizes} == sizes
    assert len({record.prompt_id for record in first}) == len(first)


def test_conflicting_duplicate_and_insufficient_prompts_fail() -> None:
    conflicting = _rows()
    conflicting.append({"prompt_id": "p-03", "text": "Different text"})
    with pytest.raises(ValueError, match="conflicting"):
        prepare_multipref_prompts(
            conflicting,
            split_sizes={"train": 6, "validation": 2, "test": 2},
            seed=1,
        )
    with pytest.raises(ValueError, match="unique prompts"):
        prepare_multipref_prompts(
            _rows(),
            split_sizes={"train": 10, "validation": 2, "test": 2},
            seed=1,
        )


def test_prompt_jsonl_roundtrip_and_duplicate_guard(tmp_path) -> None:
    records = prepare_multipref_prompts(
        _rows(),
        split_sizes={"train": 6, "validation": 2, "test": 2},
        seed=4,
    )
    path = tmp_path / "prompts.jsonl"
    save_prompt_jsonl(path, records)
    assert load_prompt_jsonl(path) == records
    assert PromptRecord.from_dict(records[0].to_dict()) == records[0]

    duplicated = tmp_path / "duplicated.jsonl"
    payload = json.dumps(records[0].to_dict())
    duplicated.write_text(f"{payload}\n{payload}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        load_prompt_jsonl(duplicated)


def test_multipref_snapshot_loader_uses_only_local_train_parquet(tmp_path) -> None:
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    first = snapshot / "data" / "train-00001-of-00002.parquet"
    second = snapshot / "data" / "train-00000-of-00002.parquet"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    cache = tmp_path / "datasets"
    calls = []

    class DownloadConfig:
        def __init__(self, *, local_files_only: bool) -> None:
            self.local_files_only = local_files_only

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return ["rows"]

    datasets_module = SimpleNamespace(
        DownloadConfig=DownloadConfig,
        load_dataset=load_dataset,
    )
    assert load_multipref_parquet_snapshot(
        datasets_module,
        snapshot,
        datasets_cache=cache,
    ) == ["rows"]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("parquet",)
    assert kwargs["data_files"] == {"train": [str(second), str(first)]}
    assert kwargs["split"] == "train"
    assert kwargs["cache_dir"] == str(cache.resolve())
    assert kwargs["download_config"].local_files_only is True
