import random

import torch

from smart_reward.seeding import SeedBundle, derive_seed, seed_python_and_torch


def test_named_seeds_are_stable_distinct_and_json_ready() -> None:
    first = SeedBundle.from_base_seed(20260722)
    second = SeedBundle.from_base_seed(20260722)
    different = SeedBundle.from_base_seed(20260723)

    assert first == second
    assert first != different
    assert len(set(first.to_dict().values())) == len(first.to_dict())
    assert first.prompt_split == derive_seed(20260722, "prompt_split")


def test_seed_python_and_torch_replays_both_streams() -> None:
    seed_python_and_torch(91)
    first_python = [random.random() for _ in range(3)]
    first_torch = torch.rand(3)

    seed_python_and_torch(91)
    assert [random.random() for _ in range(3)] == first_python
    torch.testing.assert_close(torch.rand(3), first_torch)
