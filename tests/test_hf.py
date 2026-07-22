from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import smart_reward.hf as hf
from smart_reward.hf import (
    assert_noop_logits,
    build_exact_token_candidates,
    build_oracle_chat,
    configure_fixed_a_lora,
    extract_scalar_oracle_logits,
    generate_exact_candidates,
    pool_final_response_hidden_state,
    score_exact_candidates,
    score_oracle_chats,
    validate_exact_generation_kwargs,
)


def test_generation_kwargs_force_unwarped_temperature_one_distribution() -> None:
    actual = validate_exact_generation_kwargs(
        {
            "do_sample": True,
            "temperature": 1,
            "top_p": 1.0,
            "top_k": 0,
            "min_new_tokens": 0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 12,
            "num_return_sequences": 4,
        }
    )

    assert actual["do_sample"] is True
    assert actual["temperature"] == 1.0
    assert actual["top_p"] == 1.0
    assert actual["top_k"] == 0
    assert actual["min_new_tokens"] == 0
    assert actual["repetition_penalty"] == 1.0
    assert actual["num_beams"] == 1
    assert actual["forced_eos_token_id"] is None
    assert actual["bad_words_ids"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"do_sample": False},
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"top_k": 20},
        {"min_new_tokens": 1},
        {"repetition_penalty": 1.1},
        {"num_beams": 2},
        {"logits_processor": []},
        {"stopping_criteria": []},
        {"generation_config": object()},
        {"unknown_model_knob": True},
    ],
)
def test_generation_kwargs_fail_closed_on_distribution_changes(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        validate_exact_generation_kwargs(kwargs)


class _FakePolicy(nn.Module):
    def __init__(self, generated: torch.Tensor) -> None:
        super().__init__()
        self.generated = generated
        self.scale_lora_B = nn.Parameter(torch.tensor(0.0))
        self.generation_config = SimpleNamespace(eos_token_id=9, pad_token_id=0)
        self.generate_kwargs: dict[str, object] | None = None
        self.scored_input_pointer: int | None = None

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.generate_kwargs = dict(kwargs)
        return self.generated

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        self.scored_input_pointer = input_ids.data_ptr()
        vocabulary_size = 10
        token_axis = torch.arange(vocabulary_size, dtype=torch.float32)
        base = token_axis.view(1, 1, -1).expand(*input_ids.shape, -1)
        logits = base + self.scale_lora_B * token_axis.square().view(1, 1, -1)
        return SimpleNamespace(logits=logits)


def test_generation_and_scoring_reuse_exact_ids_and_same_policy_instance() -> None:
    prompt_ids = torch.tensor([[4, 5]])
    prompt_mask = torch.ones_like(prompt_ids)
    generated = torch.tensor(
        [
            [4, 5, 7, 9, 0],
            [4, 5, 8, 6, 9],
            [4, 5, 1, 2, 3],
        ]
    )
    model = _FakePolicy(generated).eval()

    candidates = generate_exact_candidates(
        model,
        prompt_ids,
        prompt_attention_mask=prompt_mask,
        generation_kwargs={"max_new_tokens": 3, "num_return_sequences": 3},
    )

    assert candidates.input_ids is generated
    assert candidates.response_mask.tolist() == [
        [False, False, True, True, False],
        [False, False, True, True, True],
        [False, False, True, True, True],
    ]
    assert candidates.terminated_by_eos.tolist() == [True, True, False]
    assert candidates.reached_max_length.tolist() == [False, False, True]
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["temperature"] == 1.0
    assert model.generate_kwargs["top_p"] == 1.0
    assert model.generate_kwargs["num_beams"] == 1

    log_probabilities = score_exact_candidates(model, candidates)
    assert log_probabilities.shape == (3,)
    assert log_probabilities.requires_grad
    assert model.scored_input_pointer == generated.data_ptr()

    other_policy = _FakePolicy(generated).eval()
    with pytest.raises(ValueError, match="exact policy instance"):
        score_exact_candidates(other_policy, candidates)

    with torch.no_grad():
        model.scale_lora_B.add_(1.0)
    with pytest.raises(ValueError, match="changed between generation and scoring"):
        score_exact_candidates(model, candidates)


def test_exact_candidate_builder_includes_eos_and_handles_left_padding() -> None:
    prompt_ids = torch.tensor([[0, 0, 3, 4], [0, 6, 7, 8]])
    prompt_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    generated = torch.tensor(
        [
            [0, 0, 3, 4, 2, 9, 0],
            [0, 6, 7, 8, 5, 1, 9],
        ]
    )

    batch = build_exact_token_candidates(
        prompt_ids,
        SimpleNamespace(sequences=generated),
        eos_token_id=[9],
        prompt_attention_mask=prompt_mask,
        pad_token_id=0,
        max_length=7,
    )

    assert batch.input_ids is generated
    assert batch.response_mask.tolist() == [
        [False, False, False, False, True, True, False],
        [False, False, False, False, True, True, True],
    ]
    assert batch.attention_mask.tolist() == [
        [False, False, True, True, True, True, False],
        [False, True, True, True, True, True, True],
    ]


def test_candidate_builder_rejects_retokenized_prefix_and_tokens_after_eos() -> None:
    prompt_ids = torch.tensor([[3, 4]])
    with pytest.raises(ValueError, match="exact prompt token prefix"):
        build_exact_token_candidates(
            prompt_ids,
            torch.tensor([[3, 8, 9]]),
            eos_token_id=9,
            pad_token_id=0,
        )
    with pytest.raises(ValueError, match="non-padding token"):
        build_exact_token_candidates(
            prompt_ids,
            torch.tensor([[3, 4, 9, 7]]),
            eos_token_id=9,
            pad_token_id=0,
        )


class _FakeAdapter(nn.Module):
    def __init__(self, *, nonzero_b: bool = False) -> None:
        super().__init__()
        self.base_weight = nn.Parameter(torch.tensor([4.0]))
        # Deliberately register z before a: the stable layout must sort names.
        self.z_lora_B = nn.Parameter(torch.full((1, 2), float(nonzero_b)))
        self.z_lora_A = nn.Parameter(torch.tensor([[0.2], [-0.3]]))
        self.a_lora_B = nn.Parameter(torch.zeros(1, 1))
        self.a_lora_A = nn.Parameter(torch.tensor([[0.7]]))


def test_configure_fixed_a_lora_freezes_a_opens_zero_b_and_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter()
    fake_peft = SimpleNamespace(get_peft_model=lambda model, config: adapter)
    monkeypatch.setattr(hf.importlib, "import_module", lambda name: fake_peft)

    setup = configure_fixed_a_lora(nn.Linear(1, 1), object())

    assert setup.model is adapter
    assert setup.trainable_names == ("a_lora_B", "z_lora_B")
    assert setup.layout.names == setup.trainable_names
    assert len(setup.a_state_sha256) == 64
    assert setup.named_tangent_parameters() == (
        ("a_lora_B", adapter.a_lora_B),
        ("z_lora_B", adapter.z_lora_B),
    )
    assert not adapter.base_weight.requires_grad
    assert not adapter.a_lora_A.requires_grad
    assert not adapter.z_lora_A.requires_grad
    assert adapter.a_lora_B.requires_grad
    assert adapter.z_lora_B.requires_grad

    second_adapter = _FakeAdapter()
    monkeypatch.setattr(
        hf.importlib,
        "import_module",
        lambda name: SimpleNamespace(get_peft_model=lambda model, config: second_adapter),
    )
    second = configure_fixed_a_lora(nn.Linear(1, 1), object())
    assert second.a_state_sha256 == setup.a_state_sha256


def test_configure_fixed_a_lora_fails_on_missing_peft_or_nonzero_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(hf.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"smart-reward-model\[llm\]"):
        configure_fixed_a_lora(nn.Linear(1, 1), object())

    adapter = _FakeAdapter(nonzero_b=True)
    monkeypatch.setattr(
        hf.importlib,
        "import_module",
        lambda name: SimpleNamespace(get_peft_model=lambda model, config: adapter),
    )
    with pytest.raises(ValueError, match="initialized exactly to zero"):
        configure_fixed_a_lora(nn.Linear(1, 1), object())


def test_noop_logits_check_reports_error_and_fails_above_tolerance() -> None:
    reference = torch.tensor([[1.0, -2.0]])
    adapted = reference + torch.tensor([[1.0e-7, 0.0]])
    measured_error = assert_noop_logits(reference, adapted, atol=1.0e-6, rtol=0.0)
    assert 0.0 < measured_error < 1.0e-6
    with pytest.raises(AssertionError, match="max_abs_error"):
        assert_noop_logits(reference, reference + 0.1, atol=1.0e-6, rtol=0.0)


def test_final_response_pooling_uses_eos_or_final_length_limited_token() -> None:
    first_layer = torch.full((2, 5, 3), -1.0)
    final_layer = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    response_mask = torch.tensor(
        [
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1],
        ]
    )

    pooled = pool_final_response_hidden_state((first_layer, final_layer), response_mask)
    torch.testing.assert_close(pooled, torch.stack((final_layer[0, 3], final_layer[1, 4])))

    with pytest.raises(ValueError, match="at least one response"):
        pool_final_response_hidden_state(final_layer, torch.zeros(2, 5))
    with pytest.raises(ValueError, match="contiguous"):
        pool_final_response_hidden_state(
            final_layer,
            torch.tensor([[0, 1, 0, 1, 0], [0, 1, 1, 1, 0]]),
        )


class _FakeTokenizer:
    def __init__(self) -> None:
        self.chats: list[list[dict[str, str]]] | None = None
        self.kwargs: dict[str, object] | None = None

    def apply_chat_template(
        self, chats: list[list[dict[str, str]]], **kwargs: object
    ) -> dict[str, torch.Tensor]:
        self.chats = chats
        self.kwargs = kwargs
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeOracle(nn.Module):
    def forward(self, **model_inputs: torch.Tensor) -> SimpleNamespace:
        batch_size = model_inputs["input_ids"].shape[0]
        return SimpleNamespace(logits=torch.arange(batch_size, dtype=torch.float32)[:, None])


def test_oracle_chat_follows_model_card_without_system_and_extracts_scalar() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeOracle()
    model.train()

    scores = score_oracle_chats(model, tokenizer, ["p1", "p2"], ["r1", "r2"])

    torch.testing.assert_close(scores, torch.tensor([0.0, 1.0]))
    assert model.training
    assert tokenizer.chats == [
        [
            {"role": "user", "content": "p1"},
            {"role": "assistant", "content": "r1"},
        ],
        [
            {"role": "user", "content": "p2"},
            {"role": "assistant", "content": "r2"},
        ],
    ]
    assert all(message["role"] != "system" for chat in tokenizer.chats for message in chat)
    assert tokenizer.kwargs == {
        "tokenize": True,
        "add_generation_prompt": False,
        "padding": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    assert build_oracle_chat("prompt", "response")[0]["role"] == "user"

    with pytest.raises(ValueError, match=r"\(batch, 1\)"):
        extract_scalar_oracle_logits(SimpleNamespace(logits=torch.zeros(2, 2)))
