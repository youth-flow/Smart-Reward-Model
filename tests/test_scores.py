import json

import pytest
import torch
from torch import nn

from smart_reward.scores import (
    ParameterLayout,
    edge_score_differences,
    empirical_score_diagnostics,
    per_sample_scores,
    select_named_tangent_parameters,
    sequence_log_probs,
)


def test_sequence_log_probs_uses_causal_shift_and_response_mask() -> None:
    logits = torch.tensor(
        [
            [
                [3.0, -1.0, 0.5],
                [-0.5, 2.0, 0.0],
                [0.2, -0.3, 1.4],
                [7.0, -4.0, 2.0],
            ]
        ],
        requires_grad=True,
    )
    input_ids = torch.tensor([[0, 1, 2, 0]])
    response_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)

    actual = sequence_log_probs(logits, input_ids, response_mask)
    normalized = logits.log_softmax(dim=-1)
    expected = normalized[0, 1, 2] + normalized[0, 2, 0]
    torch.testing.assert_close(actual, expected.unsqueeze(0))

    gradient = torch.autograd.grad(actual.sum(), logits)[0]
    torch.testing.assert_close(gradient[:, 0], torch.zeros_like(gradient[:, 0]))
    torch.testing.assert_close(gradient[:, 3], torch.zeros_like(gradient[:, 3]))
    assert gradient[:, 1:3].abs().sum() > 0


def test_sequence_log_probs_does_not_count_prompt_predictions() -> None:
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(2, 5, 4, generator=generator)
    input_ids = torch.tensor([[1, 2, 3, 0, 1], [2, 0, 1, 3, 2]])
    response_mask = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])

    baseline = sequence_log_probs(logits, input_ids, response_mask)
    changed_prompt_logits = logits.clone()
    changed_prompt_logits[:, :2] = 100.0 * torch.randn(
        changed_prompt_logits[:, :2].shape,
        generator=generator,
    )
    actual = sequence_log_probs(changed_prompt_logits, input_ids, response_mask)
    torch.testing.assert_close(actual, baseline)


@pytest.mark.parametrize(
    ("input_ids", "response_mask", "error"),
    [
        (torch.zeros(2, 3), torch.zeros(2, 3), TypeError),
        (torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 2), ValueError),
        (
            torch.zeros(2, 3, dtype=torch.long),
            torch.tensor([[0, 0, 2], [0, 1, 1]]),
            ValueError,
        ),
    ],
)
def test_sequence_log_probs_validates_ids_shape_and_binary_mask(
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        sequence_log_probs(torch.zeros(2, 3, 5), input_ids, response_mask)


class _TangentModel(nn.Module):
    def __init__(self, *, mixed_trainable: bool = False) -> None:
        super().__init__()
        self.first_lora_B = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.frozen_weight = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.second_lora_B = nn.Parameter(torch.tensor([[3.0]]))
        self.ordinary_weight = nn.Parameter(
            torch.tensor(4.0),
            requires_grad=mixed_trainable,
        )


def test_lora_b_selection_is_ordered_and_fails_closed() -> None:
    model = _TangentModel()
    selected = select_named_tangent_parameters(model)
    assert [name for name, _ in selected] == ["first_lora_B", "second_lora_B"]

    with pytest.raises(ValueError, match="ordinary_weight"):
        select_named_tangent_parameters(_TangentModel(mixed_trainable=True))

    empty = nn.Linear(2, 2)
    empty.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable lora_B"):
        select_named_tangent_parameters(empty)


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_B = nn.Parameter(
            torch.tensor([[0.20, -0.10, 0.05], [-0.30, 0.40, 0.15]], dtype=torch.float64)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.lora_B


def test_per_sample_scores_shape_layout_and_finite_difference() -> None:
    model = _TinyPolicy()
    features = torch.tensor(
        [
            [[1.0, 0.5], [0.2, -0.4], [0.7, 0.1], [-0.2, 0.8]],
            [[-0.3, 0.2], [0.6, 0.9], [0.1, -0.7], [0.4, 0.3]],
        ],
        dtype=torch.float64,
    )
    input_ids = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]])
    response_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    named_parameters = select_named_tangent_parameters(model)
    layout = ParameterLayout.from_named_parameters(named_parameters)

    log_probs = sequence_log_probs(model(features), input_ids, response_mask)
    scores = per_sample_scores(log_probs, named_parameters, layout=layout)

    assert scores.shape == (2, model.lora_B.numel())
    assert scores.dtype == torch.float32
    assert not scores.requires_grad
    assert layout.names == ("lora_B",)
    assert layout.offsets == (0,)
    assert layout.numels == (6,)
    assert ParameterLayout.from_metadata(layout.to_metadata()) == layout
    json.dumps(layout.to_metadata())

    epsilon = 1e-6
    finite_difference = torch.empty_like(scores, dtype=torch.float64)
    with torch.no_grad():
        flat_parameter = model.lora_B.view(-1)
        for parameter_index in range(flat_parameter.numel()):
            original = flat_parameter[parameter_index].item()
            flat_parameter[parameter_index] = original + epsilon
            plus = sequence_log_probs(model(features), input_ids, response_mask)
            flat_parameter[parameter_index] = original - epsilon
            minus = sequence_log_probs(model(features), input_ids, response_mask)
            flat_parameter[parameter_index] = original
            finite_difference[:, parameter_index] = (plus - minus) / (2.0 * epsilon)

    torch.testing.assert_close(scores.double(), finite_difference, rtol=2e-5, atol=2e-6)


def test_parameter_layout_flattens_batched_gradients_in_named_order() -> None:
    first = nn.Parameter(torch.zeros(2))
    second = nn.Parameter(torch.zeros(1, 2))
    layout = ParameterLayout.from_named_parameters(
        (("first_lora_B", first), ("second_lora_B", second))
    )
    gradients = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]]),
    )
    expected = torch.tensor([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]])
    torch.testing.assert_close(layout.flatten_per_sample_gradients(gradients), expected)


def test_edge_score_differences_reverse_sign_when_edges_are_swapped() -> None:
    node_scores = torch.tensor([[1.0, -2.0], [0.5, 4.0], [-3.0, 1.5]])
    left = torch.tensor([0, 2, 1])
    right = torch.tensor([1, 0, 2])

    differences = edge_score_differences(node_scores, left, right)
    reversed_differences = edge_score_differences(node_scores, right, left)
    torch.testing.assert_close(reversed_differences, -differences)
    torch.testing.assert_close(
        edge_score_differences(node_scores, torch.stack((left, right), dim=1)),
        differences,
    )


def test_empirical_score_diagnostics_reports_mean_rms_ratio() -> None:
    scores = torch.tensor([[3.0, 4.0], [-3.0, 4.0]])
    diagnostics = empirical_score_diagnostics(scores)
    assert diagnostics.mean_norm == pytest.approx(4.0)
    assert diagnostics.rms == pytest.approx(5.0)
    assert diagnostics.mean_norm_over_rms == pytest.approx(0.8)
    assert diagnostics["mean_norm/rms"] == pytest.approx(0.8)
