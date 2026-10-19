from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.transformer_final.recall_improvement_study_v1.cell_launcher import (
    _construct_trainable_adamw,
    _freeze_policy,
)


class _DebertaLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deberta = nn.Module()
        self.deberta.embeddings = nn.Embedding(8, 4)
        self.deberta.encoder = nn.Module()
        self.deberta.encoder.layer = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(12)]
        )
        self.deberta.encoder.LayerNorm = nn.LayerNorm(4)
        self.deberta.encoder.rel_embeddings = nn.Embedding(8, 4)
        self.pooler = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 2)


def _expected_names(method: str) -> set[str]:
    if method == "full":
        return {
            "deberta.embeddings.weight",
            "deberta.encoder.LayerNorm.weight",
            "deberta.encoder.LayerNorm.bias",
            "deberta.encoder.rel_embeddings.weight",
            *{
                f"deberta.encoder.layer.{index}.{suffix}"
                for index in range(12)
                for suffix in ("weight", "bias")
            },
            "pooler.weight",
            "pooler.bias",
            "classifier.weight",
            "classifier.bias",
        }
    first_trainable_layer = 8 if method == "partial_4" else 10
    return {
        *{
            f"deberta.encoder.layer.{index}.{suffix}"
            for index in range(first_trainable_layer, 12)
            for suffix in ("weight", "bias")
        },
        "pooler.weight",
        "pooler.bias",
        "classifier.weight",
        "classifier.bias",
    }


@pytest.mark.parametrize("method", ["full", "partial_4", "partial_2"])
def test_adamw_membership_and_cpu_update_follow_freeze_policy(method: str) -> None:
    model = _DebertaLike()
    if method != "full":
        _freeze_policy(model, last_layers=4 if method == "partial_4" else 2)

    named_parameters = dict(model.named_parameters())
    expected_names = _expected_names(method)
    actual_trainable_names = {
        name for name, parameter in named_parameters.items() if parameter.requires_grad
    }
    assert actual_trainable_names == expected_names

    expected_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer, selected_parameters = _construct_trainable_adamw(
        torch.optim.AdamW,
        model.parameters(),
        expected_parameters,
        lr=0.003,
        weight_decay=0.07,
    )

    optimized_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimized_names = {
        name
        for name, parameter in named_parameters.items()
        if any(parameter is selected for selected in optimized_parameters)
    }
    assert optimized_names == expected_names
    assert len({id(parameter) for parameter in optimized_parameters}) == len(
        optimized_parameters
    )
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.003)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.07)
    assert {id(parameter) for parameter in selected_parameters} == {
        id(parameter) for parameter in expected_parameters
    }

    frozen_parameters = [
        parameter for parameter in model.parameters() if not parameter.requires_grad
    ]
    frozen_before = [parameter.detach().clone() for parameter in frozen_parameters]
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in selected_parameters)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in selected_parameters)
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in selected_parameters
        if parameter.grad is not None
    )
    assert all(parameter.grad is None for parameter in frozen_parameters)

    optimizer.step()

    assert all(
        torch.equal(parameter.detach(), before)
        for parameter, before in zip(frozen_parameters, frozen_before, strict=True)
    )
    assert all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in selected_parameters
        if parameter.grad is not None
    )


def test_adamw_rejects_duplicate_model_parameters() -> None:
    model = _DebertaLike()
    parameters = list(model.parameters())
    parameters.append(parameters[0])

    with pytest.raises(RuntimeError, match="duplicate parameters"):
        _construct_trainable_adamw(
            torch.optim.AdamW,
            parameters,
            model.parameters(),
            lr=1e-5,
            weight_decay=0.01,
        )


def test_adamw_rejects_missing_intended_trainable_parameter() -> None:
    model = _DebertaLike()
    policy = _freeze_policy(model, last_layers=4)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    with pytest.raises(RuntimeError, match="do not match"):
        _construct_trainable_adamw(
            torch.optim.AdamW,
            trainable[:-1],
            trainable,
            lr=1e-5,
            weight_decay=0.01,
        )
    assert policy["last_trainable_layers"] == 4


def test_v3_attempt_requires_a_separate_gpu_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.transformer_final.recall_improvement_study_v3_optimizer_fix import (
        _check_authorization,
    )

    monkeypatch.delenv("RECALL_STUDY_V3_GPU_AUTHORIZED", raising=False)
    with pytest.raises(PermissionError, match="separate authorization"):
        _check_authorization()
