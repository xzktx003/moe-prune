from __future__ import annotations

import pytest
import torch

from moe_prune.code.src.aimer_selector import (
    build_aimer_keep_mask,
    build_aimer_keep_table_for_model,
    compute_aimer_removal_score,
)


class _Proj(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight)


class _Expert(torch.nn.Module):
    def __init__(self, gate_proj: torch.Tensor, up_proj: torch.Tensor, down_proj: torch.Tensor) -> None:
        super().__init__()
        self.gate_proj = _Proj(gate_proj)
        self.up_proj = _Proj(up_proj)
        self.down_proj = _Proj(down_proj)


class _FusedExperts(torch.nn.Module):
    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(gate_up_proj)
        self.down_proj = torch.nn.Parameter(down_proj)


def test_compute_aimer_removal_score_matches_l1_over_rms_formula() -> None:
    gate = torch.tensor([[1.0, -1.0]])
    up = torch.tensor([[2.0, -2.0]])
    down = torch.tensor([[3.0], [-3.0]])

    score = compute_aimer_removal_score(gate, up, down, eps=0.0)

    values = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    expected = values.abs().mean() / torch.sqrt(values.square().mean())
    assert score.item() == pytest.approx(expected.item())


def test_build_aimer_keep_mask_inverts_static_removal_importance_for_dynamic_keep() -> None:
    gate = torch.tensor([[0.45, 0.35, 0.20]], dtype=torch.float32)
    keep_selected = torch.tensor([[0.5, 1.5, 1.0]], dtype=torch.float32)

    keep_mask, p_final = build_aimer_keep_mask(gate, keep_selected, tau=0.30)

    expected_score = gate * keep_selected
    expected_p_final = expected_score / expected_score.sum(dim=-1, keepdim=True)
    assert torch.allclose(p_final, expected_p_final)
    assert keep_mask.tolist() == [[False, True, False]]


def test_build_aimer_keep_table_for_model_supports_module_list_experts() -> None:
    expert0 = _Expert(
        gate_proj=torch.ones(2, 2),
        up_proj=2 * torch.ones(2, 2),
        down_proj=3 * torch.ones(2, 2),
    )
    expert1 = _Expert(
        gate_proj=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        up_proj=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        down_proj=torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
    )
    mlp = type("MLP", (), {"experts": torch.nn.ModuleList([expert0, expert1])})()
    layer = type("Layer", (), {"mlp": mlp})()
    model = type("Model", (), {"model": type("Inner", (), {"layers": [layer]})()})()

    keep_table = build_aimer_keep_table_for_model(model)

    assert 0 in keep_table
    assert keep_table[0].shape == (2,)
    assert torch.isfinite(keep_table[0]).all()
    assert keep_table[0].mean().item() == pytest.approx(1.0)
    assert keep_table[0][1].item() > keep_table[0][0].item()


def test_build_aimer_keep_table_for_model_supports_fused_experts() -> None:
    gate_up_proj = torch.stack(
        [
            torch.ones(4, 2),
            torch.eye(4, 2),
        ],
        dim=0,
    )
    down_proj = torch.stack(
        [
            2 * torch.ones(2, 2),
            torch.eye(2, 2),
        ],
        dim=0,
    )
    layer = type(
        "Layer",
        (),
        {
            "enable_moe_block": True,
            "router": object(),
            "experts": _FusedExperts(gate_up_proj, down_proj),
        },
    )()
    language_model = type("LanguageModel", (), {"layers": [layer]})()
    outer_model = type("Outer", (), {"language_model": language_model})()
    model = type("Model", (), {"model": outer_model})()

    keep_table = build_aimer_keep_table_for_model(model)

    assert 0 in keep_table
    assert keep_table[0].shape == (2,)
    assert torch.isfinite(keep_table[0]).all()
    assert keep_table[0].mean().item() == pytest.approx(1.0)
