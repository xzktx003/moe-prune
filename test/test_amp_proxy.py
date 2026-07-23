from __future__ import annotations

import pytest
import torch

from moe_prune.code.src.amp_proxy import (
    build_amp_table_for_model,
    build_router_proto_amp_table_for_layer,
    build_router_proto_amp_table_for_model,
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


class _LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))


class _FusedExperts(torch.nn.Module):
    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(gate_up_proj)
        self.down_proj = torch.nn.Parameter(down_proj)


class _Router(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight)


def test_build_amp_table_for_model_supports_module_list_experts() -> None:
    hidden = 2
    intermediate = 2
    expert0 = _Expert(
        gate_proj=torch.ones(intermediate, hidden),
        up_proj=2 * torch.ones(intermediate, hidden),
        down_proj=3 * torch.ones(hidden, intermediate),
    )
    expert1 = _Expert(
        gate_proj=4 * torch.ones(intermediate, hidden),
        up_proj=5 * torch.ones(intermediate, hidden),
        down_proj=6 * torch.ones(hidden, intermediate),
    )

    mlp = type("MLP", (), {"experts": torch.nn.ModuleList([expert0, expert1])})()
    layer = type("Layer", (), {"mlp": mlp, "post_attention_layernorm": _LayerNorm(hidden)})()
    model = type("Model", (), {"model": type("Inner", (), {"layers": [layer]})()})()

    amp_table = build_amp_table_for_model(model)

    assert 0 in amp_table
    assert amp_table[0].shape == (2,)


def test_build_router_proto_amp_table_for_model_supports_module_list_experts() -> None:
    hidden = 2
    intermediate = 2
    expert0 = _Expert(
        gate_proj=torch.tensor([[1.0, 0.0], [0.5, 1.0]]),
        up_proj=torch.tensor([[2.0, 0.0], [0.5, 2.0]]),
        down_proj=torch.tensor([[1.0, 0.25], [0.0, 1.0]]),
    )
    expert1 = _Expert(
        gate_proj=torch.tensor([[0.5, 1.0], [1.0, 0.5]]),
        up_proj=torch.tensor([[1.5, 0.0], [0.0, 1.5]]),
        down_proj=torch.tensor([[0.75, 0.0], [0.25, 1.25]]),
    )

    mlp = type(
        "MLP",
        (),
        {
            "experts": torch.nn.ModuleList([expert0, expert1]),
            "gate": _Router(torch.tensor([[1.0, 0.5], [0.25, 1.0]])),
        },
    )()
    layer = type("Layer", (), {"mlp": mlp, "post_attention_layernorm": _LayerNorm(hidden)})()
    model = type("Model", (), {"model": type("Inner", (), {"layers": [layer]})()})()

    proto_table = build_router_proto_amp_table_for_model(model)

    assert 0 in proto_table
    assert proto_table[0].shape == (2,)
    assert torch.isfinite(proto_table[0]).all()
    assert proto_table[0].mean().item() == pytest.approx(1.0)


def test_router_weight_centering_defaults_to_false() -> None:
    torch.manual_seed(7)
    router_weight = torch.randn(3, 4)
    gate_up_proj = torch.randn(3, 6, 4)
    down_proj = torch.randn(3, 4, 3)

    default_table = build_router_proto_amp_table_for_layer(
        router_weight, gate_up_proj, down_proj
    )
    explicit_raw_table = build_router_proto_amp_table_for_layer(
        router_weight,
        gate_up_proj,
        down_proj,
        center_router_weights=False,
    )

    assert torch.allclose(default_table, explicit_raw_table)


def test_centered_router_proto_is_invariant_to_shared_layer_translation() -> None:
    torch.manual_seed(11)
    router_weight = torch.randn(4, 5)
    shared_translation = torch.randn(1, 5)
    gate_up_proj = torch.randn(4, 6, 5)
    down_proj = torch.randn(4, 5, 3)

    centered = build_router_proto_amp_table_for_layer(
        router_weight,
        gate_up_proj,
        down_proj,
        center_router_weights=True,
    )
    translated = build_router_proto_amp_table_for_layer(
        router_weight + shared_translation,
        gate_up_proj,
        down_proj,
        center_router_weights=True,
    )

    assert torch.allclose(centered, translated, atol=1e-5, rtol=1e-5)


def test_build_amp_table_for_model_supports_gemma4_layer_experts() -> None:
    hidden = 2
    intermediate = 2
    gate_up_proj = torch.stack(
        [
            torch.ones(2 * intermediate, hidden),
            2 * torch.ones(2 * intermediate, hidden),
        ],
        dim=0,
    )
    down_proj = torch.stack(
        [
            3 * torch.ones(hidden, intermediate),
            4 * torch.ones(hidden, intermediate),
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
            "pre_feedforward_layernorm_2": _LayerNorm(hidden),
        },
    )()
    language_model = type("LanguageModel", (), {"layers": [layer]})()
    outer_model = type("Outer", (), {"language_model": language_model})()
    model = type("Model", (), {"model": outer_model})()

    amp_table = build_amp_table_for_model(model)

    assert 0 in amp_table
    assert amp_table[0].shape == (2,)
