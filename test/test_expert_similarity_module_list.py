from __future__ import annotations

import torch

from moe_prune.code.src.expert_similarity import build_model_similarity_table


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


def test_build_model_similarity_table_supports_module_list_experts() -> None:
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

    sim_table = build_model_similarity_table(model, mode="fast")

    assert 0 in sim_table
    assert sim_table[0].shape == (2, 2)


def test_build_model_similarity_table_supports_gemma4_layer_experts() -> None:
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

    sim_table = build_model_similarity_table(model, mode="fast")

    assert 0 in sim_table
    assert sim_table[0].shape == (2, 2)
