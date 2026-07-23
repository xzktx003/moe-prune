from __future__ import annotations

from types import SimpleNamespace

import torch

from moe_prune.code.src.expert_similarity import (
    build_fast_channel_signature,
    build_fast_signatures_for_layer,
    build_layer_similarity_table,
    build_similarity_matrix,
    canonicalize_expert_weights,
)


def test_build_fast_signatures_for_layer_matches_loop_formula() -> None:
    torch.manual_seed(0)
    num_experts = 4
    hidden_size = 6
    inter_size = 3

    gamma = torch.randn(hidden_size, dtype=torch.float32)
    gate_up_proj = torch.randn(
        num_experts,
        inter_size * 2,
        hidden_size,
        dtype=torch.float32,
    )
    down_proj = torch.randn(
        num_experts,
        hidden_size,
        inter_size,
        dtype=torch.float32,
    )

    vectorized = build_fast_signatures_for_layer(
        gamma,
        gate_up_proj,
        down_proj,
    )

    loop_signatures = []
    for expert_idx in range(num_experts):
        E, B, G = canonicalize_expert_weights(
            gate_up_proj[expert_idx],
            down_proj[expert_idx],
        )
        loop_signatures.append(build_fast_channel_signature(gamma, E, B, G))
    loop_stacked = torch.stack(loop_signatures, dim=0)

    torch.testing.assert_close(vectorized, loop_stacked, rtol=1e-5, atol=1e-6)


def test_build_layer_similarity_table_fast_matches_loop_similarity() -> None:
    torch.manual_seed(1)
    num_experts = 3
    hidden_size = 5
    inter_size = 4

    gamma = torch.randn(hidden_size, dtype=torch.float32)
    gate_up_proj = torch.randn(
        num_experts,
        inter_size * 2,
        hidden_size,
        dtype=torch.float32,
    )
    down_proj = torch.randn(
        num_experts,
        hidden_size,
        inter_size,
        dtype=torch.float32,
    )
    moe_layer = SimpleNamespace(
        experts=SimpleNamespace(
            gate_up_proj=gate_up_proj,
            down_proj=down_proj,
        )
    )

    loop_signatures = []
    for expert_idx in range(num_experts):
        E, B, G = canonicalize_expert_weights(
            gate_up_proj[expert_idx],
            down_proj[expert_idx],
        )
        loop_signatures.append(build_fast_channel_signature(gamma, E, B, G))
    expected = build_similarity_matrix(torch.stack(loop_signatures, dim=0))

    actual = build_layer_similarity_table(moe_layer, gamma, mode="fast")

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
