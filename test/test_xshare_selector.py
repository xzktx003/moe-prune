from __future__ import annotations

import torch

from moe_prune.code.src.xshare_selector import (
    build_xshare_batch_hot_expert_mask,
    build_xshare_keep_mask_from_hot_experts,
    build_xshare_keep_mask_from_topk_demand,
    xshare_budget_from_tau,
)


def test_xshare_budget_from_tau_clamps_to_valid_range() -> None:
    assert xshare_budget_from_tau(8, 0.0) == 8
    assert xshare_budget_from_tau(8, 0.5) == 4
    assert xshare_budget_from_tau(8, 0.99) == 1
    assert xshare_budget_from_tau(8, 2.0) == 1
    assert xshare_budget_from_tau(8, -1.0) == 8


def test_xshare_hot_expert_mask_uses_batch_aggregate_router_demand() -> None:
    router_logits = torch.tensor(
        [
            [5.0, 1.0, 0.0, 0.0],
            [4.0, 3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
        ]
    )

    hot_mask, demand = build_xshare_batch_hot_expert_mask(router_logits, tau=0.5)

    assert hot_mask.tolist() == [True, True, False, False]
    assert demand[0] > demand[2]
    assert demand[1] > demand[3]


def test_xshare_keep_mask_drops_cold_expert_slots_and_preserves_token_top1() -> None:
    selected = torch.tensor([[0, 2, 3], [3, 1, 0]])
    gate = torch.tensor([[0.60, 0.30, 0.10], [0.55, 0.35, 0.10]])
    hot_mask = torch.tensor([True, True, False, False])

    keep_mask, score = build_xshare_keep_mask_from_hot_experts(selected, gate, hot_mask)

    assert score.tolist() == [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert keep_mask.tolist() == [[True, False, False], [True, True, True]]


def test_xshare_topk_fallback_aggregates_routed_demand() -> None:
    selected = torch.tensor([[0, 2], [2, 1], [2, 3]])
    gate = torch.tensor([[0.7, 0.3], [0.8, 0.2], [0.6, 0.4]])

    keep_mask, score, demand = build_xshare_keep_mask_from_topk_demand(
        selected,
        gate,
        tau=0.5,
        num_experts=4,
    )

    assert demand.argmax().item() == 2
    assert score.tolist() == [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    assert keep_mask.tolist() == [[True, True], [True, False], [True, False]]
