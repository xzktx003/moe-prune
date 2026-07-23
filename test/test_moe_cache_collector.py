from __future__ import annotations

import torch

from moe_prune.code.src.moe_cache_collector import route_qwen3_topk


def test_route_qwen3_topk_accepts_three_tensor_router_output() -> None:
    def _router(hidden_states: torch.Tensor):
        logits = torch.tensor([[3.0, 1.0, 0.0]], dtype=hidden_states.dtype)
        weights = torch.tensor([[0.84, 0.11]], dtype=hidden_states.dtype)
        experts = torch.tensor([[0, 1]], dtype=torch.long)
        return logits, weights, experts

    hidden = torch.zeros((1, 3), dtype=torch.float32)
    logits, weights, experts = route_qwen3_topk(_router, hidden, top_k=2, norm_topk_prob=True)

    assert logits.shape == (1, 3)
    assert torch.allclose(weights, torch.tensor([[0.84, 0.11]], dtype=torch.float32))
    assert experts.tolist() == [[0, 1]]


def test_route_qwen3_topk_accepts_extended_tuple_router_output() -> None:
    def _router(hidden_states: torch.Tensor):
        del hidden_states
        logits = torch.tensor([[0.1, 1.2, 0.3]], dtype=torch.float32)
        return logits, torch.tensor([1.0]), {"ignored": True}, "extra"

    hidden = torch.zeros((1, 3), dtype=torch.float32)
    logits, weights, experts = route_qwen3_topk(_router, hidden, top_k=2, norm_topk_prob=True)

    assert torch.allclose(logits, torch.tensor([[0.1, 1.2, 0.3]], dtype=torch.float32))
    assert experts.tolist() == [[1, 2]]
    assert weights.shape == (1, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))
