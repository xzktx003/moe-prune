from __future__ import annotations

import torch

from moe_prune.code.src.top_p_selector import (
    build_top_p_keep_mask,
    build_top_p_residual_score,
)


def test_top_p_residual_score_uses_router_mass_before_each_expert() -> None:
    gate = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])

    score = build_top_p_residual_score(gate)

    assert torch.allclose(
        score,
        torch.tensor([[1.0, 0.5, 0.2], [0.1, 1.0, 0.3]]),
        atol=1e-6,
    )


def test_top_p_keep_mask_keeps_prefix_until_residual_threshold() -> None:
    gate = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])

    keep_mask, score = build_top_p_keep_mask(gate, tau=0.25)

    assert torch.allclose(
        score,
        torch.tensor([[1.0, 0.5, 0.2], [0.1, 1.0, 0.3]]),
        atol=1e-6,
    )
    assert torch.equal(
        keep_mask,
        torch.tensor([[True, True, False], [False, True, True]]),
    )


def test_top_p_keep_mask_preserves_top1_when_tau_gt_one() -> None:
    gate = torch.tensor([[0.2, 0.7, 0.1]])

    keep_mask, _ = build_top_p_keep_mask(gate, tau=1.5)

    assert torch.equal(keep_mask, torch.tensor([[False, True, False]]))
