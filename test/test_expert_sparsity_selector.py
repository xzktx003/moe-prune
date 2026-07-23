from __future__ import annotations

import torch

from moe_prune.code.src.expert_sparsity_selector import (
    build_expert_sparsity_keep_mask,
    build_expert_sparsity_ratio_score,
)


def test_expert_sparsity_ratio_score_normalizes_by_top1() -> None:
    gate = torch.tensor([[0.5, 0.25, 0.125], [0.1, 0.4, 0.2]])

    score = build_expert_sparsity_ratio_score(gate)

    assert torch.allclose(
        score,
        torch.tensor([[1.0, 0.5, 0.25], [0.25, 1.0, 0.5]]),
        atol=1e-6,
    )


def test_expert_sparsity_keep_mask_keeps_ratio_above_beta_and_top1() -> None:
    gate = torch.tensor([[0.5, 0.24, 0.10], [0.1, 0.4, 0.19]])

    keep_mask, score = build_expert_sparsity_keep_mask(gate, beta=0.5)

    assert torch.allclose(
        score,
        torch.tensor([[1.0, 0.48, 0.2], [0.25, 1.0, 0.475]]),
        atol=1e-6,
    )
    assert torch.equal(
        keep_mask,
        torch.tensor([[True, False, False], [False, True, False]]),
    )


def test_expert_sparsity_keep_mask_preserves_top1_when_beta_gt_one() -> None:
    gate = torch.tensor([[0.2, 0.7, 0.1]])

    keep_mask, _ = build_expert_sparsity_keep_mask(gate, beta=1.5)

    assert torch.equal(keep_mask, torch.tensor([[False, True, False]]))
