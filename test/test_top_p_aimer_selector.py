from __future__ import annotations

import pytest
import torch

from moe_prune.code.src.quantile_search import build_candidate_scores_from_p_final
from moe_prune.code.code_v2.top_p_aimer_selector import (
    build_top_p_aimer_keep_mask,
    build_top_p_aimer_score,
)


def test_top_p_aimer_score_combines_normalized_top_p_and_aimer_views() -> None:
    gate = torch.tensor([[0.50, 0.25, 0.15, 0.10]])
    keep_selected = torch.tensor([[1.0, 3.0, 0.5, 2.0]])

    combined, aux = build_top_p_aimer_score(gate, keep_selected)

    assert combined[0, 0].item() == pytest.approx(1.0)
    assert aux["aimer_score"][0, 1] > aux["top_p_normalized_score"][0, 1]
    assert combined[0, 1].item() == pytest.approx(aux["aimer_score"][0, 1].item())
    assert combined[0, 2].item() == pytest.approx(aux["top_p_normalized_score"][0, 2].item())


def test_top_p_aimer_keep_mask_keeps_top1_and_high_score_candidates() -> None:
    gate = torch.tensor([[0.50, 0.25, 0.15, 0.10]])
    keep_selected = torch.tensor([[1.0, 3.0, 0.5, 2.0]])

    keep_mask, aux = build_top_p_aimer_keep_mask(gate, keep_selected, tau=0.20)

    assert torch.equal(keep_mask, torch.tensor([[True, True, False, False]]))
    assert aux["combined_score"][0, 0].item() == pytest.approx(1.0)


def test_top_p_aimer_keep_mask_preserves_top1_even_above_score_range() -> None:
    gate = torch.tensor([[0.50, 0.25, 0.15, 0.10]])
    keep_selected = torch.ones_like(gate)

    keep_mask, _ = build_top_p_aimer_keep_mask(gate, keep_selected, tau=1.5)

    assert torch.equal(keep_mask, torch.tensor([[True, False, False, False]]))


def test_top_p_aimer_quantile_candidates_exclude_forced_top1() -> None:
    gate = torch.tensor([[0.50, 0.25, 0.15, 0.10]])
    keep_selected = torch.tensor([[1.0, 3.0, 0.5, 2.0]])

    combined, _ = build_top_p_aimer_score(gate, keep_selected)
    candidate_scores, candidate_mask = build_candidate_scores_from_p_final(combined)

    assert torch.equal(candidate_mask, torch.tensor([[False, True, True, True]]))
    assert candidate_scores.tolist() == pytest.approx(combined[0, 1:].tolist())

