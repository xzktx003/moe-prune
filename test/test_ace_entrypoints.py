from __future__ import annotations

from moe_prune.code.src.quantile_collector import ace_candidate_scores
from moe_prune.code.src.runtime_pruner import (
    ACE_GSP_COEFFICIENT,
    ACE_RCR_COEFFICIENT,
    combine_ace_scores,
)

import torch


def test_ace_quantile_candidates_exclude_gate_top1() -> None:
    gate = torch.tensor([[0.6, 0.3, 0.1]])
    p_final = torch.tensor([[0.2, 0.7, 0.1]])

    candidates = ace_candidate_scores(p_final, gate)

    assert torch.equal(candidates, torch.tensor([0.7, 0.1]))


def test_ace_scales_gsp_and_rcr_before_maximum() -> None:
    p_gsp = torch.tensor([[0.08, 0.04, 0.02]])
    p_rcr = torch.tensor([[0.20, 0.90, 0.30]])

    combined = combine_ace_scores(p_gsp, p_rcr)

    assert ACE_GSP_COEFFICIENT == 1.0
    assert ACE_RCR_COEFFICIENT == 0.1
    assert torch.allclose(combined, torch.tensor([[0.08, 0.09, 0.03]]))