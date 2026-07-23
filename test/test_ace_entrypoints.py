from __future__ import annotations

from moe_prune.code.scripts.shared import run_evalscope_eval
from moe_prune.code.scripts.shared.run_dual_view_eval import run
from moe_prune.code.src.quantile_collector import _candidate_scores_for_method

import torch


def test_ace_is_exposed_by_primary_entrypoints() -> None:
    assert "ace" in run_evalscope_eval.SUPPORTED_METHODS


def test_ace_quantile_candidates_exclude_gate_top1() -> None:
    gate = torch.tensor([[0.6, 0.3, 0.1]])
    p_final = torch.tensor([[0.2, 0.7, 0.1]])

    candidates = _candidate_scores_for_method("ace", p_final, gate)

    assert torch.equal(candidates, torch.tensor([0.7, 0.1]))


def test_legacy_dual_view_entry_is_callable() -> None:
    assert callable(run)