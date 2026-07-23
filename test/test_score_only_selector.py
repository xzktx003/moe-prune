from __future__ import annotations

import torch

from moe_prune.code.src.score_only_selector import build_score_only_keep_mask


def test_build_score_only_keep_mask_uses_absolute_gate_cutoff() -> None:
    gate = torch.tensor(
        [[0.60, 0.25, 0.15], [0.41, 0.40, 0.19]],
        dtype=torch.float32,
    )
    keep_mask = build_score_only_keep_mask(gate, gate_threshold=0.40)
    assert keep_mask.tolist() == [[True, False, False], [True, True, False]]


def test_build_score_only_keep_mask_always_keeps_top1() -> None:
    gate = torch.tensor([[0.39, 0.33, 0.28]], dtype=torch.float32)
    keep_mask = build_score_only_keep_mask(gate, gate_threshold=0.40)
    assert keep_mask.tolist() == [[True, False, False]]
