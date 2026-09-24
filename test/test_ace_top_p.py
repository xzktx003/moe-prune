from __future__ import annotations

import torch

from moe_prune.code.src.ace_top_p import top_p_cutoff_for_pruned_count, top_p_truncate


def test_zero_tau_keeps_everything():
    score = torch.tensor([[0.5, 0.3, 0.2]])
    assert bool(top_p_truncate(score, 0.0).all().item())


def test_truncation_follows_score_order_not_position():
    score = torch.tensor([[0.1, 0.6, 0.3]])
    keep = top_p_truncate(score, 0.25)
    # Total mass is 1.0, so dropping the 0.1 tail leaves 0.9 covered.
    assert keep[0].tolist() == [False, True, True]


def test_highest_score_slot_is_always_kept():
    score = torch.tensor([[0.05, 0.9, 0.05]])
    keep = top_p_truncate(score, 0.99)
    assert bool(keep[0, 1].item())
    assert int(keep.sum().item()) == 1


def test_tokens_keep_different_numbers_of_slots():
    score = torch.tensor([[0.8, 0.2, 0.0], [0.34, 0.33, 0.33]])
    keep = top_p_truncate(score, 0.3)
    assert int(keep[0].sum().item()) == 1
    assert int(keep[1].sum().item()) == 3


def test_cutoff_roundtrips_to_target_prune_count():
    score = torch.tensor([0.4, 0.3, 0.2, 0.1])
    for pruned in (0, 1, 2, 3):
        tau = top_p_cutoff_for_pruned_count(score, pruned)
        kept = int(top_p_truncate(score.unsqueeze(0), tau).sum().item())
        assert kept == score.numel() - pruned
