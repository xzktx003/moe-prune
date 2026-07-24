"""Tests for code/src/quantile_search.py."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from moe_prune.code.src.quantile_search import (
    QUANTILE_COMPATIBLE_METHODS,
    build_candidate_scores_from_p_final,
    build_global_threshold_table_by_quantile,
    candidate_collection_cache_matches,
    load_candidate_collection_meta,
    load_candidate_collection_chunks,
    fast_quantile_by_kthvalue,
    threshold_for_pruned_count,
    write_candidate_collection_meta,
)


def pytest_approx(value):
    return pytest.approx(value, abs=1e-6)


def test_quantile_compatible_methods_contract():
    assert set(QUANTILE_COMPATIBLE_METHODS) == {
        "ace",
        "method1",
        "method4",
        "score_only",
        "naee",
        "aimer",
        "expert_sparsity",
        "top_p",
        "top_p_aimer",
        "entropy_dynamic_k",
        "sere",
    }


def test_build_candidate_scores_excludes_top1():
    p_final = torch.tensor([[0.5, 0.2, 0.1, 0.1, 0.1]])
    candidate_scores, candidate_mask = build_candidate_scores_from_p_final(p_final, min_keep=1)
    assert candidate_mask.shape == p_final.shape
    assert not candidate_mask[0, 0].item()
    assert candidate_mask[0, 1:].all().item()
    assert candidate_scores.tolist() == [pytest_approx(v) for v in [0.2, 0.1, 0.1, 0.1]]


def test_build_candidate_scores_min_keep_top_n():
    p_final = torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]])
    candidate_scores, candidate_mask = build_candidate_scores_from_p_final(p_final, min_keep=3)
    expected_mask = torch.tensor([[False, False, False, True, True]])
    assert torch.equal(candidate_mask, expected_mask)
    assert candidate_scores.tolist() == [pytest_approx(v) for v in [0.1, 0.05]]


def test_fast_quantile_kthvalue_matches_sorted_lookup():
    scores = torch.tensor([0.9, 0.1, 0.5, 0.7, 0.3])
    sorted_scores = torch.sort(scores).values
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        n = sorted_scores.numel()
        rank = int(q * (n - 1)) + 1
        rank = max(1, min(rank, n))
        expected = sorted_scores[rank - 1]
        got = fast_quantile_by_kthvalue(scores, q)
        assert math.isclose(float(got), float(expected), rel_tol=0, abs_tol=1e-6), (q, got, expected)


def test_build_global_threshold_table_monotone_in_rate():
    torch.manual_seed(7)
    cache = {
        0: {"p_final": torch.rand(64, 8)},
        1: {"p_final": torch.rand(48, 8)},
        2: {"p_final": torch.rand(72, 8)},
    }
    table, stats = build_global_threshold_table_by_quantile(
        layer_score_cache=cache,
        target_rates=(0.05, 0.2, 0.5, 0.9),
        min_keep=1,
    )
    rates = sorted(table.keys())
    tau_values = [table[r] for r in rates]
    assert tau_values == sorted(tau_values), tau_values
    assert all(stats[r]["total_candidates"] == sum(c["p_final"].shape[0] * 7 for c in cache.values()) for r in rates)
    assert all(stats[r]["achieved_global_slot_prune_rate"] <= r + 0.02 for r in rates)


def test_threshold_for_pruned_count_hits_exact_count_with_distinct_boundary():
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    tau, stats = threshold_for_pruned_count(scores, pruned_count=2)
    assert stats["achieved_pruned_candidates"] == 2
    assert 0.2 < float(tau) < 0.3


def test_threshold_for_pruned_count_picks_closest_reachable_count_for_ties():
    scores = torch.tensor([0.1, 0.1, 0.2, 0.3])
    tau, stats = threshold_for_pruned_count(scores, pruned_count=1)
    assert stats["desired_pruned_candidates"] == 1
    assert stats["achieved_pruned_candidates"] == 0
    assert stats["boundary_strategy"] == "boundary"
    assert float(tau) == pytest.approx(0.1, abs=1e-6)


def test_build_global_threshold_table_global_slot_rate_targets_runtime_metric():
    cache = {
        0: {"p_final": torch.tensor([[0.9, 0.1, 0.2, 0.3]])},
    }
    table, stats = build_global_threshold_table_by_quantile(
        layer_score_cache=cache,
        target_rates=[0.25, 0.50],
        min_keep=1,
        target_is_global_slot_rate=True,
    )
    assert 0.1 < table[0.25] < 0.2
    assert 0.2 < table[0.50] < 0.3
    assert stats[0.25]["desired_pruned_candidates"] == 1
    assert stats[0.25]["achieved_global_slot_prune_rate"] == pytest.approx(0.25, abs=1e-6)
    assert stats[0.50]["desired_pruned_candidates"] == 2
    assert stats[0.50]["achieved_global_slot_prune_rate"] == pytest.approx(0.50, abs=1e-6)


def test_build_global_threshold_table_skips_empty_layers():
    cache = {
        0: {"p_final": torch.empty(0, 8)},
        1: {"p_final": torch.rand(16, 8)},
    }
    table, _ = build_global_threshold_table_by_quantile(
        layer_score_cache=cache,
        target_rates=(0.5,),
        min_keep=1,
    )
    assert 0.5 in table


def test_load_candidate_collection_chunks_reads_chunked_payloads(tmp_path: Path):
    torch.save(
        {"candidate_scores": torch.tensor([0.1, 0.2]), "total_slots": 10},
        tmp_path / "chunk_000000.pt",
    )
    torch.save(
        {"candidate_scores": torch.tensor([0.3]), "total_slots": 6},
        tmp_path / "chunk_000001.pt",
    )

    global_candidates, total_slots, chunk_count = load_candidate_collection_chunks(tmp_path)

    assert chunk_count == 2
    assert total_slots == 16
    assert torch.equal(global_candidates, torch.tensor([0.1, 0.2, 0.3]))


def test_candidate_collection_meta_round_trip(tmp_path: Path):
    payload = {
        "collection_complete": True,
        "source": "evalscope_full_unpruned_generate",
        "method": "method1",
        "dataset": "arc-c",
    }
    write_candidate_collection_meta(tmp_path, payload)

    loaded = load_candidate_collection_meta(tmp_path)

    assert loaded == payload
    assert candidate_collection_cache_matches(
        tmp_path,
        {"source": "evalscope_full_unpruned_generate", "method": "method1", "dataset": "arc-c"},
    )
    assert not candidate_collection_cache_matches(
        tmp_path,
        {"source": "evalscope_full_unpruned_generate", "method": "method4", "dataset": "arc-c"},
    )
