"""Tests for code/src/quantile_search.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moe_prune.code.src.quantile_search import (
    build_threshold_table_from_global_candidates,
    candidate_collection_cache_matches,
    load_candidate_collection_meta,
    load_candidate_collection_chunks,
    threshold_for_pruned_count,
    write_candidate_collection_meta,
)


def pytest_approx(value):
    return pytest.approx(value, abs=1e-6)


def test_build_threshold_table_monotone_in_rate():
    torch.manual_seed(7)
    candidates = torch.rand(128)
    table, stats = build_threshold_table_from_global_candidates(
        global_candidates=candidates,
        total_slots=160,
        target_rates=(0.05, 0.2, 0.5, 0.9),
    )
    rates = sorted(table.keys())
    tau_values = [table[r] for r in rates]
    assert tau_values == sorted(tau_values), tau_values
    assert all(stats[r]["total_candidates"] == candidates.numel() for r in rates)
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


def test_build_threshold_table_global_slot_rate_targets_runtime_metric():
    candidates = torch.tensor([0.1, 0.2, 0.3])
    table, stats = build_threshold_table_from_global_candidates(
        global_candidates=candidates,
        total_slots=4,
        target_rates=[0.25, 0.50],
        target_is_global_slot_rate=True,
    )
    assert 0.1 < table[0.25] < 0.2
    assert 0.2 < table[0.50] < 0.3
    assert stats[0.25]["desired_pruned_candidates"] == 1
    assert stats[0.25]["achieved_global_slot_prune_rate"] == pytest.approx(0.25, abs=1e-6)
    assert stats[0.50]["desired_pruned_candidates"] == 2
    assert stats[0.50]["achieved_global_slot_prune_rate"] == pytest.approx(0.50, abs=1e-6)


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
        "method": "ace",
        "dataset": "arc-c",
    }
    write_candidate_collection_meta(tmp_path, payload)

    loaded = load_candidate_collection_meta(tmp_path)

    assert loaded == payload
    assert candidate_collection_cache_matches(
        tmp_path,
        {"source": "evalscope_full_unpruned_generate", "method": "ace", "dataset": "arc-c"},
    )
    assert not candidate_collection_cache_matches(
        tmp_path,
        {"source": "different_source", "method": "ace", "dataset": "arc-c"},
    )
