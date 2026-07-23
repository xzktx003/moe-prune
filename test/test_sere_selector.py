from __future__ import annotations

import pytest
import torch

from moe_prune.code.src.sere_selector import (
    build_sere_dissimilarity_score,
    build_sere_keep_mask_and_gate,
)


def test_sere_dissimilarity_scores_against_primary_expert() -> None:
    selected = torch.tensor([[2, 4, 1], [3, 0, 2]])
    sim = torch.eye(5)
    sim[2, 4] = sim[4, 2] = 0.80
    sim[2, 1] = sim[1, 2] = 0.20
    sim[3, 0] = sim[0, 3] = 0.60
    sim[3, 2] = sim[2, 3] = 0.10

    score = build_sere_dissimilarity_score(selected, sim)

    assert score.tolist() == [
        [pytest.approx(1.0), pytest.approx(0.20), pytest.approx(0.80)],
        [pytest.approx(1.0), pytest.approx(0.40), pytest.approx(0.90)],
    ]


def test_sere_reroutes_similar_secondary_mass_to_primary() -> None:
    selected = torch.tensor([[2, 4, 1]])
    gate = torch.tensor([[0.50, 0.30, 0.20]])
    sim = torch.eye(5)
    sim[2, 4] = sim[4, 2] = 0.90
    sim[2, 1] = sim[1, 2] = 0.10

    keep_mask, score, gate_rerouted = build_sere_keep_mask_and_gate(
        selected,
        gate,
        sim,
        tau=0.50,
    )

    assert score.tolist() == [[pytest.approx(1.0), pytest.approx(0.10), pytest.approx(0.90)]]
    assert keep_mask.tolist() == [[True, False, True]]
    assert gate_rerouted.tolist() == [[pytest.approx(0.80), pytest.approx(0.0), pytest.approx(0.20)]]


def test_sere_keeps_primary_even_when_threshold_above_one() -> None:
    selected = torch.tensor([[2, 4, 1]])
    gate = torch.tensor([[0.50, 0.30, 0.20]])
    sim = torch.eye(5)

    keep_mask, _, gate_rerouted = build_sere_keep_mask_and_gate(
        selected,
        gate,
        sim,
        tau=1.5,
    )

    assert keep_mask.tolist() == [[True, False, False]]
    assert gate_rerouted.tolist() == [[pytest.approx(1.0), pytest.approx(0.0), pytest.approx(0.0)]]
