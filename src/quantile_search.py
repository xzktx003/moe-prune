"""Global ACE score collection and top-p cutoff calibration.

Uses a single ACE calibration forward followed by a pooled score-mass cutoff
lookup. Runtime applies the resulting ``tau`` to each token's ACE scores.
See ``scripts/shared/quantile_calibration.py`` for the calibration entry point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch

from .ace_top_p import top_p_cutoff_for_pruned_count


def _target_pruned_candidate_count(
    rate: float,
    total_slots: int,
    total_candidates: int,
    target_is_global_slot_rate: bool,
) -> int:
    if total_candidates <= 0:
        return 0
    base = float(rate) * (float(total_slots) if target_is_global_slot_rate else float(total_candidates))
    target = int(round(base))
    return max(0, min(target, int(total_candidates)))


def threshold_for_pruned_count(
    scores: torch.Tensor,
    pruned_count: int,
) -> tuple[torch.Tensor, Dict[str, float | str]]:
    """Choose ``tau`` to match ``keep_mask = score >= tau`` as closely as possible.

    The runtime rule prunes slots with ``score < tau``. This helper therefore
    selects a threshold value that targets an exact prune-count on the
    calibration distribution, not just a raw quantile rank. When ties at the
    boundary make the exact count impossible, it picks the closer of the two
    reachable counts and records the achieved count in ``stats``.
    """

    flat = scores.float().flatten()
    total = int(flat.numel())
    if total == 0:
        raise ValueError("threshold_for_pruned_count requires at least one score")

    pruned_count = max(0, min(int(pruned_count), total))
    sorted_scores = torch.sort(flat).values
    plus_inf = torch.tensor(float("inf"), device=sorted_scores.device, dtype=sorted_scores.dtype)

    if pruned_count == 0:
        tau = sorted_scores[0]
        return tau, {
            "desired_pruned_candidates": 0,
            "achieved_pruned_candidates": 0,
            "boundary_strategy": "keep-all-candidates",
        }

    if pruned_count == total:
        tau = torch.nextafter(sorted_scores[-1], plus_inf)
        return tau, {
            "desired_pruned_candidates": total,
            "achieved_pruned_candidates": total,
            "boundary_strategy": "prune-all-candidates",
        }

    lower = sorted_scores[pruned_count - 1]
    upper = sorted_scores[pruned_count]
    if float(lower.item()) < float(upper.item()):
        tau = (lower + upper) * 0.5
        return tau, {
            "desired_pruned_candidates": pruned_count,
            "achieved_pruned_candidates": pruned_count,
            "boundary_strategy": "midpoint",
        }

    boundary = lower
    lower_achieved = int(torch.searchsorted(sorted_scores, boundary, right=False).item())
    upper_achieved = int(torch.searchsorted(sorted_scores, boundary, right=True).item())

    if abs(upper_achieved - pruned_count) < abs(lower_achieved - pruned_count):
        tau = torch.nextafter(boundary, plus_inf)
        achieved = upper_achieved
        strategy = "nextafter-up"
    else:
        tau = boundary
        achieved = lower_achieved
        strategy = "boundary"

    return tau, {
        "desired_pruned_candidates": pruned_count,
        "achieved_pruned_candidates": achieved,
        "boundary_strategy": strategy,
    }


def build_threshold_table_from_global_candidates(
    global_candidates: torch.Tensor,
    total_slots: int,
    target_rates: Iterable[float],
    target_is_global_slot_rate: bool = True,
) -> tuple[Dict[float, float], Dict[float, Dict[str, object]]]:
    """Build thresholds directly from a pre-collected global candidate-score pool."""

    flat = global_candidates.float().flatten()
    total_candidates = int(flat.numel())
    if total_candidates <= 0:
        raise ValueError("No candidate scores collected; check runtime collection.")

    threshold_table: Dict[float, float] = {}
    stats_table: Dict[float, Dict[str, object]] = {}
    for raw_rate in target_rates:
        rate = float(raw_rate)
        desired_pruned_candidates = _target_pruned_candidate_count(
            rate=rate,
            total_slots=total_slots,
            total_candidates=total_candidates,
            target_is_global_slot_rate=target_is_global_slot_rate,
        )
        q = float(desired_pruned_candidates) / float(max(total_candidates, 1))
        # tau is the ACE score-mass fraction dropped by top-p truncation, so it
        # is inverted from the cumulative-mass rule rather than a score quantile.
        global_tau = top_p_cutoff_for_pruned_count(flat, desired_pruned_candidates)
        achieved_pruned_candidates = int(desired_pruned_candidates)
        threshold_table[rate] = float(global_tau)
        stats_table[rate] = {
            "candidate_quantile": float(q),
            "total_slots": int(total_slots),
            "total_candidates": int(total_candidates),
            "desired_pruned_candidates": int(desired_pruned_candidates),
            "achieved_pruned_candidates": int(achieved_pruned_candidates),
            "achieved_candidate_prune_rate": (
                float(achieved_pruned_candidates) / float(max(total_candidates, 1))
            ),
            "achieved_global_slot_prune_rate": (
                float(achieved_pruned_candidates) / float(max(total_slots, 1))
            ),
            "global_tau": float(global_tau),
            "target_is_global_slot_rate": bool(target_is_global_slot_rate),
            "boundary_strategy": "top-p-cumulative-mass",
        }
    return threshold_table, stats_table


def load_candidate_collection_chunks(
    chunks_dir: str | Path,
) -> tuple[torch.Tensor, int, int]:
    """Load chunked candidate-score dumps written by evalscope quantile collection."""

    root = Path(chunks_dir)
    chunk_paths = sorted(root.glob("chunk_*.pt"))
    if not chunk_paths:
        raise ValueError(f"No quantile collection chunks found in {root}")

    candidate_parts = []
    total_slots = 0
    for path in chunk_paths:
        payload = torch.load(path, map_location="cpu")
        candidate_scores = payload.get("candidate_scores")
        if candidate_scores is None:
            continue
        flat = candidate_scores.float().flatten()
        if flat.numel() > 0:
            candidate_parts.append(flat)
        total_slots += int(payload.get("total_slots", 0))

    if not candidate_parts:
        raise ValueError(f"No candidate scores found across quantile collection chunks in {root}")

    global_candidates = torch.cat(candidate_parts, dim=0)
    return global_candidates, int(total_slots), len(chunk_paths)


def collection_meta_path(chunks_dir: str | Path) -> Path:
    return Path(chunks_dir) / "collection_meta.json"


def write_candidate_collection_meta(chunks_dir: str | Path, payload: Mapping[str, object]) -> Path:
    root = Path(chunks_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = collection_meta_path(root)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_candidate_collection_meta(chunks_dir: str | Path) -> Dict[str, object] | None:
    path = collection_meta_path(chunks_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def candidate_collection_cache_matches(
    chunks_dir: str | Path,
    expected: Mapping[str, object],
) -> bool:
    payload = load_candidate_collection_meta(chunks_dir)
    if not payload or not payload.get("collection_complete", False):
        return False
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return True


__all__ = [
    "build_threshold_table_from_global_candidates",
    "candidate_collection_cache_matches",
    "load_candidate_collection_chunks",
    "load_candidate_collection_meta",
    "threshold_for_pruned_count",
    "write_candidate_collection_meta",
]
# End of ACE quantile API.


