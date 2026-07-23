"""Quantile-based global threshold search (see docs/prd/快速搜索阈值方案.md).

Replaces multi-round binary search with a single calibration forward followed
by a one-shot global score quantile lookup. Only applicable to methods whose
runtime decision is ``p_final >= tau`` against a per-slot scalar score with a
shared, network-wide ``tau``. See ``code/scripts/shared/quantile_calibration.py``
for the calibration entry point and ``run_ppl_search.py`` for the search wiring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import torch


QUANTILE_COMPATIBLE_METHODS: Tuple[str, ...] = (
    "ace",
    "gsp",
    "rcr",
    "score_only",
    "naee",
    "aimer",
    "expert_sparsity",
    "top_p",
    "sere",
)


def build_candidate_scores_from_p_final(
    p_final: torch.Tensor,
    min_keep: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(candidate_scores, candidate_mask)`` for one layer's ``p_final``.

    The runtime force-keeps the top-``max(min_keep, 1)`` slots ranked by
    ``p_final`` (which equals the score's argmax for every supported method,
    including GSP/RCR/NAEE where p_final is built from
    ``gate * amp_selected`` and column 0 is *not* guaranteed to be the max).
    This helper mirrors that selection so the calibration candidate pool is
    aligned with what the runtime actually treats as prunable.
    """

    if p_final.ndim != 2:
        raise ValueError(f"p_final must be 2D [T, k], got shape {tuple(p_final.shape)}")

    keep_count = max(int(min_keep), 1)
    keep_count = min(keep_count, p_final.shape[-1])
    top_keep_idx = torch.topk(p_final, k=keep_count, dim=-1).indices
    forced_keep = torch.zeros_like(p_final, dtype=torch.bool)
    forced_keep.scatter_(dim=-1, index=top_keep_idx, value=True)

    candidate_mask = ~forced_keep
    candidate_scores = p_final[candidate_mask]
    return candidate_scores, candidate_mask


def fast_quantile_by_kthvalue(scores: torch.Tensor, q: float) -> torch.Tensor:
    """Compute a 1D quantile via ``torch.kthvalue`` (no full sort).

    Returns the ``q``-th quantile as a 0-D tensor; the score at rank
    ``floor(q*(n-1)) + 1`` (1-indexed).
    """

    flat = scores.float().flatten()
    n = flat.numel()
    if n == 0:
        raise ValueError("fast_quantile_by_kthvalue requires at least one score")

    q = float(min(max(q, 0.0), 1.0))
    rank = int(q * (n - 1)) + 1
    rank = max(1, min(rank, n))
    return torch.kthvalue(flat, rank).values


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
        global_tau, threshold_stats = threshold_for_pruned_count(flat, desired_pruned_candidates)
        achieved_pruned_candidates = int(threshold_stats["achieved_pruned_candidates"])
        threshold_table[rate] = float(global_tau.item())
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
            "global_tau": float(global_tau.item()),
            "target_is_global_slot_rate": bool(target_is_global_slot_rate),
            "boundary_strategy": str(threshold_stats["boundary_strategy"]),
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


def build_global_threshold_table_by_quantile(
    layer_score_cache: Mapping[int, Mapping[str, torch.Tensor]],
    target_rates: Iterable[float],
    min_keep: int = 1,
    target_is_global_slot_rate: bool = True,
) -> tuple[Dict[float, float], Dict[float, Dict[str, object]]]:
    """Generate per-rate global thresholds from a calibration ``p_final`` cache.

    ``layer_score_cache[layer_id]["p_final"]`` must be a ``[T_l, k]`` tensor of
    the same per-slot score used by the runtime decision rule.
    """

    all_candidate_scores = []
    total_slots = 0
    total_candidates = 0

    for cache in layer_score_cache.values():
        p_final = cache["p_final"]
        if not torch.is_tensor(p_final):
            raise TypeError("layer_score_cache entries must contain torch tensors")
        if p_final.numel() == 0:
            continue

        candidate_scores, _ = build_candidate_scores_from_p_final(
            p_final=p_final,
            min_keep=min_keep,
        )
        if candidate_scores.numel() > 0:
            all_candidate_scores.append(candidate_scores.float().flatten())

        total_slots += int(p_final.numel())
        total_candidates += int(candidate_scores.numel())

    if not all_candidate_scores:
        raise ValueError("No candidate scores collected; check min_keep / top-k.")

    global_candidates = torch.cat(all_candidate_scores, dim=0)

    return build_threshold_table_from_global_candidates(
        global_candidates=global_candidates,
        total_slots=total_slots,
        target_rates=target_rates,
        target_is_global_slot_rate=target_is_global_slot_rate,
    )
