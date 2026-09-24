"""Top-p truncation over the ACE score.

ACE still produces the per-slot score (``max(1.0 * normalized GSP, 0.1 *
normalized RCR)``). This module only decides how many of those scored slots to
keep: slots are taken in descending ACE-score order until their cumulative
score mass covers ``1 - tau``, so ``tau`` is the fraction of ACE score mass that
is dropped and different tokens keep different numbers of experts.
"""

from __future__ import annotations

import torch


def top_p_truncate(score: torch.Tensor, tau: float, eps: float = 1e-8) -> torch.Tensor:
    """Keep the smallest ACE-score prefix whose mass reaches ``1 - tau``.

    Slots are ordered by descending ACE score. A slot is kept while the score
    mass accumulated before it is still below ``1 - tau``, which also keeps the
    slot that crosses the cutoff. ``tau == 0`` keeps every slot and larger
    values prune more. The highest-scoring slot is always kept.
    """

    sorted_score, sorted_idx = torch.sort(score.float(), dim=-1, descending=True)
    total = sorted_score.sum(dim=-1, keepdim=True).clamp_min(eps)
    cumulative_before = (torch.cumsum(sorted_score, dim=-1) - sorted_score) / total
    keep_sorted = cumulative_before < (1.0 - float(tau))
    keep_sorted[:, 0] = True
    keep_mask = torch.zeros(score.shape, dtype=torch.bool, device=score.device)
    keep_mask.scatter_(dim=-1, index=sorted_idx, src=keep_sorted)
    return keep_mask


def top_p_cutoff_for_pruned_count(score: torch.Tensor, pruned_count: int) -> float:
    """Invert :func:`top_p_truncate` for a pooled score vector.

    Returns a ``tau`` that makes ``top_p_truncate`` drop ``pruned_count`` of the
    pooled slots when they are truncated together as one sequence. The value is
    placed halfway between the cutoff that drops exactly the desired tail and
    the next reachable cutoff, so re-running the truncation reproduces the
    target count instead of landing on a boundary.
    """

    flat = score.float().flatten()
    total = int(flat.numel())
    if total == 0:
        raise ValueError("top_p_cutoff_for_pruned_count requires at least one score")
    pruned_count = max(0, min(int(pruned_count), total))
    if pruned_count == 0:
        return 0.0
    sorted_score = torch.sort(flat, descending=True).values
    mass = float(sorted_score.sum().item())
    if mass <= 0.0:
        return 0.0
    kept = total - pruned_count
    tail_mass = float(sorted_score[kept:].sum().item()) / mass
    if kept == 0:
        return tail_mass
    boundary_mass = float(sorted_score[kept - 1].item()) / mass
    return float(tail_mass + 0.5 * boundary_mass)
