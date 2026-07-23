from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch

from .runtime_pruner import build_keep_mask_topk, compute_importance_score, renorm_gate_after_pruning


def search_tau_topk(
    layer_cache: Dict[str, torch.Tensor],
    amp_layer: torch.Tensor,
    tau_grid: Iterable[float],
    err_budget: Optional[float],
    eps: float = 1e-8,
) -> Dict[str, float]:
    topk_idx = layer_cache["topk_idx"].long()
    gate = layer_cache["gate"].float()
    expert_outs = layer_cache["expert_outs"].float()
    full_out = layer_cache["full_out"].float()

    amp_selected = amp_layer[topk_idx].float()
    score = compute_importance_score(gate, amp_selected)

    best: Optional[Dict[str, float]] = None
    for tau in tau_grid:
        keep_mask = build_keep_mask_topk(score, tau=tau, eps=eps)
        gate_kept = renorm_gate_after_pruning(gate, keep_mask, eps=eps)
        pred_out = (gate_kept.unsqueeze(-1) * expert_outs).sum(dim=1)

        rel_mse = (
            ((pred_out - full_out).pow(2).sum(dim=-1))
            / (full_out.pow(2).sum(dim=-1) + eps)
        ).mean().item()
        pruning_ratio = 1.0 - keep_mask.float().mean().item()
        avg_active_experts = keep_mask.sum(dim=-1).float().mean().item()

        candidate = {
            "tau": float(tau),
            "rel_mse": float(rel_mse),
            "pruning_ratio": float(pruning_ratio),
            "avg_active_experts": float(avg_active_experts),
        }
        if err_budget is not None and rel_mse > err_budget:
            continue
        if best is None or candidate["pruning_ratio"] > best["pruning_ratio"]:
            best = candidate

    if best is None:
        return {
            "tau": float(list(tau_grid)[0]),
            "rel_mse": float("inf"),
            "pruning_ratio": 0.0,
            "avg_active_experts": float(topk_idx.shape[1]),
        }
    return best


def search_all_layers_tau(
    layer_caches: Dict[int, Dict[str, torch.Tensor]],
    amp_table: Dict[int, torch.Tensor],
    tau_grid: Iterable[float],
    err_budget: Optional[float],
    eps: float = 1e-8,
) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    for layer_idx, layer_cache in layer_caches.items():
        if layer_idx not in amp_table:
            continue
        results[layer_idx] = search_tau_topk(
            layer_cache=layer_cache,
            amp_layer=amp_table[layer_idx],
            tau_grid=tau_grid,
            err_budget=err_budget,
            eps=eps,
        )
    return results

