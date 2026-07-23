from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

from .runtime_pruner import RuntimeStats, build_keep_mask_topk, compute_importance_score, renorm_gate_after_pruning


@dataclass
class ProxyMetrics:
    gate_corr: float
    amp_corr: float
    proxy_corr: float


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).item()
    if denom <= eps:
        return 0.0
    return float(torch.dot(x, y).item() / (denom + eps))


def proxy_validity_metrics(layer_cache: Dict[str, torch.Tensor], amp_layer: torch.Tensor) -> ProxyMetrics:
    topk_idx = layer_cache["topk_idx"].long()
    gate = layer_cache["gate"].float()
    expert_outs = layer_cache["expert_outs"].float()

    amp_selected = amp_layer[topk_idx].float()
    proxy = compute_importance_score(gate, amp_selected)
    actual = gate * torch.norm(expert_outs, dim=-1)

    return ProxyMetrics(
        gate_corr=_pearson_corr(gate, actual),
        amp_corr=_pearson_corr(amp_selected, actual),
        proxy_corr=_pearson_corr(proxy, actual),
    )


def rel_mse_for_tau(
    layer_cache: Dict[str, torch.Tensor],
    amp_layer: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
) -> Dict[str, float]:
    topk_idx = layer_cache["topk_idx"].long()
    gate = layer_cache["gate"].float()
    expert_outs = layer_cache["expert_outs"].float()
    full_out = layer_cache["full_out"].float()

    amp_selected = amp_layer[topk_idx].float()
    score = compute_importance_score(gate, amp_selected)
    keep_mask = build_keep_mask_topk(score, tau=tau, eps=eps)
    gate_kept = renorm_gate_after_pruning(gate, keep_mask, eps=eps)
    pred_out = (gate_kept.unsqueeze(-1) * expert_outs).sum(dim=1)
    rel_mse = (
        ((pred_out - full_out).pow(2).sum(dim=-1))
        / (full_out.pow(2).sum(dim=-1) + eps)
    ).mean().item()
    pruning_ratio = 1.0 - keep_mask.float().mean().item()
    avg_active_experts = keep_mask.sum(dim=-1).float().mean().item()
    return {
        "tau": float(tau),
        "rel_mse": float(rel_mse),
        "pruning_ratio": float(pruning_ratio),
        "avg_active_experts": float(avg_active_experts),
    }


def summarize_runtime_stats(runtime_stats: RuntimeStats) -> Dict[str, float]:
    return {
        "mean_pruning_ratio": runtime_stats.mean_pruning_ratio(),
        "layers_with_stats": float(len(runtime_stats.layers)),
    }

