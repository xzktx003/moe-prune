from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Dict, List, Mapping, Optional

import torch

from .amp_proxy import split_gate_up_proj
from .model_structure import iter_moe_layer_bindings
from .runtime_pruner import (
    RuntimeStats,
    compute_moe_weighted_hidden_states,
    compute_optional_shared_expert_output,
    renorm_gate_after_pruning,
    route_qwen3_topk,
)


def compute_aimer_removal_score(
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """AIMER calibration-free expert score.

    Original AIMER ranks larger values as more removable for static pruning.
    This repo uses the score as a source for a dynamic-skipping baseline, so
    ``build_aimer_keep_table_for_model`` inverts the normalized score before
    combining it with token-level router probabilities.
    """

    tensors = (
        gate_proj_weight.detach().float(),
        up_proj_weight.detach().float(),
        down_proj_weight.detach().float(),
    )
    abs_sum = sum(weight.abs().sum() for weight in tensors)
    l2_sq = sum(weight.square().sum() for weight in tensors)
    numel = sum(weight.numel() for weight in tensors)
    mean_abs = abs_sum / float(numel)
    rms = torch.sqrt(l2_sq / float(numel) + eps)
    return mean_abs / (rms + eps)


def build_aimer_keep_table_for_layer(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    removal_scores: List[torch.Tensor] = []
    for expert_idx in range(gate_up_proj.shape[0]):
        gate_proj_weight, up_proj_weight = split_gate_up_proj(gate_up_proj[expert_idx])
        removal_scores.append(
            compute_aimer_removal_score(
                gate_proj_weight=gate_proj_weight,
                up_proj_weight=up_proj_weight,
                down_proj_weight=down_proj[expert_idx],
                eps=eps,
            )
        )

    removal = torch.stack(removal_scores)
    normalized_removal = removal / (removal.mean() + eps)
    keep = 1.0 / (normalized_removal + eps)
    return keep / (keep.mean() + eps)


@torch.no_grad()
def build_aimer_keep_table_for_model(model, eps: float = 1e-8) -> Dict[int, torch.Tensor]:
    keep_table: Dict[int, torch.Tensor] = {}
    for binding in iter_moe_layer_bindings(model):
        layer_idx = binding.layer_idx
        experts = binding.experts
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            keep_table[layer_idx] = build_aimer_keep_table_for_layer(
                gate_up_proj=experts.gate_up_proj.detach(),
                down_proj=experts.down_proj.detach(),
                eps=eps,
            ).cpu()
            continue

        removal_scores: List[torch.Tensor] = []
        for expert_layer in experts:
            removal_scores.append(
                compute_aimer_removal_score(
                    gate_proj_weight=expert_layer.gate_proj.weight.detach(),
                    up_proj_weight=expert_layer.up_proj.weight.detach(),
                    down_proj_weight=expert_layer.down_proj.weight.detach(),
                    eps=eps,
                )
            )
        removal = torch.stack(removal_scores)
        normalized_removal = removal / (removal.mean() + eps)
        keep = 1.0 / (normalized_removal + eps)
        keep_table[layer_idx] = (keep / (keep.mean() + eps)).cpu()
    return keep_table


def build_aimer_keep_mask(
    gate: torch.Tensor,
    keep_selected: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    score = gate.float() * keep_selected.to(device=gate.device, dtype=torch.float32)
    p_final = score / (score.sum(dim=-1, keepdim=True) + eps)
    keep_mask = p_final >= float(tau)
    best_idx = p_final.argmax(dim=-1, keepdim=True)
    keep_mask.scatter_(dim=-1, index=best_idx, value=True)
    return keep_mask, p_final


def moe_forward_with_aimer_selection(
    hidden_states: torch.Tensor,
    router,
    experts,
    keep_layer: torch.Tensor,
    tau: float,
    top_k: int = 8,
    norm_topk_prob: bool = True,
    eps: float = 1e-8,
    return_aux: bool = False,
    moe_backend: str = "triton",
    shared_expert=None,
    shared_expert_gate=None,
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat_states = hidden_states.view(-1, hidden_dim)

    _, gate, selected_experts = route_qwen3_topk(
        router,
        flat_states,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
    )
    keep_selected = keep_layer.to(device=gate.device)[selected_experts]
    keep_mask, p_final = build_aimer_keep_mask(gate, keep_selected, tau=tau, eps=eps)
    gate_kept = renorm_gate_after_pruning(gate, keep_mask, eps=eps)
    final_hidden, expert_outputs, full_out = compute_moe_weighted_hidden_states(
        flat_states,
        experts,
        selected_experts,
        gate_kept,
        keep_mask=keep_mask,
        moe_backend=moe_backend,
    )
    shared_output = compute_optional_shared_expert_output(
        flat_states,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
    )
    if shared_output is not None:
        final_hidden = final_hidden + shared_output
    final_hidden = final_hidden.reshape(batch_size, sequence_length, hidden_dim)

    aux = {
        "topk_idx": selected_experts.detach(),
        "gate": gate.detach(),
        "keep_selected": keep_selected.detach(),
        "p_final": p_final.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


def moe_experts_forward_with_aimer_selection(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    keep_layer: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
    return_aux: bool = False,
    moe_backend: str = "triton",
):
    keep_selected = keep_layer.to(device=routing_weights.device)[selected_experts]
    keep_mask, p_final = build_aimer_keep_mask(routing_weights, keep_selected, tau=tau, eps=eps)
    gate_kept = renorm_gate_after_pruning(routing_weights, keep_mask, eps=eps)
    final_hidden, expert_outputs, full_out = compute_moe_weighted_hidden_states(
        hidden_states,
        experts,
        selected_experts,
        gate_kept,
        keep_mask=keep_mask,
        moe_backend=moe_backend,
    )

    aux = {
        "topk_idx": selected_experts.detach(),
        "gate": routing_weights.detach(),
        "keep_selected": keep_selected.detach(),
        "p_final": p_final.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


@contextmanager
def patch_qwen3_moe_blocks_aimer(
    model,
    keep_table: Mapping[int, torch.Tensor],
    tau_by_layer: Mapping[int, float],
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    originals: List[tuple[object, object]] = []

    for binding in iter_moe_layer_bindings(model):
        layer_idx = binding.layer_idx
        if layer_idx not in keep_table:
            continue
        if binding.kind == "mlp" and binding.router is None:
            continue
        patch_target = binding.patch_target
        original_forward = patch_target.forward
        keep_layer = keep_table[layer_idx]
        tau = float(tau_by_layer.get(layer_idx, 0.0))

        if binding.kind == "mlp":
            top_k = binding.top_k
            norm_topk_prob = binding.norm_topk_prob

            def _forward(
                self,
                hidden_states,
                _layer_idx=layer_idx,
                _keep_layer=keep_layer,
                _tau=tau,
                _top_k=top_k,
                _norm_topk_prob=norm_topk_prob,
            ):
                output, aux = moe_forward_with_aimer_selection(
                    hidden_states=hidden_states,
                    router=self.gate,
                    experts=self.experts,
                    keep_layer=_keep_layer,
                    tau=_tau,
                    top_k=_top_k,
                    norm_topk_prob=_norm_topk_prob,
                    moe_backend=moe_backend,
                    shared_expert=getattr(self, "shared_expert", None),
                    shared_expert_gate=getattr(self, "shared_expert_gate", None),
                    return_aux=True,
                )
                if runtime_stats is not None:
                    runtime_stats.update(_layer_idx, aux["keep_mask"])
                return output
        else:
            def _forward(
                self,
                hidden_states,
                top_k_index,
                top_k_weights,
                _layer_idx=layer_idx,
                _keep_layer=keep_layer,
                _tau=tau,
            ):
                output, aux = moe_experts_forward_with_aimer_selection(
                    hidden_states=hidden_states,
                    experts=self,
                    selected_experts=top_k_index,
                    routing_weights=top_k_weights,
                    keep_layer=_keep_layer,
                    tau=_tau,
                    moe_backend=moe_backend,
                    return_aux=True,
                )
                if runtime_stats is not None:
                    runtime_stats.update(_layer_idx, aux["keep_mask"])
                return output

        originals.append((patch_target, original_forward))
        patch_target.forward = MethodType(_forward, patch_target)

    try:
        yield model
    finally:
        for patch_target, original_forward in originals:
            patch_target.forward = original_forward
