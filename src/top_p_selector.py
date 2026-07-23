from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Dict, Optional

import torch

from .model_structure import iter_moe_layer_bindings
from .runtime_pruner import (
    RuntimeStats,
    compute_moe_weighted_hidden_states,
    compute_optional_shared_expert_output,
    renorm_gate_after_pruning,
    route_qwen3_topk,
)


def build_top_p_residual_score(gate: torch.Tensor) -> torch.Tensor:
    """Score each routed expert by the residual router mass before selecting it.

    This is a runtime-skipping adaptation of top-p / dynamic top-p routing:
    experts are considered in descending router probability order and retained
    while the residual mass before the expert is at least ``tau``. The top expert
    is always retained by downstream masking code.
    """

    sorted_gate, sorted_idx = torch.sort(gate.float(), dim=-1, descending=True)
    cumulative_before = torch.cumsum(sorted_gate, dim=-1) - sorted_gate
    residual_sorted = 1.0 - cumulative_before
    residual = torch.empty_like(residual_sorted)
    residual.scatter_(dim=-1, index=sorted_idx, src=residual_sorted)
    return residual


def build_top_p_keep_mask(
    gate: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_score = build_top_p_residual_score(gate)
    keep_mask = residual_score >= float(tau)
    top1_idx = gate.argmax(dim=-1, keepdim=True)
    keep_mask.scatter_(dim=-1, index=top1_idx, value=True)
    return keep_mask, residual_score


def moe_forward_with_top_p_selection(
    hidden_states: torch.Tensor,
    router,
    experts,
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
    keep_mask, residual_score = build_top_p_keep_mask(gate, tau=tau)
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
        "residual_score": residual_score.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


def moe_experts_forward_with_top_p_selection(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
    return_aux: bool = False,
    moe_backend: str = "triton",
):
    keep_mask, residual_score = build_top_p_keep_mask(routing_weights, tau=tau)
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
        "residual_score": residual_score.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


@contextmanager
def patch_qwen3_moe_blocks_top_p(
    model,
    tau_by_layer: Dict[int, float],
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    originals = []

    for binding in iter_moe_layer_bindings(model):
        layer_idx = binding.layer_idx
        if binding.kind == "mlp" and binding.router is None:
            continue
        patch_target = binding.patch_target
        original_forward = patch_target.forward
        tau = tau_by_layer.get(layer_idx, 0.0)

        if binding.kind == "mlp":
            top_k = binding.top_k
            norm_topk_prob = binding.norm_topk_prob

            def _forward(
                self,
                hidden_states,
                _layer_idx=layer_idx,
                _tau=tau,
                _top_k=top_k,
                _norm_topk_prob=norm_topk_prob,
            ):
                output, aux = moe_forward_with_top_p_selection(
                    hidden_states=hidden_states,
                    router=self.gate,
                    experts=self.experts,
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
                _tau=tau,
            ):
                output, aux = moe_experts_forward_with_top_p_selection(
                    hidden_states=hidden_states,
                    experts=self,
                    selected_experts=top_k_index,
                    routing_weights=top_k_weights,
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
