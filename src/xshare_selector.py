from __future__ import annotations

import math
from contextlib import contextmanager
from types import MethodType
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .model_structure import iter_moe_layer_bindings
from .runtime_pruner import (
    RuntimeStats,
    compute_moe_weighted_hidden_states,
    compute_optional_shared_expert_output,
    renorm_gate_after_pruning,
    route_qwen3_topk,
)


def xshare_budget_from_tau(num_experts: int, tau: float) -> int:
    """Convert a pruning-ratio style ``tau`` into a batch expert budget."""

    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    keep_fraction = 1.0 - float(tau)
    budget = int(math.ceil(float(num_experts) * keep_fraction))
    return max(1, min(int(num_experts), budget))


def build_xshare_batch_hot_expert_mask(
    router_logits: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select batch-level hot experts by aggregate router probability.

    This is a HuggingFace-runtime adaptation of XShare's batch-aware expert
    selection idea: aggregate expert demand within the current forward pass and
    keep only a fixed batch expert budget. The budget is controlled by ``tau``
    as a target expert-pruning ratio.
    """

    if router_logits.ndim != 2:
        raise ValueError(
            f"router_logits must be 2D [tokens, experts], got shape {tuple(router_logits.shape)}"
        )
    num_experts = int(router_logits.shape[-1])
    budget = xshare_budget_from_tau(num_experts, tau)
    demand = F.softmax(router_logits.float(), dim=-1).sum(dim=0)
    hot_idx = torch.topk(demand, k=budget, dim=0).indices
    hot_mask = torch.zeros(num_experts, dtype=torch.bool, device=router_logits.device)
    hot_mask.scatter_(dim=0, index=hot_idx, value=True)
    return hot_mask, demand


def build_xshare_keep_mask_from_hot_experts(
    selected_experts: torch.Tensor,
    gate: torch.Tensor,
    hot_expert_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask routed top-k slots whose experts are outside the batch hot set."""

    if selected_experts.shape != gate.shape:
        raise ValueError(
            "selected_experts and gate must have the same shape, got "
            f"{tuple(selected_experts.shape)} vs {tuple(gate.shape)}"
        )
    hot_mask = hot_expert_mask.to(device=selected_experts.device, dtype=torch.bool)
    keep_mask = hot_mask[selected_experts.to(torch.long)]
    top1_idx = gate.argmax(dim=-1, keepdim=True)
    keep_mask.scatter_(dim=-1, index=top1_idx, value=True)
    return keep_mask, keep_mask.to(gate.dtype)


def build_xshare_keep_mask_from_topk_demand(
    selected_experts: torch.Tensor,
    gate: torch.Tensor,
    tau: float,
    num_experts: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fallback XShare score when only top-k routed experts are available."""

    if selected_experts.shape != gate.shape:
        raise ValueError(
            "selected_experts and gate must have the same shape, got "
            f"{tuple(selected_experts.shape)} vs {tuple(gate.shape)}"
        )
    inferred_experts = int(selected_experts.max().item()) + 1 if selected_experts.numel() else 1
    total_experts = int(num_experts) if num_experts is not None else inferred_experts
    total_experts = max(total_experts, inferred_experts, 1)
    budget = xshare_budget_from_tau(total_experts, tau)

    demand = gate.new_zeros((total_experts,), dtype=torch.float32)
    demand.scatter_add_(
        dim=0,
        index=selected_experts.reshape(-1).to(torch.long),
        src=gate.float().reshape(-1),
    )
    hot_idx = torch.topk(demand, k=budget, dim=0).indices
    hot_mask = torch.zeros(total_experts, dtype=torch.bool, device=gate.device)
    hot_mask.scatter_(dim=0, index=hot_idx, value=True)
    keep_mask, score = build_xshare_keep_mask_from_hot_experts(
        selected_experts=selected_experts,
        gate=gate,
        hot_expert_mask=hot_mask,
    )
    return keep_mask, score, demand


def moe_forward_with_xshare_selection(
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

    router_logits, gate, selected_experts = route_qwen3_topk(
        router,
        flat_states,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
    )
    hot_mask, demand = build_xshare_batch_hot_expert_mask(router_logits, tau=tau)
    keep_mask, score = build_xshare_keep_mask_from_hot_experts(
        selected_experts=selected_experts,
        gate=gate,
        hot_expert_mask=hot_mask,
    )
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
        "xshare_score": score.detach(),
        "xshare_demand": demand.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


def moe_experts_forward_with_xshare_selection(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
    return_aux: bool = False,
    moe_backend: str = "triton",
):
    keep_mask, score, demand = build_xshare_keep_mask_from_topk_demand(
        selected_experts=selected_experts,
        gate=routing_weights,
        tau=tau,
        num_experts=len(experts) if hasattr(experts, "__len__") else None,
    )
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
        "xshare_score": score.detach(),
        "xshare_demand": demand.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


@contextmanager
def patch_qwen3_moe_blocks_xshare(
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
                output, aux = moe_forward_with_xshare_selection(
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
                output, aux = moe_experts_forward_with_xshare_selection(
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
