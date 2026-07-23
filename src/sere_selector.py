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
    route_qwen3_topk,
)


def build_sere_dissimilarity_score(
    selected_experts: torch.Tensor,
    sim_matrix: torch.Tensor,
) -> torch.Tensor:
    """Score routed expert slots by dissimilarity to the token's primary expert.

    This is a post-hoc HF-runtime adaptation of SERE's similarity-based
    secondary-expert rerouting. The primary expert is the first routed slot for
    each token. Secondary slots with high similarity to that primary can be
    rerouted into it, so the thresholdable score is ``1 - similarity``.
    """

    if selected_experts.ndim != 2:
        raise ValueError(
            f"selected_experts must be 2D [T, k], got shape {tuple(selected_experts.shape)}"
        )
    if sim_matrix.ndim != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError(f"sim_matrix must be square, got shape {tuple(sim_matrix.shape)}")
    sim = sim_matrix.to(device=selected_experts.device, dtype=torch.float32)
    primary = selected_experts[:, :1].to(torch.long)
    secondary = selected_experts.to(torch.long)
    similarity = sim[primary, secondary].squeeze(1)
    dissimilarity = 1.0 - similarity.clamp(min=0.0, max=1.0)
    dissimilarity[:, 0] = 1.0
    return dissimilarity


def build_sere_keep_mask_and_gate(
    selected_experts: torch.Tensor,
    gate: torch.Tensor,
    sim_matrix: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(keep_mask, score, gate_rerouted)`` for SERE-style rerouting.

    Secondary slots whose dissimilarity to the primary slot is below ``tau`` are
    rerouted into the primary slot and then dropped. Their router mass is added
    to the primary gate before the final MoE call. This counts removed slots in
    the same metric as the other runtime-skipping baselines while preserving the
    original token's total router mass.
    """

    score = build_sere_dissimilarity_score(selected_experts, sim_matrix)
    keep_mask = score >= float(tau)
    keep_mask[:, 0] = True

    gate_f = gate.float()
    rerouted_mass = (gate_f * (~keep_mask).to(gate_f.dtype)).sum(dim=-1, keepdim=True)
    gate_rerouted = gate_f * keep_mask.to(gate_f.dtype)
    gate_rerouted[:, :1] = gate_rerouted[:, :1] + rerouted_mass
    gate_rerouted = gate_rerouted / (gate_rerouted.sum(dim=-1, keepdim=True) + eps)
    return keep_mask, score, gate_rerouted.to(gate.dtype)


def moe_forward_with_sere_selection(
    hidden_states: torch.Tensor,
    router,
    experts,
    sim_matrix: torch.Tensor,
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
    keep_mask, score, gate_rerouted = build_sere_keep_mask_and_gate(
        selected_experts=selected_experts,
        gate=gate,
        sim_matrix=sim_matrix,
        tau=tau,
        eps=eps,
    )
    final_hidden, expert_outputs, full_out = compute_moe_weighted_hidden_states(
        flat_states,
        experts,
        selected_experts,
        gate_rerouted,
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
        "sere_dissimilarity_score": score.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_rerouted.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


def moe_experts_forward_with_sere_selection(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    sim_matrix: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
    return_aux: bool = False,
    moe_backend: str = "triton",
):
    keep_mask, score, gate_rerouted = build_sere_keep_mask_and_gate(
        selected_experts=selected_experts,
        gate=routing_weights,
        sim_matrix=sim_matrix,
        tau=tau,
        eps=eps,
    )
    final_hidden, expert_outputs, full_out = compute_moe_weighted_hidden_states(
        hidden_states,
        experts,
        selected_experts,
        gate_rerouted,
        keep_mask=keep_mask,
        moe_backend=moe_backend,
    )

    aux = {
        "topk_idx": selected_experts.detach(),
        "gate": routing_weights.detach(),
        "sere_dissimilarity_score": score.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_rerouted.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


@contextmanager
def patch_qwen3_moe_blocks_sere(
    model,
    sim_table: Dict[int, torch.Tensor],
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
        sim_matrix = sim_table[layer_idx]

        if binding.kind == "mlp":
            top_k = binding.top_k
            norm_topk_prob = binding.norm_topk_prob

            def _forward(
                self,
                hidden_states,
                _layer_idx=layer_idx,
                _tau=tau,
                _sim_matrix=sim_matrix,
                _top_k=top_k,
                _norm_topk_prob=norm_topk_prob,
            ):
                output, aux = moe_forward_with_sere_selection(
                    hidden_states=hidden_states,
                    router=self.gate,
                    experts=self.experts,
                    sim_matrix=_sim_matrix,
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
                _sim_matrix=sim_matrix,
            ):
                output, aux = moe_experts_forward_with_sere_selection(
                    hidden_states=hidden_states,
                    experts=self,
                    selected_experts=top_k_index,
                    routing_weights=top_k_weights,
                    sim_matrix=_sim_matrix,
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
