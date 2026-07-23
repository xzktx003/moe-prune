from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from .model_structure import iter_moe_layer_bindings
from .triton_group_gemm import (
    compute_fused_expert_outputs_triton,
    compute_fused_experts_triton,
    triton_moe_available,
)


def route_qwen3_topk(
    router,
    hidden_states: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate Qwen3MoeSparseMoeBlock's router logic.

    Returns ``(router_logits, routing_weights, selected_experts)`` compatible
    with both the legacy fused router (which returned all three directly) and
    the current transformers layout (router is a bare ``nn.Linear`` returning
    only logits).
    """
    router_out = router(hidden_states)
    if isinstance(router_out, tuple) and len(router_out) == 3:
        return router_out  # legacy fused router
    router_logits = router_out[0] if isinstance(router_out, tuple) else router_out

    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    return router_logits, routing_weights, selected_experts


@dataclass
class LayerPruningStats:
    layer_idx: int
    total_tokens: int = 0
    total_slots: int = 0
    kept_slots: int = 0

    @property
    def pruning_ratio(self) -> float:
        if self.total_slots == 0:
            return 0.0
        return 1.0 - (self.kept_slots / self.total_slots)

    @property
    def avg_active_experts(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.kept_slots / self.total_tokens


@dataclass
class RuntimeStats:
    layers: Dict[int, LayerPruningStats] = field(default_factory=dict)

    def update(self, layer_idx: int, keep_mask: torch.Tensor) -> None:
        layer_stats = self.layers.setdefault(layer_idx, LayerPruningStats(layer_idx=layer_idx))
        keep_mask_cpu = keep_mask.detach()
        layer_stats.total_tokens += int(keep_mask_cpu.shape[0])
        layer_stats.total_slots += int(keep_mask_cpu.numel())
        layer_stats.kept_slots += int(keep_mask_cpu.sum().item())

    def mean_pruning_ratio(self) -> float:
        if not self.layers:
            return 0.0
        total_slots = sum(layer.total_slots for layer in self.layers.values())
        kept_slots = sum(layer.kept_slots for layer in self.layers.values())
        if total_slots == 0:
            return 0.0
        return 1.0 - (kept_slots / total_slots)


def compute_importance_score(gate: torch.Tensor, amp_selected: torch.Tensor) -> torch.Tensor:
    return gate * amp_selected.to(gate.device, dtype=gate.dtype)


def build_keep_mask_topk(score: torch.Tensor, tau: float, eps: float = 1e-8) -> torch.Tensor:
    normalized = score / (score.sum(dim=-1, keepdim=True) + eps)
    keep_mask = normalized >= tau
    max_idx = score.argmax(dim=-1, keepdim=True)
    keep_mask.scatter_(dim=-1, index=max_idx, value=True)
    return keep_mask


def build_keep_mask_dual_view(
    gate: torch.Tensor,
    topk_idx: torch.Tensor,
    slanc_amp: torch.Tensor,
    proto_amp: torch.Tensor,
    tau_l: float,
    min_keep: int = 1,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    slanc_selected = slanc_amp.to(gate.device, dtype=gate.dtype)[topk_idx]
    proto_selected = proto_amp.to(gate.device, dtype=gate.dtype)[topk_idx]

    score_slanc = compute_importance_score(gate, slanc_selected)
    score_proto = compute_importance_score(gate, proto_selected)
    p_slanc = score_slanc.float() / (score_slanc.float().sum(dim=-1, keepdim=True) + eps)
    p_proto = score_proto.float() / (score_proto.float().sum(dim=-1, keepdim=True) + eps)
    p_final = torch.maximum(p_slanc, p_proto)

    keep_mask = p_final >= float(tau_l)
    top1_idx = gate.argmax(dim=-1, keepdim=True)
    keep_mask.scatter_(dim=-1, index=top1_idx, value=True)

    if min_keep > 1:
        keep_count = min(int(min_keep), p_final.shape[-1])
        top_keep_idx = torch.topk(p_final, k=keep_count, dim=-1).indices
        keep_mask.scatter_(dim=-1, index=top_keep_idx, value=True)

    return keep_mask, {
        "slanc_amp_selected": slanc_selected.detach(),
        "proto_amp_selected": proto_selected.detach(),
        "score_slanc": score_slanc.detach(),
        "score_proto": score_proto.detach(),
        "p_slanc": p_slanc.detach(),
        "p_proto": p_proto.detach(),
        "p_final": p_final.detach(),
    }


def renorm_gate_after_pruning(gate: torch.Tensor, keep_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    gate_kept = gate * keep_mask.to(gate.dtype)
    return gate_kept / (gate_kept.sum(dim=-1, keepdim=True) + eps)


def compute_expert_outputs(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    keep_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-slot expert outputs before gate weighting.

    hidden_states: [tokens, hidden]
    selected_experts: [tokens, top_k]
    keep_mask: optional [tokens, top_k] boolean mask. When provided, only the
        kept expert slots are executed and pruned slots remain zero.
    returns: [tokens, top_k, hidden]
    """
    tokens, top_k = selected_experts.shape
    hidden_dim = hidden_states.shape[-1]
    outputs = hidden_states.new_zeros((tokens, top_k, hidden_dim))
    if keep_mask is None:
        active_mask = torch.ones_like(selected_experts, dtype=torch.bool)
    else:
        if keep_mask.shape != selected_experts.shape:
            raise ValueError(
                "keep_mask shape must match selected_experts shape, got "
                f"{tuple(keep_mask.shape)} vs {tuple(selected_experts.shape)}"
            )
        active_mask = keep_mask.to(dtype=torch.bool)

    active_positions = torch.nonzero(active_mask, as_tuple=False)
    if active_positions.numel() == 0:
        return outputs
    active_experts = selected_experts[active_mask]

    # Two possible layouts:
    #  - Fused: experts.gate_up_proj / experts.down_proj / experts.act_fn tensors (legacy).
    #  - ModuleList: each element is a Qwen3MoeMLP with gate_proj/up_proj/down_proj (transformers>=4.50).
    fused_layout = hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj") and not isinstance(experts, torch.nn.ModuleList)

    unique_experts = torch.unique(active_experts)
    for expert_idx in unique_experts.tolist():
        positions = active_positions[active_experts == expert_idx]
        if positions.numel() == 0:
            continue
        token_idx = positions[:, 0]
        slot_idx = positions[:, 1]
        current_state = hidden_states[token_idx]
        if fused_layout:
            gate_up = F.linear(current_state, experts.gate_up_proj[expert_idx])
            gate_branch, up_branch = gate_up.chunk(2, dim=-1)
            current_hidden = experts.act_fn(gate_branch) * up_branch
            current_hidden = F.linear(current_hidden, experts.down_proj[expert_idx])
        else:
            current_hidden = experts[expert_idx](current_state)
        outputs[token_idx, slot_idx] = current_hidden.to(outputs.dtype)

    return outputs


def _can_use_triton_moe_backend(
    hidden_states: torch.Tensor,
    experts,
    moe_backend: str,
) -> bool:
    normalized = str(moe_backend).lower()
    if normalized not in {"torch", "triton"}:
        raise ValueError(f"Unsupported MoE backend: {moe_backend}")
    if normalized != "triton":
        return False
    if not hidden_states.is_cuda or not triton_moe_available():
        return False
    return hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")


def compute_moe_weighted_hidden_states(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    keep_mask: Optional[torch.Tensor] = None,
    moe_backend: str = "triton",
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if keep_mask is None:
        keep_mask = torch.ones_like(selected_experts, dtype=torch.bool)
    if keep_mask.shape != selected_experts.shape:
        raise ValueError(
            "keep_mask shape must match selected_experts shape, got "
            f"{tuple(keep_mask.shape)} vs {tuple(selected_experts.shape)}"
        )
    if routing_weights.shape != selected_experts.shape:
        raise ValueError(
            "routing_weights shape must match selected_experts shape, got "
            f"{tuple(routing_weights.shape)} vs {tuple(selected_experts.shape)}"
        )

    if _can_use_triton_moe_backend(hidden_states, experts, moe_backend):
        expert_outputs, _ = compute_fused_expert_outputs_triton(
            hidden_states,
            experts,
            selected_experts,
            keep_mask,
        )
        final_hidden = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        full_out = None
        if bool(keep_mask.all()):
            full_out = final_hidden.detach()
        return final_hidden, expert_outputs, full_out

    expert_outputs = compute_expert_outputs(
        hidden_states,
        experts,
        selected_experts,
        keep_mask=keep_mask,
    )
    final_hidden = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
    full_out = None
    if bool(keep_mask.all()):
        full_out = final_hidden.detach()
    return final_hidden, expert_outputs, full_out


def compute_optional_shared_expert_output(
    hidden_states: torch.Tensor,
    shared_expert=None,
    shared_expert_gate=None,
) -> Optional[torch.Tensor]:
    if shared_expert is None or shared_expert_gate is None:
        return None
    shared_output = shared_expert(hidden_states)
    shared_gate = torch.sigmoid(shared_expert_gate(hidden_states))
    return shared_gate * shared_output


def moe_forward_with_amp_pruning(
    hidden_states: torch.Tensor,
    router,
    experts,
    amp_layer: torch.Tensor,
    tau_l: float,
    top_k: int = 8,
    norm_topk_prob: bool = True,
    eps: float = 1e-8,
    return_aux: bool = False,
    moe_backend: str = "triton",
    shared_expert=None,
    shared_expert_gate=None,
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

    _, gate, selected_experts = route_qwen3_topk(
        router,
        hidden_states_reshaped,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
    )
    amp_selected = amp_layer.to(gate.device)[selected_experts]
    score = compute_importance_score(gate, amp_selected)
    keep_mask = build_keep_mask_topk(score, tau=tau_l, eps=eps)
    gate_kept = renorm_gate_after_pruning(gate, keep_mask, eps=eps)

    final_hidden, expert_outputs, full_out = compute_moe_weighted_hidden_states(
        hidden_states_reshaped,
        experts,
        selected_experts,
        gate_kept,
        keep_mask=keep_mask,
        moe_backend=moe_backend,
    )
    shared_output = compute_optional_shared_expert_output(
        hidden_states_reshaped,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
    )
    if shared_output is not None:
        final_hidden = final_hidden + shared_output
    final_hidden = final_hidden.reshape(batch_size, sequence_length, hidden_dim)

    aux = {
        "topk_idx": selected_experts.detach(),
        "gate": gate.detach(),
        "amp_selected": amp_selected.detach(),
        "score": score.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


def moe_forward_with_dual_view_pruning(
    hidden_states: torch.Tensor,
    router,
    experts,
    slanc_amp_layer: torch.Tensor,
    proto_amp_layer: torch.Tensor,
    tau_l: float,
    top_k: int = 8,
    norm_topk_prob: bool = True,
    min_keep: int = 1,
    eps: float = 1e-8,
    return_aux: bool = False,
    moe_backend: str = "triton",
    shared_expert=None,
    shared_expert_gate=None,
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

    _, gate, selected_experts = route_qwen3_topk(
        router,
        hidden_states_reshaped,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
    )
    keep_mask, dual_aux = build_keep_mask_dual_view(
        gate=gate,
        topk_idx=selected_experts,
        slanc_amp=slanc_amp_layer,
        proto_amp=proto_amp_layer,
        tau_l=tau_l,
        min_keep=min_keep,
        eps=eps,
    )
    gate_kept = renorm_gate_after_pruning(gate, keep_mask, eps=eps)

    final_hidden, expert_outputs, full_out = compute_moe_weighted_hidden_states(
        hidden_states_reshaped,
        experts,
        selected_experts,
        gate_kept,
        keep_mask=keep_mask,
        moe_backend=moe_backend,
    )
    shared_output = compute_optional_shared_expert_output(
        hidden_states_reshaped,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
    )
    if shared_output is not None:
        final_hidden = final_hidden + shared_output
    final_hidden = final_hidden.reshape(batch_size, sequence_length, hidden_dim)

    aux = {
        "topk_idx": selected_experts.detach(),
        "gate": gate.detach(),
        **dual_aux,
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


def moe_experts_forward_with_amp_pruning(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    amp_layer: torch.Tensor,
    tau_l: float,
    eps: float = 1e-8,
    return_aux: bool = False,
    moe_backend: str = "triton",
):
    amp_selected = amp_layer.to(routing_weights.device)[selected_experts]
    score = compute_importance_score(routing_weights, amp_selected)
    keep_mask = build_keep_mask_topk(score, tau=tau_l, eps=eps)
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
        "amp_selected": amp_selected.detach(),
        "score": score.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


def moe_experts_forward_with_dual_view_pruning(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    slanc_amp_layer: torch.Tensor,
    proto_amp_layer: torch.Tensor,
    tau_l: float,
    min_keep: int = 1,
    eps: float = 1e-8,
    return_aux: bool = False,
    moe_backend: str = "triton",
):
    keep_mask, dual_aux = build_keep_mask_dual_view(
        gate=routing_weights,
        topk_idx=selected_experts,
        slanc_amp=slanc_amp_layer,
        proto_amp=proto_amp_layer,
        tau_l=tau_l,
        min_keep=min_keep,
        eps=eps,
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
        **dual_aux,
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if return_aux:
        return final_hidden, aux
    return final_hidden


@contextmanager
def patch_qwen3_moe_blocks(
    model,
    amp_table: Dict[int, torch.Tensor],
    tau_by_layer: Dict[int, float],
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    originals: List[tuple[object, object]] = []

    for binding in iter_moe_layer_bindings(model):
        layer_idx = binding.layer_idx
        if layer_idx not in amp_table:
            continue
        if binding.kind == "mlp" and binding.router is None:
            continue
        patch_target = binding.patch_target
        original_forward = patch_target.forward
        amp_layer = amp_table[layer_idx]
        tau_l = tau_by_layer.get(layer_idx, 0.0)

        if binding.kind == "mlp":
            top_k = binding.top_k
            norm_topk_prob = binding.norm_topk_prob

            def _forward(self, hidden_states, _layer_idx=layer_idx, _amp_layer=amp_layer, _tau=tau_l, _top_k=top_k, _norm_topk_prob=norm_topk_prob):
                output, aux = moe_forward_with_amp_pruning(
                    hidden_states=hidden_states,
                    router=self.gate,
                    experts=self.experts,
                    amp_layer=_amp_layer,
                    tau_l=_tau,
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
            def _forward(self, hidden_states, top_k_index, top_k_weights, _layer_idx=layer_idx, _amp_layer=amp_layer, _tau=tau_l):
                output, aux = moe_experts_forward_with_amp_pruning(
                    hidden_states=hidden_states,
                    experts=self,
                    selected_experts=top_k_index,
                    routing_weights=top_k_weights,
                    amp_layer=_amp_layer,
                    tau_l=_tau,
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


@contextmanager
def patch_qwen3_moe_blocks_dual_view(
    model,
    slanc_amp_table: Dict[int, torch.Tensor],
    proto_amp_table: Dict[int, torch.Tensor],
    tau_by_layer: Dict[int, float],
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
    min_keep: int = 1,
):
    originals: List[tuple[object, object]] = []

    for binding in iter_moe_layer_bindings(model):
        layer_idx = binding.layer_idx
        if layer_idx not in slanc_amp_table or layer_idx not in proto_amp_table:
            continue
        if binding.kind == "mlp" and binding.router is None:
            continue
        patch_target = binding.patch_target
        original_forward = patch_target.forward
        slanc_amp_layer = slanc_amp_table[layer_idx]
        proto_amp_layer = proto_amp_table[layer_idx]
        tau_l = tau_by_layer.get(layer_idx, 0.0)

        if binding.kind == "mlp":
            top_k = binding.top_k
            norm_topk_prob = binding.norm_topk_prob

            def _forward(
                self,
                hidden_states,
                _layer_idx=layer_idx,
                _slanc_amp_layer=slanc_amp_layer,
                _proto_amp_layer=proto_amp_layer,
                _tau=tau_l,
                _top_k=top_k,
                _norm_topk_prob=norm_topk_prob,
            ):
                output, aux = moe_forward_with_dual_view_pruning(
                    hidden_states=hidden_states,
                    router=self.gate,
                    experts=self.experts,
                    slanc_amp_layer=_slanc_amp_layer,
                    proto_amp_layer=_proto_amp_layer,
                    tau_l=_tau,
                    top_k=_top_k,
                    norm_topk_prob=_norm_topk_prob,
                    min_keep=min_keep,
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
                _slanc_amp_layer=slanc_amp_layer,
                _proto_amp_layer=proto_amp_layer,
                _tau=tau_l,
            ):
                output, aux = moe_experts_forward_with_dual_view_pruning(
                    hidden_states=hidden_states,
                    experts=self,
                    selected_experts=top_k_index,
                    routing_weights=top_k_weights,
                    slanc_amp_layer=_slanc_amp_layer,
                    proto_amp_layer=_proto_amp_layer,
                    tau_l=_tau,
                    min_keep=min_keep,
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


def build_uniform_tau_by_layer(layer_indices: Iterable[int], tau: float) -> Dict[int, float]:
    return {int(layer_idx): float(tau) for layer_idx in layer_indices}
