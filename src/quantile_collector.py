from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Dict, List

import torch
import torch.nn.functional as F

from .model_structure import iter_moe_layer_bindings
from .runtime_pruner import combine_ace_scores, compute_expert_outputs


EPS = 1e-8


def route_topk(
    router,
    hidden_states: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router_out = router(hidden_states)
    if isinstance(router_out, tuple) and len(router_out) == 3:
        return router_out
    router_logits = router_out[0] if isinstance(router_out, tuple) else router_out
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return router_logits, routing_weights.to(hidden_states.dtype), selected_experts


def compute_expert_outputs_local(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    return compute_expert_outputs(hidden_states, experts, selected_experts)


def _selected_importance(
    amp_tables: Dict[int, torch.Tensor],
    layer_idx: int,
    topk_idx: torch.Tensor,
) -> torch.Tensor:
    return amp_tables[layer_idx].to(device=topk_idx.device, dtype=torch.float32)[topk_idx.to(torch.long)]


def ace_candidate_scores(
    p_final: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    forced_keep = torch.zeros_like(p_final, dtype=torch.bool)
    forced_keep.scatter_(dim=-1, index=gate.argmax(dim=-1, keepdim=True), value=True)
    return p_final[~forced_keep]


@contextmanager
def patched_model_for_quantile_collection(
    model,
    *,
    candidate_scores_by_layer: Dict[int, List[torch.Tensor]],
    total_slots_by_layer: Dict[int, int],
    amp_tables: Dict[int, torch.Tensor] | None = None,
    proto_amp_tables: Dict[int, torch.Tensor] | None = None,
):
    originals: List[tuple[object, object]] = []

    for binding in iter_moe_layer_bindings(model):
        layer_idx = binding.layer_idx
        if binding.kind == "mlp" and binding.router is None:
            continue
        patch_target = binding.patch_target
        original_forward = patch_target.forward
        candidate_scores_by_layer.setdefault(layer_idx, [])
        total_slots_by_layer.setdefault(layer_idx, 0)

        if binding.kind == "mlp":
            top_k = binding.top_k
            norm_topk_prob = binding.norm_topk_prob

            def _forward(self, hidden_states, _layer_idx=layer_idx, _top_k=top_k, _norm_topk_prob=norm_topk_prob):
                batch_size, sequence_length, hidden_dim = hidden_states.shape
                flat_states = hidden_states.view(-1, hidden_dim)
                _, routing_weights, selected_experts = route_topk(
                    self.gate,
                    flat_states,
                    top_k=_top_k,
                    norm_topk_prob=_norm_topk_prob,
                )
                gate = routing_weights.float()
                topk_idx = selected_experts

                assert amp_tables is not None and proto_amp_tables is not None
                amp_sel = _selected_importance(amp_tables, _layer_idx, topk_idx)
                proto_sel = _selected_importance(proto_amp_tables, _layer_idx, topk_idx)
                score_amp = gate * amp_sel
                score_proto = gate * proto_sel
                p_amp = score_amp / (score_amp.sum(dim=-1, keepdim=True) + EPS)
                p_proto = score_proto / (score_proto.sum(dim=-1, keepdim=True) + EPS)
                p_final = combine_ace_scores(p_amp, p_proto)

                candidate_scores = ace_candidate_scores(p_final, gate)
                if candidate_scores.numel() > 0:
                    candidate_scores_by_layer[_layer_idx].append(candidate_scores.detach().cpu())
                total_slots_by_layer[_layer_idx] += int(p_final.numel())

                expert_outs = compute_expert_outputs_local(flat_states, self.experts, selected_experts)
                full_out = (routing_weights.unsqueeze(-1) * expert_outs).sum(dim=1)

                shared_expert = getattr(self, "shared_expert", None)
                shared_expert_gate = getattr(self, "shared_expert_gate", None)
                if shared_expert is not None and shared_expert_gate is not None:
                    shared_out = shared_expert(flat_states)
                    shared_out = torch.sigmoid(shared_expert_gate(flat_states)) * shared_out
                    full_out = full_out + shared_out

                return full_out.view(batch_size, sequence_length, hidden_dim)
        else:
            def _forward(self, hidden_states, top_k_index, top_k_weights, _layer_idx=layer_idx):
                gate = top_k_weights.float()
                topk_idx = top_k_index

                assert amp_tables is not None and proto_amp_tables is not None
                amp_sel = _selected_importance(amp_tables, _layer_idx, topk_idx)
                proto_sel = _selected_importance(proto_amp_tables, _layer_idx, topk_idx)
                score_amp = gate * amp_sel
                score_proto = gate * proto_sel
                p_amp = score_amp / (score_amp.sum(dim=-1, keepdim=True) + EPS)
                p_proto = score_proto / (score_proto.sum(dim=-1, keepdim=True) + EPS)
                p_final = combine_ace_scores(p_amp, p_proto)

                candidate_scores = ace_candidate_scores(p_final, gate)
                if candidate_scores.numel() > 0:
                    candidate_scores_by_layer[_layer_idx].append(candidate_scores.detach().cpu())
                total_slots_by_layer[_layer_idx] += int(p_final.numel())

                expert_outs = compute_expert_outputs_local(hidden_states, self, topk_idx)
                full_out = (top_k_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
                return full_out

        originals.append((patch_target, original_forward))
        patch_target.forward = MethodType(_forward, patch_target)

    try:
        yield
    finally:
        for patch_target, original_forward in originals:
            patch_target.forward = original_forward
