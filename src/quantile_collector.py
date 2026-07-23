from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Dict, List

import torch

from .model_structure import iter_moe_layer_bindings
from .moe_cache_collector import compute_expert_outputs_local, route_qwen3_topk
from .quantile_search import build_candidate_scores_from_p_final
from .sere_selector import build_sere_dissimilarity_score
from .top_p_selector import build_top_p_residual_score


EPS = 1e-8
QUANTILE_RUNTIME_METHODS = {
    "ace",
    "gsp",
    "rcr",
    "score_only",
    "naee",
    "aimer",
    "expert_sparsity",
    "top_p",
    "sere",
}


def _selected_importance(
    amp_tables: Dict[int, torch.Tensor],
    layer_idx: int,
    topk_idx: torch.Tensor,
) -> torch.Tensor:
    return amp_tables[layer_idx].to(device=topk_idx.device, dtype=torch.float32)[topk_idx.to(torch.long)]


def _candidate_scores_for_method(
    method: str,
    p_final: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    if method != "ace":
        return build_candidate_scores_from_p_final(p_final, min_keep=1)[0]
    forced_keep = torch.zeros_like(p_final, dtype=torch.bool)
    forced_keep.scatter_(dim=-1, index=gate.argmax(dim=-1, keepdim=True), value=True)
    return p_final[~forced_keep]


@contextmanager
def patched_model_for_quantile_collection(
    model,
    *,
    method: str,
    candidate_scores_by_layer: Dict[int, List[torch.Tensor]],
    total_slots_by_layer: Dict[int, int],
    amp_tables: Dict[int, torch.Tensor] | None = None,
    proto_amp_tables: Dict[int, torch.Tensor] | None = None,
    sim_tables: Dict[int, torch.Tensor] | None = None,
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
                _, routing_weights, selected_experts = route_qwen3_topk(
                    self.gate,
                    flat_states,
                    top_k=_top_k,
                    norm_topk_prob=_norm_topk_prob,
                )
                gate = routing_weights.float()
                topk_idx = selected_experts

                if method in {"score_only", "expert_sparsity"}:
                    gate_max = gate.max(dim=-1, keepdim=True).values + EPS
                    p_final = gate / gate_max
                elif method == "gsp":
                    assert amp_tables is not None
                    amp_sel = _selected_importance(amp_tables, _layer_idx, topk_idx)
                    score = gate * amp_sel
                    p_final = score / (score.sum(dim=-1, keepdim=True) + EPS)
                elif method == "ace":
                    assert amp_tables is not None and proto_amp_tables is not None
                    amp_sel = _selected_importance(amp_tables, _layer_idx, topk_idx)
                    proto_sel = _selected_importance(proto_amp_tables, _layer_idx, topk_idx)
                    score_amp = gate * amp_sel
                    score_proto = gate * proto_sel
                    p_amp = score_amp / (score_amp.sum(dim=-1, keepdim=True) + EPS)
                    p_proto = score_proto / (score_proto.sum(dim=-1, keepdim=True) + EPS)
                    p_final = torch.maximum(p_amp, p_proto)
                elif method == "rcr":
                    assert proto_amp_tables is not None
                    proto_sel = _selected_importance(proto_amp_tables, _layer_idx, topk_idx)
                    score = gate * proto_sel
                    p_final = score / (score.sum(dim=-1, keepdim=True) + EPS)
                elif method == "naee":
                    assert amp_tables is not None
                    amp_sel = _selected_importance(amp_tables, _layer_idx, topk_idx)
                    score = gate * amp_sel
                    score = score / (score.sum(dim=-1, keepdim=True) + EPS)
                    score_max = score.max(dim=-1, keepdim=True).values + EPS
                    p_final = score / score_max
                elif method == "aimer":
                    assert amp_tables is not None
                    keep_sel = _selected_importance(amp_tables, _layer_idx, topk_idx)
                    score = gate * keep_sel
                    p_final = score / (score.sum(dim=-1, keepdim=True) + EPS)
                elif method == "top_p":
                    p_final = build_top_p_residual_score(gate)
                elif method == "sere":
                    assert sim_tables is not None
                    p_final = build_sere_dissimilarity_score(topk_idx, sim_tables[_layer_idx])
                else:
                    raise ValueError(f"Unsupported quantile method: {method}")

                candidate_scores = _candidate_scores_for_method(method, p_final, gate)
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

                if method in {"score_only", "expert_sparsity"}:
                    gate_max = gate.max(dim=-1, keepdim=True).values + EPS
                    p_final = gate / gate_max
                elif method == "gsp":
                    assert amp_tables is not None
                    amp_sel = _selected_importance(amp_tables, _layer_idx, topk_idx)
                    score = gate * amp_sel
                    p_final = score / (score.sum(dim=-1, keepdim=True) + EPS)
                elif method == "ace":
                    assert amp_tables is not None and proto_amp_tables is not None
                    amp_sel = _selected_importance(amp_tables, _layer_idx, topk_idx)
                    proto_sel = _selected_importance(proto_amp_tables, _layer_idx, topk_idx)
                    score_amp = gate * amp_sel
                    score_proto = gate * proto_sel
                    p_amp = score_amp / (score_amp.sum(dim=-1, keepdim=True) + EPS)
                    p_proto = score_proto / (score_proto.sum(dim=-1, keepdim=True) + EPS)
                    p_final = torch.maximum(p_amp, p_proto)
                elif method == "rcr":
                    assert proto_amp_tables is not None
                    proto_sel = _selected_importance(proto_amp_tables, _layer_idx, topk_idx)
                    score = gate * proto_sel
                    p_final = score / (score.sum(dim=-1, keepdim=True) + EPS)
                elif method == "naee":
                    assert amp_tables is not None
                    amp_sel = _selected_importance(amp_tables, _layer_idx, topk_idx)
                    score = gate * amp_sel
                    score = score / (score.sum(dim=-1, keepdim=True) + EPS)
                    score_max = score.max(dim=-1, keepdim=True).values + EPS
                    p_final = score / score_max
                elif method == "aimer":
                    assert amp_tables is not None
                    keep_sel = _selected_importance(amp_tables, _layer_idx, topk_idx)
                    score = gate * keep_sel
                    p_final = score / (score.sum(dim=-1, keepdim=True) + EPS)
                elif method == "top_p":
                    p_final = build_top_p_residual_score(gate)
                elif method == "sere":
                    assert sim_tables is not None
                    p_final = build_sere_dissimilarity_score(topk_idx, sim_tables[_layer_idx])
                else:
                    raise ValueError(f"Unsupported quantile method: {method}")

                candidate_scores = _candidate_scores_for_method(method, p_final, gate)
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
