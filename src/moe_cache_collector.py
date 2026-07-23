from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from .model_structure import iter_moe_layer_bindings
from .model_adapter import maybe_bf16_autocast
from .runtime_pruner import compute_expert_outputs


@dataclass
class LayerCacheAccumulator:
    topk_idx: List[torch.Tensor] = field(default_factory=list)
    gate: List[torch.Tensor] = field(default_factory=list)
    expert_outs: List[torch.Tensor] = field(default_factory=list)
    full_out: List[torch.Tensor] = field(default_factory=list)
    token_limit: int = 256

    def append(
        self,
        topk_idx: torch.Tensor,
        gate: torch.Tensor,
        expert_outs: torch.Tensor,
        full_out: torch.Tensor,
    ) -> None:
        current_tokens = sum(item.shape[0] for item in self.topk_idx)
        if current_tokens >= self.token_limit:
            return
        remaining = self.token_limit - current_tokens
        sl = slice(0, remaining)
        self.topk_idx.append(topk_idx[sl].detach().cpu())
        self.gate.append(gate[sl].detach().cpu())
        self.expert_outs.append(expert_outs[sl].detach().cpu())
        self.full_out.append(full_out[sl].detach().cpu())

    def build(self) -> Dict[str, torch.Tensor]:
        return {
            "topk_idx": torch.cat(self.topk_idx, dim=0) if self.topk_idx else torch.empty(0, 0, dtype=torch.long),
            "gate": torch.cat(self.gate, dim=0) if self.gate else torch.empty(0, 0),
            "expert_outs": torch.cat(self.expert_outs, dim=0) if self.expert_outs else torch.empty(0, 0, 0),
            "full_out": torch.cat(self.full_out, dim=0) if self.full_out else torch.empty(0, 0),
        }


def route_qwen3_topk(
    router,
    hidden_states: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router_out = router(hidden_states)
    if isinstance(router_out, tuple) and len(router_out) == 3:
        router_logits, routing_weights, selected_experts = router_out
        return router_logits, routing_weights, selected_experts

    if isinstance(router_out, tuple):
        if len(router_out) >= 1 and torch.is_tensor(router_out[0]):
            router_logits = router_out[0]
        else:
            raise TypeError(f"Unsupported router output tuple shape: {type(router_out)}")
    else:
        router_logits = router_out

    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    return router_logits, routing_weights, selected_experts


def compute_expert_outputs_local(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        return compute_expert_outputs(hidden_states, experts, selected_experts)

    tokens, top_k = selected_experts.shape
    hidden_dim = hidden_states.shape[-1]
    outputs = hidden_states.new_zeros((tokens, top_k, hidden_dim))
    unique_experts = torch.unique(selected_experts)
    for expert_idx in unique_experts.tolist():
        positions = torch.nonzero(selected_experts == expert_idx, as_tuple=False)
        if positions.numel() == 0:
            continue
        token_idx = positions[:, 0]
        slot_idx = positions[:, 1]
        current_state = hidden_states[token_idx]
        current_hidden = experts[expert_idx](current_state)
        outputs[token_idx, slot_idx] = current_hidden.to(outputs.dtype)
    return outputs


@contextmanager
def patch_qwen3_moe_for_cache_collection(model, token_limit: int = 256):
    accumulators: Dict[int, LayerCacheAccumulator] = {}
    originals: List[tuple[object, object]] = []

    for binding in iter_moe_layer_bindings(model):
        layer_idx = binding.layer_idx
        if binding.kind == "mlp" and binding.router is None:
            continue
        patch_target = binding.patch_target
        accumulators[layer_idx] = LayerCacheAccumulator(token_limit=token_limit)
        original_forward = patch_target.forward

        if binding.kind == "mlp":
            top_k = binding.top_k
            norm_topk_prob = binding.norm_topk_prob

            def _forward(self, hidden_states, _layer_idx=layer_idx, _top_k=top_k, _norm_topk_prob=norm_topk_prob):
                batch_size, sequence_length, hidden_dim = hidden_states.shape
                flat_states = hidden_states.view(-1, hidden_dim)
                router_logits, routing_weights, selected_experts = route_qwen3_topk(
                    self.gate,
                    flat_states,
                    top_k=_top_k,
                    norm_topk_prob=_norm_topk_prob,
                )
                expert_outs = compute_expert_outputs_local(flat_states, self.experts, selected_experts)
                full_out = (routing_weights.unsqueeze(-1) * expert_outs).sum(dim=1)

                shared_expert = getattr(self, "shared_expert", None)
                shared_expert_gate = getattr(self, "shared_expert_gate", None)
                if shared_expert is not None and shared_expert_gate is not None:
                    shared_out = shared_expert(flat_states)
                    shared_out = torch.sigmoid(shared_expert_gate(flat_states)) * shared_out
                    full_out = full_out + shared_out

                accumulators[_layer_idx].append(
                    topk_idx=selected_experts,
                    gate=routing_weights,
                    expert_outs=expert_outs,
                    full_out=full_out,
                )
                return full_out.view(batch_size, sequence_length, hidden_dim)
        else:
            def _forward(self, hidden_states, top_k_index, top_k_weights, _layer_idx=layer_idx):
                expert_outs = compute_expert_outputs_local(hidden_states, self, top_k_index)
                full_out = (top_k_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
                accumulators[_layer_idx].append(
                    topk_idx=top_k_index,
                    gate=top_k_weights,
                    expert_outs=expert_outs,
                    full_out=full_out,
                )
                return full_out

        originals.append((patch_target, original_forward))
        patch_target.forward = MethodType(_forward, patch_target)

    try:
        yield accumulators
    finally:
        for patch_target, original_forward in originals:
            patch_target.forward = original_forward


def collect_layer_caches(
    model,
    tokenizer,
    texts: Iterable[str],
    max_length: int = 512,
    token_limit: int = 256,
) -> Dict[int, Dict[str, torch.Tensor]]:
    if hasattr(model, "device"):
        device = model.device
    elif hasattr(model, "hf_device_map"):
        device = list(model.hf_device_map.values())[0]
    else:
        device = next(model.parameters()).device
    with patch_qwen3_moe_for_cache_collection(model, token_limit=token_limit) as accumulators:
        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with maybe_bf16_autocast():
                    model(**encoded, use_cache=False)
                if all(sum(x.shape[0] for x in acc.topk_idx) >= token_limit for acc in accumulators.values()):
                    break

    return {layer_idx: accumulator.build() for layer_idx, accumulator in accumulators.items()}
