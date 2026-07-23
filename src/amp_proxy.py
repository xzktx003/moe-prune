from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .model_structure import get_layer_gamma_weight, iter_moe_layer_bindings


@dataclass
class LayerAmpStats:
    layer_idx: int
    amp: torch.Tensor
    gamma_norm: float


def split_gate_up_proj(gate_up_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split Qwen3 gate_up_proj into gate and up branches."""
    half = gate_up_proj.shape[0] // 2
    gate_proj = gate_up_proj[:half]
    up_proj = gate_up_proj[half:]
    return gate_proj, up_proj


def compute_expert_slanc_exact(
    gamma: torch.Tensor,
    E: torch.Tensor,
    B: torch.Tensor,
    G: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the expert-only SLaNC proxy from the PRD.

    gamma: [d]
    E: [d, m]
    B: [d, m]
    G: [m, d]
    """
    gamma = gamma.float()
    E = E.float()
    B = B.float()
    G = G.float()

    gamma_E = gamma[:, None] * E
    gamma_B = gamma[:, None] * B

    norm_gamma_E = torch.norm(gamma_E, p="fro")
    norm_gamma_B = torch.norm(gamma_B, p="fro")

    BG = B @ G
    EG = E @ G

    a_E = torch.norm(gamma[:, None] * (norm_gamma_E * BG), p="fro")
    a_B = torch.norm(gamma[:, None] * (norm_gamma_B * EG), p="fro")
    return torch.sqrt(a_E * a_B + eps)


def build_amp_table_for_layer(
    gamma: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    raw_scores: List[torch.Tensor] = []
    for expert_idx in range(gate_up_proj.shape[0]):
        gate_proj_weight, up_proj_weight = split_gate_up_proj(gate_up_proj[expert_idx])
        E = up_proj_weight.transpose(0, 1).contiguous()
        B = gate_proj_weight.transpose(0, 1).contiguous()
        G = down_proj[expert_idx].transpose(0, 1).contiguous()
        raw_scores.append(compute_expert_slanc_exact(gamma, E, B, G, eps=eps))

    raw = torch.stack(raw_scores)
    return raw / (raw.mean() + eps)


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.float() ** 2) + eps)


@torch.no_grad()
def compute_expert_router_proto_amp(
    router_w_e: torch.Tensor,
    E: torch.Tensor,
    B: torch.Tensor,
    G: torch.Tensor,
    input_rms: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Measure expert amplification along the router-selected prototype direction.

    router_w_e: [hidden_size]
    E/B/G use canonical shapes [hidden, intermediate], [hidden, intermediate],
    [intermediate, hidden].
    """
    router_w_e = router_w_e.float()
    E = E.float()
    B = B.float()
    G = G.float()

    q = router_w_e / (rms(router_w_e, eps=eps) + eps)
    q = q * float(input_rms)

    up_branch = q @ E
    gate_branch = q @ B
    mid = torch.nn.functional.silu(gate_branch) * up_branch
    out = mid @ G
    return out.norm(p=2) / (q.norm(p=2) + eps)


@torch.no_grad()
def build_router_proto_amp_table_for_layer(
    router_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    input_rms: float = 1.0,
    eps: float = 1e-8,
    center_router_weights: bool = False,
) -> torch.Tensor:
    if center_router_weights:
        router_weight = router_weight - router_weight.mean(dim=0, keepdim=True)
    raw_scores: List[torch.Tensor] = []
    for expert_idx in range(gate_up_proj.shape[0]):
        gate_proj_weight, up_proj_weight = split_gate_up_proj(gate_up_proj[expert_idx])
        E = up_proj_weight.transpose(0, 1).contiguous()
        B = gate_proj_weight.transpose(0, 1).contiguous()
        G = down_proj[expert_idx].transpose(0, 1).contiguous()
        raw_scores.append(
            compute_expert_router_proto_amp(
                router_w_e=router_weight[expert_idx],
                E=E,
                B=B,
                G=G,
                input_rms=input_rms,
                eps=eps,
            )
        )

    raw = torch.stack(raw_scores)
    return raw / (raw.mean() + eps)


def build_amp_table_for_model(model, eps: float = 1e-8) -> Dict[int, torch.Tensor]:
    amp_table: Dict[int, torch.Tensor] = {}
    for binding in iter_moe_layer_bindings(model):
        layer_idx = binding.layer_idx
        layer = binding.layer
        experts = binding.experts
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            gamma = get_layer_gamma_weight(layer).detach()
            gate_up_proj = experts.gate_up_proj.detach()
            down_proj = experts.down_proj.detach()
            amp_table[layer_idx] = build_amp_table_for_layer(gamma, gate_up_proj, down_proj, eps=eps).cpu()
            continue

        raw_scores: List[torch.Tensor] = []
        gamma = get_layer_gamma_weight(layer).detach()
        for expert_layer in experts:
            gate_proj_weight = expert_layer.gate_proj.weight.detach()
            up_proj_weight = expert_layer.up_proj.weight.detach()
            down_proj_weight = expert_layer.down_proj.weight.detach()
            E = up_proj_weight.transpose(0, 1).contiguous()
            B = gate_proj_weight.transpose(0, 1).contiguous()
            G = down_proj_weight.transpose(0, 1).contiguous()
            raw_scores.append(compute_expert_slanc_exact(gamma, E, B, G, eps=eps))

        raw = torch.stack(raw_scores)
        amp_table[layer_idx] = (raw / (raw.mean() + eps)).cpu()
    return amp_table


def _input_rms_for_layer(input_rms_by_layer, layer_idx: int) -> float:
    if input_rms_by_layer is None:
        return 1.0
    if isinstance(input_rms_by_layer, dict):
        return float(input_rms_by_layer.get(layer_idx, 1.0))
    return float(input_rms_by_layer[layer_idx])



def _get_router_weight_tensor(router) -> torch.Tensor | None:
    """Extract the effective router weight tensor.

    Handles different router architectures:
    - Standard: ``router.weight`` (nn.Linear, Qwen3MoE same pattern)
    - Gemma4TextRouter: ``router.proj.weight``
    """
    if hasattr(router, "weight") and isinstance(router.weight, torch.Tensor):
        return router.weight
    for _name, module in router.named_children():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            return module.weight
    return None

@torch.no_grad()
def build_router_proto_amp_table_for_model(
    model,
    input_rms_by_layer=None,
    eps: float = 1e-8,
    center_router_weights: bool = False,
) -> Dict[int, torch.Tensor]:
    proto_amp_table: Dict[int, torch.Tensor] = {}
    for binding in iter_moe_layer_bindings(model):
        layer_idx = binding.layer_idx
        experts = binding.experts
        router = binding.router
        if router is None:
            continue
        router_weight = _get_router_weight_tensor(router)
        if router_weight is None:
            continue

        router_weight = router_weight.detach()
        input_rms = _input_rms_for_layer(input_rms_by_layer, layer_idx)
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            proto_amp_table[layer_idx] = build_router_proto_amp_table_for_layer(
                router_weight=router_weight,
                gate_up_proj=experts.gate_up_proj.detach(),
                down_proj=experts.down_proj.detach(),
                input_rms=input_rms,
                eps=eps,
                center_router_weights=center_router_weights,
            ).cpu()
            continue

        if center_router_weights:
            router_weight = router_weight - router_weight.mean(dim=0, keepdim=True)
        raw_scores: List[torch.Tensor] = []
        for expert_idx, expert_layer in enumerate(experts):
            gate_proj_weight = expert_layer.gate_proj.weight.detach()
            up_proj_weight = expert_layer.up_proj.weight.detach()
            down_proj_weight = expert_layer.down_proj.weight.detach()
            E = up_proj_weight.transpose(0, 1).contiguous()
            B = gate_proj_weight.transpose(0, 1).contiguous()
            G = down_proj_weight.transpose(0, 1).contiguous()
            raw_scores.append(
                compute_expert_router_proto_amp(
                    router_w_e=router_weight[expert_idx],
                    E=E,
                    B=B,
                    G=G,
                    input_rms=input_rms,
                    eps=eps,
                )
            )

        raw = torch.stack(raw_scores)
        proto_amp_table[layer_idx] = (raw / (raw.mean() + eps)).cpu()
    return proto_amp_table


def summarize_amp_table(amp_table: Dict[int, torch.Tensor]) -> List[LayerAmpStats]:
    return [
        LayerAmpStats(
            layer_idx=layer_idx,
            amp=amp,
            gamma_norm=float(torch.norm(amp, p=2).item()),
        )
        for layer_idx, amp in sorted(amp_table.items())
    ]
