from __future__ import annotations

from typing import Dict, List

import torch

from .amp_proxy import split_gate_up_proj
from .model_structure import get_layer_gamma_weight, iter_moe_layer_bindings


def canonicalize_expert_weights(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_up_proj = gate_up_proj.detach().to(device="cpu")
    down_proj = down_proj.detach().to(device="cpu")
    gate_proj_weight, up_proj_weight = split_gate_up_proj(gate_up_proj)
    E = up_proj_weight.transpose(0, 1).contiguous()
    B = gate_proj_weight.transpose(0, 1).contiguous()
    G = down_proj.transpose(0, 1).contiguous()
    return E, B, G


def row_l2_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x.float(), ord=2, dim=1)


def col_l2_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x.float(), ord=2, dim=0)


def build_exact_signature(
    gamma: torch.Tensor,
    E: torch.Tensor,
    B: torch.Tensor,
    G: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    gamma = gamma.detach().to(device="cpu", dtype=torch.float32)
    E = E.detach().to(device="cpu", dtype=torch.float32)
    B = B.detach().to(device="cpu", dtype=torch.float32)
    G = G.detach().to(device="cpu", dtype=torch.float32)

    gamma_E = gamma[:, None] * E
    gamma_B = gamma[:, None] * B

    norm_gamma_E = torch.norm(gamma_E, p="fro")
    norm_gamma_B = torch.norm(gamma_B, p="fro")

    M_E = gamma[:, None] * (norm_gamma_E * (B @ G))
    M_B = gamma[:, None] * (norm_gamma_B * (E @ G))
    return torch.cat([M_E.reshape(-1), M_B.reshape(-1)], dim=0)


def build_fast_channel_signature(
    gamma: torch.Tensor,
    E: torch.Tensor,
    B: torch.Tensor,
    G: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    gamma = gamma.detach().to(device="cpu", dtype=torch.float32)
    E = E.detach().to(device="cpu", dtype=torch.float32)
    B = B.detach().to(device="cpu", dtype=torch.float32)
    G = G.detach().to(device="cpu", dtype=torch.float32)

    u = row_l2_norm(gamma[:, None] * E)
    v = row_l2_norm(gamma[:, None] * B)
    w = col_l2_norm(G)
    return torch.cat([u, v, w], dim=0)


def build_similarity_matrix(
    signatures: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    signatures = signatures.detach().to(device="cpu", dtype=torch.float32)
    norms = torch.linalg.vector_norm(
        signatures,
        ord=2,
        dim=1,
        keepdim=True,
    ).clamp_min(eps)
    normalized = signatures / norms
    sim = normalized @ normalized.transpose(0, 1)
    sim = sim.clamp(min=0.0, max=1.0)
    sim.fill_diagonal_(1.0)
    return sim


def build_fast_signatures_for_layer(
    gamma: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    gamma = gamma.detach().to(device="cpu", dtype=torch.float32).abs()
    gate_up_proj = gate_up_proj.detach().to(device="cpu", dtype=torch.float32)
    down_proj = down_proj.detach().to(device="cpu", dtype=torch.float32)

    half = gate_up_proj.shape[1] // 2
    gate_proj = gate_up_proj[:, :half, :]
    up_proj = gate_up_proj[:, half:, :]

    u = torch.linalg.vector_norm(up_proj, ord=2, dim=1) * gamma
    v = torch.linalg.vector_norm(gate_proj, ord=2, dim=1) * gamma
    w = torch.linalg.vector_norm(down_proj, ord=2, dim=2)
    return torch.cat([u, v, w], dim=1)


def build_layer_similarity_table(
    experts,
    gamma: torch.Tensor,
    mode: str = "fast",
    eps: float = 1e-8,
) -> torch.Tensor:
    experts = getattr(experts, "experts", experts)
    if not hasattr(experts, "gate_up_proj") or not hasattr(experts, "down_proj"):
        signatures: List[torch.Tensor] = []
        gamma = gamma.detach().to(device="cpu", dtype=torch.float32)
        for expert_layer in experts:
            E = expert_layer.up_proj.weight.detach().transpose(0, 1).contiguous()
            B = expert_layer.gate_proj.weight.detach().transpose(0, 1).contiguous()
            G = expert_layer.down_proj.weight.detach().transpose(0, 1).contiguous()
            if mode == "fast":
                signatures.append(build_fast_channel_signature(gamma, E, B, G, eps=eps))
            elif mode == "exact":
                signatures.append(build_exact_signature(gamma, E, B, G, eps=eps))
            else:
                raise ValueError(f"Unsupported similarity mode: {mode}")
        stacked = torch.stack(signatures, dim=0)
        return build_similarity_matrix(stacked, eps=eps)

    if mode == "fast":
        stacked = build_fast_signatures_for_layer(
            gamma,
            experts.gate_up_proj,
            experts.down_proj,
        )
        return build_similarity_matrix(stacked, eps=eps)

    if mode != "exact":
        raise ValueError(f"Unsupported similarity mode: {mode}")

    signatures: List[torch.Tensor] = []
    for expert_idx in range(experts.gate_up_proj.shape[0]):
        E, B, G = canonicalize_expert_weights(
            experts.gate_up_proj[expert_idx],
            experts.down_proj[expert_idx],
        )
        signatures.append(build_exact_signature(gamma, E, B, G, eps=eps))

    stacked = torch.stack(signatures, dim=0)
    return build_similarity_matrix(stacked, eps=eps)


def build_model_similarity_table(
    model,
    mode: str = "fast",
    eps: float = 1e-8,
) -> Dict[int, torch.Tensor]:
    sim_table: Dict[int, torch.Tensor] = {}
    for binding in iter_moe_layer_bindings(model):
        layer_idx = binding.layer_idx
        gamma = get_layer_gamma_weight(binding.layer).detach().to(device="cpu")
        sim_table[layer_idx] = build_layer_similarity_table(
            binding.experts,
            gamma,
            mode=mode,
            eps=eps,
        )
    return sim_table
