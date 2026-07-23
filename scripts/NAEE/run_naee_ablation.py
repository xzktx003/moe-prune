from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset

from moe_prune.code.scripts.shared.ppl_eval import FullWikiTextPerplexity
from moe_prune.code.src.amp_proxy import build_amp_table_for_model as build_shared_amp_table_for_model
from moe_prune.code.src.evaluation import proxy_validity_metrics
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection
from moe_prune.code.src.mcqa_eval import DATASET_SPECS, evaluate_mcqa_dataset
from moe_prune.code.src.model_adapter import clear_hf_proxy_env, load_qwen3_moe, maybe_bf16_autocast
from moe_prune.code.src.model_structure import get_layer_gamma_weight, iter_moe_layer_bindings
from moe_prune.code.src.moe_cache_collector import collect_layer_caches as collect_shared_layer_caches
from moe_prune.code.src.naee_output_paths import model_tag_for_path, resolve_naee_output_dir
from moe_prune.code.src.runtime_pruner import (
    RuntimeStats,
    build_uniform_tau_by_layer,
    compute_importance_score,
    compute_moe_weighted_hidden_states,
    compute_optional_shared_expert_output,
    renorm_gate_after_pruning,
)


DATASET_ALIASES = {
    "mathqa": "mathqa",
    "openbookqa": "openbookqa",
    "arc_challenge": "arc-c",
}

DEFAULT_NAEE_OUTPUT_DIR = "results/NAEE"
DEFAULT_SCORE_ONLY_OUTPUT_DIR = "results/score_only"
DEFAULT_PPL_SEQ_LEN = 2048


def split_gate_up_proj(gate_up_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    half = gate_up_proj.shape[0] // 2
    return gate_up_proj[:half], gate_up_proj[half:]


def compute_expert_slanc_exact(
    gamma: torch.Tensor,
    E: torch.Tensor,
    B: torch.Tensor,
    G: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
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
    layer,
    experts=None,
    eps: float = 1e-8,
) -> torch.Tensor:
    gamma = get_layer_gamma_weight(layer).detach()
    if experts is None:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is None:
            experts = getattr(layer, "experts", None)
    if experts is None:
        raise AttributeError(f"Unable to resolve experts for layer {type(layer).__name__}")
    raw_scores: List[torch.Tensor] = []

    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        gate_up_proj = experts.gate_up_proj.detach()
        down_proj = experts.down_proj.detach()
        for expert_idx in range(gate_up_proj.shape[0]):
            gate_proj_weight, up_proj_weight = split_gate_up_proj(gate_up_proj[expert_idx])
            E = up_proj_weight.transpose(0, 1).contiguous()
            B = gate_proj_weight.transpose(0, 1).contiguous()
            G = down_proj[expert_idx].transpose(0, 1).contiguous()
            raw_scores.append(compute_expert_slanc_exact(gamma, E, B, G, eps=eps))
    else:
        for expert_layer in experts:
            gate_proj_weight = expert_layer.gate_proj.weight.detach()
            up_proj_weight = expert_layer.up_proj.weight.detach()
            down_proj_weight = expert_layer.down_proj.weight.detach()
            E = up_proj_weight.transpose(0, 1).contiguous()
            B = gate_proj_weight.transpose(0, 1).contiguous()
            G = down_proj_weight.transpose(0, 1).contiguous()
            raw_scores.append(compute_expert_slanc_exact(gamma, E, B, G, eps=eps))

    raw = torch.stack(raw_scores)
    return raw / (raw.mean() + eps)


def build_amp_table_for_model(model, eps: float = 1e-8) -> Dict[int, torch.Tensor]:
    amp_table: Dict[int, torch.Tensor] = {}
    for binding in iter_moe_layer_bindings(model):
        amp_table[binding.layer_idx] = build_amp_table_for_layer(
            binding.layer,
            experts=binding.experts,
            eps=eps,
        ).cpu()
    return amp_table


def route_qwen3_topk(router, hidden_states: torch.Tensor, top_k: int, norm_topk_prob: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router_out = router(hidden_states)
    if isinstance(router_out, tuple) and len(router_out) == 3:
        router_logits, routing_weights, selected_experts = router_out
        return router_logits, routing_weights, selected_experts

    if isinstance(router_out, tuple):
        if len(router_out) == 2 and torch.is_tensor(router_out[0]) and torch.is_tensor(router_out[1]):
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
    keep_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if keep_mask is None:
        active_mask = torch.ones_like(selected_experts, dtype=torch.bool)
    else:
        if keep_mask.shape != selected_experts.shape:
            raise ValueError(
                "keep_mask shape must match selected_experts shape, got "
                f"{tuple(keep_mask.shape)} vs {tuple(selected_experts.shape)}"
            )
        active_mask = keep_mask.to(dtype=torch.bool)

    tokens, top_k = selected_experts.shape
    hidden_dim = hidden_states.shape[-1]
    outputs = hidden_states.new_zeros((tokens, top_k, hidden_dim))
    active_positions = torch.nonzero(active_mask, as_tuple=False)
    if active_positions.numel() == 0:
        return outputs
    active_experts = selected_experts[active_mask]

    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        unique_experts = torch.unique(active_experts)
        for expert_idx in unique_experts.tolist():
            positions = active_positions[active_experts == expert_idx]
            if positions.numel() == 0:
                continue
            token_idx = positions[:, 0]
            slot_idx = positions[:, 1]
            current_state = hidden_states[token_idx]
            gate_up = F.linear(current_state, experts.gate_up_proj[expert_idx])
            gate_branch, up_branch = gate_up.chunk(2, dim=-1)
            current_hidden = experts.act_fn(gate_branch) * up_branch
            current_hidden = F.linear(current_hidden, experts.down_proj[expert_idx])
            outputs[token_idx, slot_idx] = current_hidden.to(outputs.dtype)
        return outputs

    unique_experts = torch.unique(active_experts)
    for expert_idx in unique_experts.tolist():
        positions = active_positions[active_experts == expert_idx]
        if positions.numel() == 0:
            continue
        token_idx = positions[:, 0]
        slot_idx = positions[:, 1]
        current_state = hidden_states[token_idx]
        current_hidden = experts[expert_idx](current_state)
        outputs[token_idx, slot_idx] = current_hidden.to(outputs.dtype)
    return outputs


def collect_layer_caches_for_naee(
    model,
    tokenizer,
    texts: Iterable[str],
    max_length: int = 512,
    token_limit: int = 256,
) -> Dict[int, Dict[str, torch.Tensor]]:
    accumulators: Dict[int, Dict[str, List[torch.Tensor] | int]] = {}
    originals: List[tuple[object, object]] = []

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        gate = getattr(mlp, "gate", None)
        if mlp is None or experts is None or gate is None:
            continue

        top_k = int(getattr(mlp, "top_k", getattr(model.config, "num_experts_per_tok", 2)))
        norm_topk_prob = bool(getattr(mlp, "norm_topk_prob", True))
        accumulators[layer_idx] = {
            "topk_idx": [],
            "gate": [],
            "expert_outs": [],
            "full_out": [],
            "token_limit": token_limit,
        }
        original_forward = mlp.forward

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

            current = accumulators[_layer_idx]
            current_tokens = sum(item.shape[0] for item in current["topk_idx"])
            if current_tokens < int(current["token_limit"]):
                remaining = int(current["token_limit"]) - current_tokens
                sl = slice(0, remaining)
                current["topk_idx"].append(selected_experts[sl].detach().cpu())
                current["gate"].append(routing_weights[sl].detach().cpu())
                current["expert_outs"].append(expert_outs[sl].detach().cpu())
                current["full_out"].append(full_out[sl].detach().cpu())

            shared_expert = getattr(self, "shared_expert", None)
            shared_expert_gate = getattr(self, "shared_expert_gate", None)
            if shared_expert is not None and shared_expert_gate is not None:
                shared_out = shared_expert(flat_states)
                shared_out = torch.sigmoid(shared_expert_gate(flat_states)) * shared_out
                final_hidden = (full_out + shared_out).view(batch_size, sequence_length, hidden_dim)
            else:
                final_hidden = full_out.view(batch_size, sequence_length, hidden_dim)
            return final_hidden

        originals.append((mlp, original_forward))
        mlp.forward = MethodType(_forward, mlp)

    if hasattr(model, "device"):
        device = model.device
    elif hasattr(model, "hf_device_map"):
        device = list(model.hf_device_map.values())[0]
    else:
        device = next(model.parameters()).device

    try:
        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with maybe_bf16_autocast():
                    model(**encoded, use_cache=False)
                if all(sum(item.shape[0] for item in acc["topk_idx"]) >= int(acc["token_limit"]) for acc in accumulators.values()):
                    break
    finally:
        for mlp, original_forward in originals:
            mlp.forward = original_forward

    built: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_idx, acc in accumulators.items():
        built[layer_idx] = {
            "topk_idx": torch.cat(acc["topk_idx"], dim=0) if acc["topk_idx"] else torch.empty(0, 0, dtype=torch.long),
            "gate": torch.cat(acc["gate"], dim=0) if acc["gate"] else torch.empty(0, 0),
            "expert_outs": torch.cat(acc["expert_outs"], dim=0) if acc["expert_outs"] else torch.empty(0, 0, 0),
            "full_out": torch.cat(acc["full_out"], dim=0) if acc["full_out"] else torch.empty(0, 0),
        }
    return built


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NAEE ablation on Qwen3-MoE-30B with beta-threshold pruning.",
    )
    add_model_selection_args(parser)
    parser.add_argument(
        "--method",
        choices=["naee", "score_only"],
        default="naee",
        help="naee uses AMP-aware score=gate*amp; score_only preserves the old gate-only relative-beta baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_NAEE_OUTPUT_DIR,
    )
    parser.add_argument(
        "--beta-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mathqa", "openbookqa", "arc_challenge"],
        choices=sorted(DATASET_SPECS.keys()),
    )
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--calibration-limit-per-dataset", type=int, default=128)
    parser.add_argument("--calibration-source", choices=["wikitext"], default="wikitext")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-max-length", type=int, default=2048)
    parser.add_argument("--calibration-token-limit", type=int, default=2048)
    parser.add_argument("--ppl-split", type=str, default="test")
    parser.add_argument("--ppl-text-column", type=str, default="text")
    parser.add_argument("--ppl-min-text-length", type=int, default=512)
    parser.add_argument("--ppl-n-ctx", type=int, default=DEFAULT_PPL_SEQ_LEN)
    parser.add_argument("--ppl-n-batch", type=int, default=DEFAULT_PPL_SEQ_LEN)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    return finalize_model_selection(parser.parse_args())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_existing_artifact(path: Path, default):
    candidates = [path]
    if path.suffix == ".json":
        candidates.append(path.with_name(f"{path.stem}.partial{path.suffix}"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(default, list) and isinstance(payload, list):
            return payload
        if isinstance(default, dict) and isinstance(payload, dict):
            return payload
    return default


def format_tau_dir(value: float) -> str:
    return f"tau_{float(value):.4f}"


def beta_row_exists(rows: Sequence[Mapping[str, float]], beta: float, eps: float = 1e-9) -> bool:
    return any(abs(float(row.get("beta", float("nan"))) - float(beta)) <= eps for row in rows)


def missing_betas(beta_grid: Sequence[float], rows: Sequence[Mapping[str, float]]) -> List[float]:
    return [float(beta) for beta in beta_grid if not beta_row_exists(rows, float(beta))]


def merge_rows_for_beta_grid(
    beta_grid: Sequence[float],
    existing_rows: Sequence[Mapping[str, float]],
    new_rows: Sequence[Mapping[str, float]],
) -> List[Dict[str, float]]:
    merged_by_beta: Dict[float, Dict[str, float]] = {}
    for row in existing_rows:
        if "beta" in row:
            merged_by_beta[float(row["beta"])] = dict(row)
    for row in new_rows:
        if "beta" in row:
            merged_by_beta[float(row["beta"])] = dict(row)
    return [merged_by_beta[float(beta)] for beta in beta_grid if float(beta) in merged_by_beta]


def load_amp_table_summary(path: Path) -> Dict[int, torch.Tensor] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return {int(layer_idx): torch.tensor(values) for layer_idx, values in payload.items()}


def calibration_covers_grid(
    calibration_summary: Mapping[str, Mapping[str, float]],
    beta_grid: Sequence[float],
) -> bool:
    return all(str(float(beta)) in calibration_summary or str(beta) in calibration_summary for beta in beta_grid)


def load_existing_ppl_rows(output_dir: Path, beta_grid: Sequence[float]) -> List[Dict[str, float]]:
    rows = load_existing_artifact(output_dir / "wikitext_ppl.json", [])
    merged = list(rows)
    for beta in beta_grid:
        per_tau_path = output_dir / "ppl_by_tau" / format_tau_dir(float(beta)) / "wikitext_ppl.json"
        per_tau_row = load_existing_artifact(per_tau_path, {})
        if per_tau_row and not beta_row_exists(merged, float(beta)):
            merged.append(per_tau_row)
    return merge_rows_for_beta_grid(beta_grid, merged, [])


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    print(f"[{timestamp}] {message}", flush=True)


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "-"
    return f"{value:.{digits}f}"


def build_relative_beta_keep_mask(
    score: torch.Tensor,
    beta: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if score.ndim != 2:
        raise ValueError(f"Expected [tokens, top_k] score tensor, got shape={tuple(score.shape)}")
    score_max = score.max(dim=-1, keepdim=True).values
    keep_mask = score >= (score_max + eps) * float(beta)
    best_idx = score.argmax(dim=-1, keepdim=True)
    keep_mask.scatter_(1, best_idx, True)
    return keep_mask


def summarize_keep_mask(keep_mask: torch.Tensor) -> Dict[str, float]:
    keep_mask_f = keep_mask.float()
    return {
        "pruning_ratio": float(1.0 - keep_mask_f.mean().item()),
        "avg_active_experts": float(keep_mask_f.sum(dim=-1).mean().item()),
    }


def compute_naee_score(
    gate: torch.Tensor,
    amp_selected: torch.Tensor | None = None,
    score_mode: str = "gate",
) -> torch.Tensor:
    """Compute the pruning score for NAEE or its score-only baseline.

    ``score_mode='amp'`` is the NAEE path: routing probability is modulated by
    per-expert static importance/AMP. ``score_mode='gate'`` preserves the old
    gate-only behavior and is reported as ``score_only``.
    """
    normalized_mode = str(score_mode).lower()
    if normalized_mode in {"gate", "score_only"}:
        return gate
    if normalized_mode in {"amp", "naee"}:
        if amp_selected is None:
            raise ValueError("amp_selected is required for NAEE AMP-aware scoring")
        return compute_importance_score(gate, amp_selected)
    raise ValueError(f"Unsupported NAEE score_mode: {score_mode}")


def moe_forward_with_naee_pruning(
    hidden_states: torch.Tensor,
    router,
    experts,
    amp_layer: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
    beta: float,
    score_mode: str = "amp",
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
    amp_selected = amp_layer.to(gate.device)[selected_experts]
    score = compute_naee_score(gate, amp_selected=amp_selected, score_mode=score_mode)
    keep_mask = build_relative_beta_keep_mask(score, beta=beta, eps=eps)
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
        "router_logits": router_logits.detach(),
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


def moe_experts_forward_with_naee_pruning(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    amp_layer: torch.Tensor,
    beta: float,
    score_mode: str = "amp",
    eps: float = 1e-8,
    return_aux: bool = False,
    moe_backend: str = "triton",
):
    amp_selected = amp_layer.to(routing_weights.device)[selected_experts]
    score = compute_naee_score(routing_weights, amp_selected=amp_selected, score_mode=score_mode)
    keep_mask = build_relative_beta_keep_mask(score, beta=beta, eps=eps)
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
        "router_logits": None,
        "topk_idx": selected_experts.detach(),
        "gate": routing_weights.detach(),
        "amp_selected": amp_selected.detach(),
        "score": score.detach(),
        "keep_mask": keep_mask.detach(),
        "gate_kept": gate_kept.detach(),
        "expert_outs": None if expert_outputs is None else expert_outputs.detach(),
        "full_out": full_out,
    }
    if beta == 0.0:
        final_hidden = aux["full_out"] if aux["full_out"] is not None else final_hidden
    if return_aux:
        return final_hidden, aux
    return final_hidden


@contextmanager
def patch_qwen3_moe_blocks_naee(
    model,
    amp_table: Mapping[int, torch.Tensor],
    beta_by_layer: Mapping[int, float],
    score_mode: str = "amp",
    runtime_stats: RuntimeStats | None = None,
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
        beta = float(beta_by_layer.get(layer_idx, 0.0))
        if binding.kind == "mlp":
            top_k = binding.top_k
            norm_topk_prob = binding.norm_topk_prob

            def _forward(
                self,
                hidden_states,
                _layer_idx=layer_idx,
                _amp_layer=amp_layer,
                _beta=beta,
                _score_mode=score_mode,
                _top_k=top_k,
                _norm_topk_prob=norm_topk_prob,
            ):
                output, aux = moe_forward_with_naee_pruning(
                    hidden_states=hidden_states,
                    router=self.gate,
                    experts=self.experts,
                    amp_layer=_amp_layer,
                    top_k=_top_k,
                    norm_topk_prob=_norm_topk_prob,
                    beta=_beta,
                    score_mode=_score_mode,
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
                _amp_layer=amp_layer,
                _beta=beta,
                _score_mode=score_mode,
            ):
                output, aux = moe_experts_forward_with_naee_pruning(
                    hidden_states=hidden_states,
                    experts=self,
                    selected_experts=top_k_index,
                    routing_weights=top_k_weights,
                    amp_layer=_amp_layer,
                    beta=_beta,
                    score_mode=_score_mode,
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
def patched_model_for_naee(
    model,
    amp_table: Mapping[int, torch.Tensor],
    beta: float,
    score_mode: str = "amp",
    runtime_stats: RuntimeStats | None = None,
    moe_backend: str = "triton",
):
    if float(beta) <= 0.0:
        yield model
        return
    beta_by_layer = build_uniform_tau_by_layer(amp_table.keys(), beta)
    with patch_qwen3_moe_blocks_naee(
        model=model,
        amp_table=amp_table,
        beta_by_layer=beta_by_layer,
        score_mode=score_mode,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
    ):
        yield model


def calibration_metrics_for_beta(
    layer_cache: Mapping[str, torch.Tensor],
    amp_layer: torch.Tensor,
    beta: float,
    score_mode: str = "amp",
    eps: float = 1e-8,
) -> Dict[str, float]:
    topk_idx = layer_cache["topk_idx"].long()
    gate = layer_cache["gate"].float()
    expert_outs = layer_cache["expert_outs"].float()
    full_out = layer_cache["full_out"].float()

    amp_selected = amp_layer[topk_idx].float()
    score = compute_naee_score(gate, amp_selected=amp_selected, score_mode=score_mode)
    keep_mask = build_relative_beta_keep_mask(score, beta=beta, eps=eps)
    gate_kept = renorm_gate_after_pruning(gate, keep_mask, eps=eps)
    pred_out = (gate_kept.unsqueeze(-1) * expert_outs).sum(dim=1)
    rel_mse = (
        ((pred_out - full_out).pow(2).sum(dim=-1))
        / (full_out.pow(2).sum(dim=-1) + eps)
    ).mean().item()

    return {
        "beta": float(beta),
        "rel_mse": float(rel_mse),
        **summarize_keep_mask(keep_mask),
    }


def aggregate_calibration_summary(
    layer_caches: Mapping[int, Mapping[str, torch.Tensor]],
    amp_table: Mapping[int, torch.Tensor],
    beta_grid: Sequence[float],
    score_mode: str = "amp",
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    valid_layers = [
        layer_idx
        for layer_idx, cache in layer_caches.items()
        if layer_idx in amp_table and cache["topk_idx"].numel() > 0
    ]
    for beta in beta_grid:
        per_layer = [
            calibration_metrics_for_beta(
                layer_caches[layer_idx],
                amp_table[layer_idx],
                beta,
                score_mode=score_mode,
            )
            for layer_idx in valid_layers
        ]
        if not per_layer:
            summary[str(beta)] = {
                "beta": float(beta),
                "layer_count": 0.0,
                "rel_mse": 0.0,
                "pruning_ratio": 0.0,
                "avg_active_experts": 0.0,
            }
            continue

        summary[str(beta)] = {
            "beta": float(beta),
            "layer_count": float(len(per_layer)),
            "rel_mse": float(sum(item["rel_mse"] for item in per_layer) / len(per_layer)),
            "pruning_ratio": float(sum(item["pruning_ratio"] for item in per_layer) / len(per_layer)),
            "avg_active_experts": float(sum(item["avg_active_experts"] for item in per_layer) / len(per_layer)),
        }
    return summary


def aggregate_proxy_summary(
    layer_caches: Mapping[int, Mapping[str, torch.Tensor]],
    amp_table: Mapping[int, torch.Tensor],
) -> Dict[str, float]:
    metrics = []
    for layer_idx, cache in layer_caches.items():
        if layer_idx not in amp_table or cache["topk_idx"].numel() == 0:
            continue
        metrics.append(proxy_validity_metrics(cache, amp_table[layer_idx]))
    if not metrics:
        return {
            "layer_count": 0.0,
            "gate_corr": 0.0,
            "amp_corr": 0.0,
            "proxy_corr": 0.0,
        }
    return {
        "layer_count": float(len(metrics)),
        "gate_corr": float(sum(item.gate_corr for item in metrics) / len(metrics)),
        "amp_corr": float(sum(item.amp_corr for item in metrics) / len(metrics)),
        "proxy_corr": float(sum(item.proxy_corr for item in metrics) / len(metrics)),
    }


def build_calibration_markdown(summary: Mapping[str, Mapping[str, float]]) -> str:
    lines = [
        "# NAEE calibration summary",
        "",
        "| beta | rel_mse | pruning_ratio | avg_active_experts | layer_count |",
        "| --- | --- | --- | --- | --- |",
    ]
    for beta_key in sorted(summary.keys(), key=float):
        row = summary[beta_key]
        lines.append(
            "| "
            + " | ".join(
                [
                    format_float(float(beta_key)),
                    format_float(row.get("rel_mse")),
                    format_float(row.get("pruning_ratio")),
                    format_float(row.get("avg_active_experts")),
                    format_float(row.get("layer_count")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_results_markdown(rows: Sequence[Mapping[str, float]], datasets: Sequence[str]) -> str:
    headers = ["beta", *[DATASET_ALIASES[name] for name in datasets], "avg_dynamic_pruning_ratio"]
    lines = [
        "# NAEE MCQA summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [format_float(row["beta"])]
        values.extend(format_float(row[name]) for name in datasets)
        values.append(format_float(row["avg_dynamic_pruning_ratio"]))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_ppl_markdown(rows: Sequence[Mapping[str, float]]) -> str:
    lines = [
        "# NAEE WikiText-2 PPL summary",
        "",
        "| beta | ppl | avg_dynamic_pruning_ratio | rows_used | windows |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_float(row["beta"]),
                    format_float(row["ppl"]),
                    format_float(row["avg_dynamic_pruning_ratio"]),
                    format_float(row["rows_used"]),
                    format_float(row["windows"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_final_summary(
    beta_grid: Sequence[float],
    datasets: Sequence[str],
    evaluation_rows: Sequence[Mapping[str, float]],
    ppl_rows: Sequence[Mapping[str, float]],
    calibration_summary: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, float | None]]:
    evaluation_by_beta = {float(row["beta"]): row for row in evaluation_rows}
    ppl_by_beta = {float(row["beta"]): row for row in ppl_rows}
    rows: List[Dict[str, float | None]] = []
    for beta in beta_grid:
        beta = float(beta)
        row: Dict[str, float | None] = {"beta": beta}
        eval_row = evaluation_by_beta.get(beta, {})
        ppl_row = ppl_by_beta.get(beta, {})
        calib_row = calibration_summary.get(str(beta), {})
        for dataset_name in datasets:
            value = eval_row.get(dataset_name)
            row[dataset_name] = float(value) if value is not None else None
        value = eval_row.get("avg_dynamic_pruning_ratio")
        row["avg_dynamic_pruning_ratio"] = float(value) if value is not None else None
        value = ppl_row.get("ppl")
        row["wikitext_ppl"] = float(value) if value is not None else None
        value = calib_row.get("rel_mse")
        row["calibration_rel_mse"] = float(value) if value is not None else None
        value = calib_row.get("avg_active_experts")
        row["calibration_avg_active_experts"] = float(value) if value is not None else None
        rows.append(row)
    return rows


def build_final_report_markdown(
    final_rows: Sequence[Mapping[str, float]],
    datasets: Sequence[str],
    args: argparse.Namespace,
    proxy_summary: Mapping[str, float],
) -> str:
    headers = [
        "beta",
        "avg_dynamic_pruning_ratio",
        "wikitext_ppl",
        *[DATASET_ALIASES[name] for name in datasets],
        "calib_rel_mse",
        "calib_avg_active",
    ]
    lines = [
        "# Qwen3-MoE-30B NAEE ablation report",
        "",
        "## Run configuration",
        "",
        f"- Model: `{args.model_path}`",
        f"- Output dir: `{args.output_dir}`",
        f"- Beta grid: `{list(map(float, args.beta_grid))}`",
        f"- Datasets: `{list(args.datasets)}`",
        f"- Eval limit: `{args.eval_limit}`",
        f"- Eval batch size: `{args.eval_batch_size}`",
        f"- Calibration token limit: `{args.calibration_token_limit}`",
        "",
        "## Proxy summary",
        "",
        f"- Layers with caches: {format_float(proxy_summary.get('layer_count'))}",
        f"- gate_corr: {format_float(proxy_summary.get('gate_corr'))}",
        f"- amp_corr: {format_float(proxy_summary.get('amp_corr'))}",
        f"- proxy_corr: {format_float(proxy_summary.get('proxy_corr'))}",
        "",
        "## Per-beta summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in final_rows:
        values = [
            format_float(row["beta"]),
            format_float(row["avg_dynamic_pruning_ratio"]),
            format_float(row["wikitext_ppl"]),
        ]
        values.extend(format_float(row[name]) for name in datasets)
        values.extend(
            [
                format_float(row["calibration_rel_mse"]),
                format_float(row["calibration_avg_active_experts"]),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def write_repro_commands(output_dir: Path, argv: Sequence[str]) -> None:
    rendered = "python -m moe_prune.code.scripts.NAEE.run_naee_ablation"
    if argv:
        rendered += " " + shlex.join(list(argv))
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Reproduce the NAEE Qwen3-MoE ablation",
            rendered,
            "",
        ]
    )
    write_text(output_dir / "repro_commands.sh", script)
    os.chmod(output_dir / "repro_commands.sh", 0o755)


def calibration_cache_path(output_dir: Path, args: argparse.Namespace, score_mode: str) -> Path:
    return output_dir / (
        "calibration_cache_"
        f"{model_tag_for_path(args.model_path)}_"
        f"{score_mode}_{args.calibration_source}_{args.calibration_split}_"
        f"samples{args.calibration_limit_per_dataset}_"
        f"max{args.calibration_max_length}_tokens{args.calibration_token_limit}.pt"
    )


def build_continuous_wikitext_calibration_windows(
    tokenizer,
    split: str,
    num_windows: int,
    window_length: int,
) -> List[str]:
    """Return consecutive WikiText token windows for calibration.

    The repository convention for pruning calibration is a contiguous
    WikiText-train segment of ``128 * 2048`` tokens.  The shared cache
    collector accepts text snippets, so we materialize contiguous token windows
    and decode them back to text before collection.
    """

    clear_hf_proxy_env()
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(str(example["text"]) for example in dataset if str(example["text"]).strip())
    tokenizer.model_max_length = 2**31 - 1
    token_ids = tokenizer(text, truncation=False, return_tensors="pt").input_ids[0]
    required_tokens = int(num_windows) * int(window_length)
    if token_ids.numel() < required_tokens:
        raise ValueError(
            f"WikiText split={split!r} only has {token_ids.numel()} tokens; "
            f"need {required_tokens} for {num_windows}x{window_length} calibration."
        )
    token_ids = token_ids[:required_tokens]
    return [
        tokenizer.decode(token_ids[start:start + window_length], skip_special_tokens=True)
        for start in range(0, required_tokens, window_length)
    ]


def main() -> None:
    args = parse_args()
    score_mode = "gate" if args.method == "score_only" else "amp"
    output_dir = resolve_naee_output_dir(
        Path(args.output_dir),
        method=args.method,
        model_path=args.model_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"starting {args.method} ablation -> output_dir={output_dir} score_mode={score_mode}")
    log(f"beta_grid={list(map(float, args.beta_grid))} datasets={list(args.datasets)}")

    write_repro_commands(output_dir, os.sys.argv[1:])
    previous_method_config = load_existing_artifact(output_dir / "method_config.json", {})
    write_json(
        output_dir / "method_config.json",
        {
            "method": args.method,
            "model_family": args.model_family,
            "model_path": args.model_path,
            "score_mode": score_mode,
            "beta_grid": [float(beta) for beta in args.beta_grid],
            "datasets": list(args.datasets),
            "eval_limit": args.eval_limit,
            "eval_batch_size": args.eval_batch_size,
            "calibration_source": args.calibration_source,
            "calibration_split": args.calibration_split,
            "calibration_limit_per_dataset": args.calibration_limit_per_dataset,
            "calibration_max_length": args.calibration_max_length,
            "calibration_token_limit": args.calibration_token_limit,
            "ppl_split": args.ppl_split,
            "ppl_text_column": args.ppl_text_column,
            "ppl_min_text_length": args.ppl_min_text_length,
            "ppl_n_ctx": args.ppl_n_ctx,
            "ppl_n_batch": args.ppl_n_batch,
        },
    )

    existing_calibration_summary = load_existing_artifact(output_dir / "calibration_summary.json", {})
    existing_proxy_summary = load_existing_artifact(output_dir / "proxy_summary.json", {})
    existing_method_config = previous_method_config
    existing_evaluation_rows = load_existing_artifact(output_dir / "results_table.json", [])
    existing_ppl_rows = load_existing_ppl_rows(output_dir, args.beta_grid)

    eval_betas_to_run = [] if args.skip_eval else missing_betas(args.beta_grid, existing_evaluation_rows)
    ppl_betas_to_run = [] if args.skip_ppl else missing_betas(args.beta_grid, existing_ppl_rows)
    calibration_ready = (
        bool(existing_calibration_summary)
        and bool(existing_proxy_summary)
        and existing_method_config.get("model_family") == args.model_family
        and existing_method_config.get("model_path") == args.model_path
        and calibration_covers_grid(existing_calibration_summary, args.beta_grid)
        and existing_method_config.get("score_mode") == score_mode
        and existing_method_config.get("calibration_source") == args.calibration_source
        and existing_method_config.get("calibration_split") == args.calibration_split
    )
    should_run_calibration = (not args.skip_calibration) and (not calibration_ready)
    needs_model = should_run_calibration or bool(eval_betas_to_run) or bool(ppl_betas_to_run)

    model = None
    tokenizer = None
    amp_table: Dict[int, torch.Tensor] = {}
    if needs_model:
        log(f"loading model/tokenizer from {args.model_path}")
        model, tokenizer = load_qwen3_moe(args.model_path, model_family=args.model_family)
        log("model/tokenizer loaded")
        amp_table_path = output_dir / "amp_table_summary.json"
        loaded_amp_table = load_amp_table_summary(amp_table_path)
        if loaded_amp_table is None:
            log("building amp table")
            amp_table = build_shared_amp_table_for_model(model)
            log(f"amp table built for {len(amp_table)} layers")
            write_json(
                amp_table_path,
                {str(layer_idx): amp.tolist() for layer_idx, amp in amp_table.items()},
            )
        else:
            amp_table = loaded_amp_table
            log(f"reusing existing amp table from {amp_table_path} ({len(amp_table)} layers)")
    else:
        log("all requested beta outputs already exist; skipping model load")

    calibration_summary: Dict[str, Dict[str, float]] = dict(existing_calibration_summary)
    proxy_summary: Dict[str, float] = dict(existing_proxy_summary)
    if should_run_calibration:
        assert model is not None and tokenizer is not None
        cache_path = calibration_cache_path(output_dir, args, score_mode)
        if cache_path.exists():
            log(f"reusing calibration tensor cache from {cache_path}")
            layer_caches = torch.load(cache_path, map_location="cpu")
        else:
            log("starting calibration cache collection")
            calibration_texts = build_continuous_wikitext_calibration_windows(
                tokenizer=tokenizer,
                split=args.calibration_split,
                num_windows=args.calibration_limit_per_dataset,
                window_length=args.calibration_max_length,
            )
            layer_caches = collect_shared_layer_caches(
                model=model,
                tokenizer=tokenizer,
                texts=calibration_texts,
                max_length=args.calibration_max_length,
                token_limit=args.calibration_token_limit,
            )
            torch.save(layer_caches, cache_path)
            log(f"calibration tensor cache written to {cache_path}")
        calibration_summary = aggregate_calibration_summary(layer_caches, amp_table, args.beta_grid, score_mode=score_mode)
        proxy_summary = aggregate_proxy_summary(layer_caches, amp_table)
        write_json(output_dir / "calibration_summary.json", calibration_summary)
        write_text(output_dir / "calibration_summary.md", build_calibration_markdown(calibration_summary))
        write_json(output_dir / "proxy_summary.json", proxy_summary)
        log("calibration/proxy summaries written")
    else:
        if args.skip_calibration:
            log("skip-calibration enabled; reusing existing calibration/proxy artifacts when present")
        elif calibration_ready:
            log("reusing existing calibration/proxy artifacts")
        if not (output_dir / "calibration_summary.json").exists():
            write_json(output_dir / "calibration_summary.json", calibration_summary)
        if not (output_dir / "proxy_summary.json").exists():
            write_json(output_dir / "proxy_summary.json", proxy_summary)

    evaluation_rows: List[Dict[str, float]] = list(existing_evaluation_rows)
    if not args.skip_eval:
        if not eval_betas_to_run:
            log("all requested eval beta rows already exist; skipping eval")
        new_evaluation_rows: List[Dict[str, float]] = []
        for beta in eval_betas_to_run:
            assert model is not None and tokenizer is not None
            beta = float(beta)
            log(f"starting eval beta={beta:.4f}")
            runtime_stats = RuntimeStats()
            beta_results: Dict[str, Dict[str, float]] = {}
            with patched_model_for_naee(model, amp_table=amp_table, beta=beta, score_mode=score_mode, runtime_stats=runtime_stats):
                for dataset_name in args.datasets:
                    log(f"eval beta={beta:.4f} dataset={dataset_name} start")
                    metrics = evaluate_mcqa_dataset(
                        model=model,
                        tokenizer=tokenizer,
                        dataset_name=dataset_name,
                        limit=args.eval_limit,
                        batch_size=args.eval_batch_size,
                    )
                    metrics["mean_pruning_ratio"] = runtime_stats.mean_pruning_ratio()
                    beta_results[dataset_name] = metrics
                    log(
                        "eval beta={:.4f} dataset={} done accuracy={:.4f} mean_pruning_ratio={:.4f}".format(
                            beta,
                            dataset_name,
                            float(metrics["accuracy"]),
                            float(metrics["mean_pruning_ratio"]),
                        )
                    )
            write_json(output_dir / f"results_beta_{beta}.json", beta_results)

            pruning_values = [item["mean_pruning_ratio"] for item in beta_results.values()]
            row: Dict[str, float] = {
                "beta": beta,
                "method": args.method,
                "avg_dynamic_pruning_ratio": (
                    float(sum(pruning_values) / len(pruning_values)) if pruning_values else 0.0
                ),
            }
            for dataset_name in args.datasets:
                row[dataset_name] = float(beta_results[dataset_name]["accuracy"])
            new_evaluation_rows.append(row)
            evaluation_rows = merge_rows_for_beta_grid(args.beta_grid, evaluation_rows, new_evaluation_rows)
            write_json(output_dir / "results_table.partial.json", evaluation_rows)
            write_text(output_dir / "results_table.partial.md", build_results_markdown(evaluation_rows, args.datasets))
            log(f"results table partial written for beta={beta:.4f}")

        evaluation_rows = merge_rows_for_beta_grid(args.beta_grid, evaluation_rows, new_evaluation_rows)
        write_json(output_dir / "results_table.json", evaluation_rows)
        write_text(output_dir / "results_table.md", build_results_markdown(evaluation_rows, args.datasets))
        log("final results tables written")
    else:
        if not (output_dir / "results_table.json").exists():
            write_json(output_dir / "results_table.json", evaluation_rows)

    ppl_rows: List[Dict[str, float]] = list(existing_ppl_rows)
    if not args.skip_ppl:
        if not ppl_betas_to_run:
            log("all requested WikiText PPL beta rows already exist; skipping PPL")
        new_ppl_rows: List[Dict[str, float]] = []
        for beta in ppl_betas_to_run:
            assert model is not None and tokenizer is not None
            beta = float(beta)
            log(f"starting WikiText PPL beta={beta:.4f}")
            evaluator = FullWikiTextPerplexity(
                model=model,
                tokenizer=tokenizer,
                split=args.ppl_split,
                text_column=args.ppl_text_column,
                min_text_length=args.ppl_min_text_length,
            )
            runtime_stats = RuntimeStats()
            with patched_model_for_naee(model, amp_table=amp_table, beta=beta, score_mode=score_mode, runtime_stats=runtime_stats):
                metrics = evaluator.calculate_corpus_ppl(n_ctx=args.ppl_n_ctx, n_batch=args.ppl_n_batch)
            ppl_row = {
                "beta": beta,
                "method": args.method,
                "ppl": float(metrics["ppl"]),
                "avg_dynamic_pruning_ratio": float(runtime_stats.mean_pruning_ratio()),
                "rows_used": float(evaluator.num_rows),
                "windows": float(metrics["windows"]),
            }
            per_tau_dir = output_dir / "ppl_by_tau" / format_tau_dir(beta)
            write_json(per_tau_dir / "wikitext_ppl.json", ppl_row)
            write_text(per_tau_dir / "wikitext_ppl.md", build_ppl_markdown([ppl_row]))
            new_ppl_rows.append(ppl_row)
            ppl_rows = merge_rows_for_beta_grid(args.beta_grid, ppl_rows, new_ppl_rows)
            write_json(output_dir / "wikitext_ppl.partial.json", ppl_rows)
            write_text(output_dir / "wikitext_ppl.partial.md", build_ppl_markdown(ppl_rows))
            log(
                "wikitext ppl beta={:.4f} done ppl={:.4f} avg_dynamic_pruning_ratio={:.4f}".format(
                    beta,
                    float(ppl_row["ppl"]),
                    float(ppl_row["avg_dynamic_pruning_ratio"]),
                )
            )

        ppl_rows = merge_rows_for_beta_grid(args.beta_grid, ppl_rows, new_ppl_rows)
        write_json(output_dir / "wikitext_ppl.json", ppl_rows)
        write_text(output_dir / "wikitext_ppl.md", build_ppl_markdown(ppl_rows))
        log("final WikiText PPL tables written")
    else:
        if not (output_dir / "wikitext_ppl.json").exists():
            write_json(output_dir / "wikitext_ppl.json", ppl_rows)

    final_rows = build_final_summary(
        beta_grid=args.beta_grid,
        datasets=args.datasets,
        evaluation_rows=evaluation_rows,
        ppl_rows=ppl_rows,
        calibration_summary=calibration_summary,
    )
    write_json(output_dir / "final_summary.json", final_rows)
    write_text(
        output_dir / "final_report.md",
        build_final_report_markdown(final_rows, args.datasets, args, proxy_summary),
    )
    log("final summary/report written")


if __name__ == "__main__":
    main()
