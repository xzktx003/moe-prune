"""Qwen3-MoE adaptation of DiEP (Differentiable Expert Pruning) dynamic skipping.

This script implements the DiEP paper's ``dynamic_skipping`` mode, adapted to
Qwen3-MoE (top_k=8) instead of Mixtral (top_k=2). The workflow is:

1. **Calibration** (runs once, cached to disk): For each MoE layer, collect the
   ratio ``r = p[rank1] / p[rank0]`` of the two strongest routing probabilities
   for every calibration token. The per-layer *median* of these ratios is the
   layer's static score ``beta_layer`` (the original DiEP per-layer threshold).
   The resulting score dict is pickled so subsequent runs reuse it.

2. **Dynamic skipping at inference** for each token:
     - compute ``r = softmax(router_logits)[rank1] / softmax(router_logits)[rank0]``
     - if ``r < tau * beta_layer`` -> keep only the top-1 expert (drop rank>=1)
     - else -> keep all top-k experts

3. **tau grid evaluation**: Sweeps ``tau`` values, computing WikiText-2 PPL and
   the realised average expert pruning ratio for each.

The script mirrors the conventions of the sibling ablations
the shared ACE evaluation utilities and
``code/run_naee_ablation.py`` so it plugs into the existing result reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moe_prune.code.src.model_adapter import (  # noqa: E402
    clear_hf_proxy_env,
    load_qwen3_moe,
    maybe_bf16_autocast,
)
from moe_prune.code.src.model_families import (  # noqa: E402
    add_model_selection_args,
    finalize_model_selection,
)
from moe_prune.code.src.model_structure import iter_moe_layer_bindings  # noqa: E402
from moe_prune.code.src.runtime_pruner import (  # noqa: E402
    RuntimeStats,
    compute_optional_shared_expert_output,
    compute_moe_weighted_hidden_states,
    compute_expert_outputs,
    renorm_gate_after_pruning,
    route_qwen3_topk,
)

EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Calibration text loading
# ---------------------------------------------------------------------------


def load_wikitext_calibration_text(
    split: str = "train",
    text_column: str = "text",
    min_text_length: int = 0,
    row_limit: Optional[int] = None,
) -> tuple[str, int]:
    clear_hf_proxy_env()
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts: List[str] = []
    used_rows = 0
    for sample in data:
        text = sample[text_column]
        if len(text) < min_text_length:
            continue
        texts.append("\n" if text == "" else text)
        used_rows += 1
        if row_limit is not None and used_rows >= row_limit:
            break
    return "".join(texts), used_rows


def load_wikitext_eval_text(
    split: str = "test",
    text_column: str = "text",
    min_text_length: int = 512,
) -> tuple[str, int]:
    clear_hf_proxy_env()
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts: List[str] = []
    used_rows = 0
    for sample in data:
        text = sample[text_column]
        if len(text) < min_text_length:
            continue
        texts.append(" \n" if text == "" else text)
        used_rows += 1
    return "".join(texts), used_rows


# ---------------------------------------------------------------------------
# Calibration: per-layer top2/top1 ratio collection
# ---------------------------------------------------------------------------


@dataclass
class LayerRatioAccumulator:
    layer_idx: int
    top_k: int
    norm_topk_prob: bool
    ratios: List[torch.Tensor]

    def append(self, ratio: torch.Tensor) -> None:
        self.ratios.append(ratio.detach().float().cpu())

    def median(self) -> float:
        if not self.ratios:
            return 0.0
        flat = torch.cat(self.ratios).flatten()
        if flat.numel() == 0:
            return 0.0
        return float(torch.median(flat).item())

    def mean(self) -> float:
        if not self.ratios:
            return 0.0
        flat = torch.cat(self.ratios).flatten()
        if flat.numel() == 0:
            return 0.0
        return float(flat.mean().item())

    def count(self) -> int:
        return int(sum(item.numel() for item in self.ratios))


@contextmanager
def patch_qwen3_moe_blocks_for_calibration(
    model,
    accumulators: Dict[int, LayerRatioAccumulator],
):
    originals: List[tuple[object, object]] = []
    for binding in iter_moe_layer_bindings(model):
        if binding.router is None:
            continue

        layer_idx = binding.layer_idx
        top_k = binding.top_k
        norm_topk_prob = binding.norm_topk_prob
        accumulators[layer_idx] = LayerRatioAccumulator(
            layer_idx=layer_idx,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            ratios=[],
        )
        patch_target = binding.patch_target
        original_forward = patch_target.forward

        if binding.kind == "mlp":
            def _forward(self, hidden_states, _layer_idx=layer_idx):
                batch_size, sequence_length, hidden_dim = hidden_states.shape
                flat = hidden_states.view(-1, hidden_dim)
                router_logits = self.gate(flat)
                if isinstance(router_logits, tuple):
                    router_logits = router_logits[0]
                full_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
                current_top_k = accumulators[_layer_idx].top_k
                sorted_probs, _ = torch.sort(full_probs, dim=-1, descending=True)
                denom = sorted_probs[:, 0].clamp_min(EPSILON)
                ratio = sorted_probs[:, 1] / denom
                accumulators[_layer_idx].append(ratio)

                routing_weights, selected_experts = torch.topk(full_probs, current_top_k, dim=-1)
                if accumulators[_layer_idx].norm_topk_prob:
                    routing_weights = routing_weights / routing_weights.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(EPSILON)
                routing_weights = routing_weights.to(flat.dtype)
                expert_outputs = compute_expert_outputs(flat, self.experts, selected_experts)
                final = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
                shared_output = compute_optional_shared_expert_output(
                    flat,
                    shared_expert=getattr(self, "shared_expert", None),
                    shared_expert_gate=getattr(self, "shared_expert_gate", None),
                )
                if shared_output is not None:
                    final = final + shared_output
                final = final.view(batch_size, sequence_length, hidden_dim)
                return final
        else:
            router = binding.router

            def _forward(self, hidden_states, top_k_index, top_k_weights, _layer_idx=layer_idx, _router=router):
                router_logits = _router(hidden_states)
                if isinstance(router_logits, tuple):
                    router_logits = router_logits[0]
                full_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
                sorted_probs, _ = torch.sort(full_probs, dim=-1, descending=True)
                denom = sorted_probs[:, 0].clamp_min(EPSILON)
                ratio = sorted_probs[:, 1] / denom
                accumulators[_layer_idx].append(ratio)
                expert_outputs = compute_expert_outputs(hidden_states, self, top_k_index)
                return (top_k_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)

        originals.append((patch_target, original_forward))
        patch_target.forward = MethodType(_forward, patch_target)

    try:
        yield
    finally:
        for patch_target, original_forward in originals:
            patch_target.forward = original_forward


def _resolve_device(model):
    if hasattr(model, "device") and model.device.type != "meta":
        return model.device
    if hasattr(model, "hf_device_map"):
        for mapped in model.hf_device_map.values():
            if mapped not in ("cpu", "disk"):
                return mapped
    return next(model.parameters()).device


def calibrate_layer_scores(
    model,
    tokenizer,
    num_samples: int,
    max_length: int,
    calibration_split: str = "train",
) -> Dict[str, object]:
    """Run DiEP calibration and return a dict ready to be pickled."""

    text, used_rows = load_wikitext_calibration_text(split=calibration_split)
    tokenizer.model_max_length = 2**31 - 1
    enc = tokenizer(text, truncation=False, return_tensors="pt")
    token_ids = enc.input_ids
    required_tokens = int(num_samples) * int(max_length)
    available_tokens = int(token_ids.size(1))
    if available_tokens < required_tokens:
        raise RuntimeError(
            "Insufficient WikiText calibration tokens: "
            f"need {required_tokens} ({num_samples} * {max_length}), got {available_tokens}."
        )

    device = _resolve_device(model)

    samples_used = 0
    accumulators: Dict[int, LayerRatioAccumulator] = {}
    with patch_qwen3_moe_blocks_for_calibration(model, accumulators):
        total = required_tokens
        idx = 0
        while samples_used < num_samples and idx < total:
            end = idx + max_length
            window = token_ids[:, idx:end].to(device)
            with torch.inference_mode():
                with maybe_bf16_autocast():
                    model(input_ids=window, use_cache=False)
            samples_used += 1
            idx = end

    per_layer_beta: Dict[int, float] = {}
    per_layer_mean: Dict[int, float] = {}
    per_layer_count: Dict[int, int] = {}
    top_k_by_layer: Dict[int, int] = {}
    for layer_idx, acc in accumulators.items():
        per_layer_beta[int(layer_idx)] = acc.median()
        per_layer_mean[int(layer_idx)] = acc.mean()
        per_layer_count[int(layer_idx)] = acc.count()
        top_k_by_layer[int(layer_idx)] = int(acc.top_k)

    return {
        "per_layer_beta": per_layer_beta,
        "per_layer_mean": per_layer_mean,
        "per_layer_token_count": per_layer_count,
        "top_k_by_layer": top_k_by_layer,
        "num_layers": len(accumulators),
        "num_samples_requested": int(num_samples),
        "num_samples_used": int(samples_used),
        "calibration_split": calibration_split,
        "max_length": int(max_length),
        "calibration_tokens_required": int(required_tokens),
        "calibration_tokens_available": int(available_tokens),
        "calibration_rows_available": int(used_rows),
        "calibration_text_policy": "wikitext_train_contiguous_first_n_tokens",
        "ratio_statistic": "median_p1_over_p0_softmax_all_experts",
    }


def score_cache_path(base_dir: Path, model_path: str, num_samples: int, max_length: int, split: str) -> Path:
    fingerprint = hashlib.sha1(
        f"{Path(model_path).as_posix()}|{num_samples}|{max_length}|{split}".encode("utf-8")
    ).hexdigest()[:12]
    model_tag = Path(model_path).name or "model"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"diep_score_{model_tag}_n{num_samples}_L{max_length}_{split}_{fingerprint}.pkl"


def load_or_compute_score(
    model,
    tokenizer,
    score_path: Path,
    num_samples: int,
    max_length: int,
    calibration_split: str,
    force_recalibrate: bool,
) -> Dict[str, object]:
    if score_path.exists() and not force_recalibrate:
        with open(score_path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and "per_layer_beta" in payload:
            return payload
    payload = calibrate_layer_scores(
        model=model,
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        calibration_split=calibration_split,
    )
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with open(score_path, "wb") as handle:
        pickle.dump(payload, handle)
    return payload


# ---------------------------------------------------------------------------
# Dynamic skipping at inference
# ---------------------------------------------------------------------------


def build_diep_keep_mask(
    routing_weights: torch.Tensor,
    full_probs: torch.Tensor,
    beta_layer: float,
    tau: float,
    eps: float = EPSILON,
) -> torch.Tensor:
    """DiEP dynamic skipping decision on Qwen3-MoE top-k routing.

    For each token, compare ``p[rank1] / p[rank0]`` over the *full* expert
    probabilities. When the ratio is below ``tau * beta_layer`` we keep only
    the top-1 expert (drop ranks 1..k-1). Otherwise we keep all top-k slots.

    Args:
        routing_weights: ``[tokens, top_k]`` weights after (optional) normalisation.
        full_probs: ``[tokens, num_experts]`` pre-topk softmax probabilities.
        beta_layer: Per-layer calibrated threshold from DiEP.
        tau: Global scale applied during sweep.

    Returns:
        Boolean keep mask of shape ``[tokens, top_k]``.
    """
    sorted_probs, _ = torch.sort(full_probs, dim=-1, descending=True)
    denom = sorted_probs[:, 0].clamp_min(eps)
    ratio = sorted_probs[:, 1] / denom
    threshold = float(tau) * float(beta_layer)
    # Tokens whose rank1/rank0 ratio is *below* threshold -> collapse to top-1.
    collapse_to_top1 = ratio < threshold
    tokens, top_k = routing_weights.shape
    # Default keep-all mask; flip non-top-1 columns off for collapsed tokens.
    keep_mask = routing_weights.new_ones((tokens, top_k), dtype=torch.bool)
    if top_k > 1 and bool(collapse_to_top1.any().item()):
        rank_positions = routing_weights.argmax(dim=-1, keepdim=True)
        idx = torch.arange(top_k, device=routing_weights.device).expand(tokens, top_k)
        keep_when_collapsed = idx == rank_positions
        keep_mask = torch.where(
            collapse_to_top1.unsqueeze(-1),
            keep_when_collapsed,
            keep_mask,
        )
    return keep_mask


def moe_forward_with_diep_skip(
    hidden_states: torch.Tensor,
    router,
    experts,
    beta_layer: float,
    tau: float,
    top_k: int,
    norm_topk_prob: bool,
    runtime_stats: Optional[RuntimeStats] = None,
    layer_idx: Optional[int] = None,
    moe_backend: str = "triton",
    shared_expert=None,
    shared_expert_gate=None,
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)

    router_logits = router(flat)
    if isinstance(router_logits, tuple):
        router_logits = router_logits[0]
    full_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(full_probs, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(EPSILON)
    routing_weights = routing_weights.to(flat.dtype)

    keep_mask = build_diep_keep_mask(
        routing_weights=routing_weights,
        full_probs=full_probs,
        beta_layer=beta_layer,
        tau=tau,
    )
    gate_kept = renorm_gate_after_pruning(routing_weights, keep_mask, eps=EPSILON)

    final, _, _ = compute_moe_weighted_hidden_states(
        flat,
        experts,
        selected_experts,
        gate_kept,
        keep_mask=keep_mask,
        moe_backend=moe_backend,
    )
    shared_output = compute_optional_shared_expert_output(
        flat,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
    )
    if shared_output is not None:
        final = final + shared_output
    final = final.view(batch_size, sequence_length, hidden_dim)

    if runtime_stats is not None and layer_idx is not None:
        runtime_stats.update(layer_idx, keep_mask)

    return final


def moe_experts_forward_with_diep_skip(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    full_probs: torch.Tensor,
    beta_layer: float,
    tau: float,
    runtime_stats: Optional[RuntimeStats] = None,
    layer_idx: Optional[int] = None,
    moe_backend: str = "triton",
):
    keep_mask = build_diep_keep_mask(
        routing_weights=routing_weights,
        full_probs=full_probs,
        beta_layer=beta_layer,
        tau=tau,
    )
    gate_kept = renorm_gate_after_pruning(routing_weights, keep_mask, eps=EPSILON)
    final, _, _ = compute_moe_weighted_hidden_states(
        hidden_states,
        experts,
        selected_experts,
        gate_kept,
        keep_mask=keep_mask,
        moe_backend=moe_backend,
    )

    if runtime_stats is not None and layer_idx is not None:
        runtime_stats.update(layer_idx, keep_mask)

    return final


@contextmanager
def patch_qwen3_moe_blocks_diep(
    model,
    per_layer_beta: Mapping[int, float],
    tau: float,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    originals: List[tuple[object, object]] = []
    for binding in iter_moe_layer_bindings(model):
        if binding.router is None:
            continue
        layer_idx = binding.layer_idx
        beta_layer = float(per_layer_beta.get(layer_idx, per_layer_beta.get(str(layer_idx), 0.0)))
        patch_target = binding.patch_target
        original_forward = patch_target.forward

        if binding.kind == "mlp":
            top_k = binding.top_k
            norm_topk_prob = binding.norm_topk_prob

            def _forward(
                self,
                hidden_states,
                _layer_idx=layer_idx,
                _beta=beta_layer,
                _top_k=top_k,
                _norm_topk_prob=norm_topk_prob,
            ):
                output = moe_forward_with_diep_skip(
                    hidden_states=hidden_states,
                    router=self.gate,
                    experts=self.experts,
                    beta_layer=_beta,
                    tau=tau,
                    top_k=_top_k,
                    norm_topk_prob=_norm_topk_prob,
                    runtime_stats=runtime_stats,
                    layer_idx=_layer_idx,
                    moe_backend=moe_backend,
                    shared_expert=getattr(self, "shared_expert", None),
                    shared_expert_gate=getattr(self, "shared_expert_gate", None),
                )
                return output
        else:
            router = binding.router

            def _forward(self, hidden_states, top_k_index, top_k_weights, _layer_idx=layer_idx, _beta=beta_layer, _router=router):
                router_logits = _router(hidden_states)
                if isinstance(router_logits, tuple):
                    router_logits = router_logits[0]
                full_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
                return moe_experts_forward_with_diep_skip(
                    hidden_states=hidden_states,
                    experts=self,
                    selected_experts=top_k_index,
                    routing_weights=top_k_weights,
                    full_probs=full_probs,
                    beta_layer=_beta,
                    tau=tau,
                    runtime_stats=runtime_stats,
                    layer_idx=_layer_idx,
                    moe_backend=moe_backend,
                )

        originals.append((patch_target, original_forward))
        patch_target.forward = MethodType(_forward, patch_target)

    try:
        yield
    finally:
        for patch_target, original_forward in originals:
            patch_target.forward = original_forward


# ---------------------------------------------------------------------------
# PPL evaluation
# ---------------------------------------------------------------------------


def evaluate_wikitext_ppl(
    model,
    tokenizer,
    *,
    split: str,
    text_column: str,
    min_text_length: int,
    n_ctx: int,
) -> Dict[str, float]:
    text, rows_used = load_wikitext_eval_text(
        split=split,
        text_column=text_column,
        min_text_length=min_text_length,
    )
    tokenizer.model_max_length = 2**31 - 1
    device = _resolve_device(model)
    tokens = tokenizer(text, truncation=False, return_tensors="pt").input_ids.to(device)

    nll_sum = 0.0
    token_count = 0
    window_count = 0
    seq_len = tokens.size(1)

    for begin in range(0, seq_len, n_ctx):
        end = min(begin + n_ctx, seq_len)
        trg_len = end - begin
        input_ids = tokens[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.inference_mode():
            with maybe_bf16_autocast():
                outputs = model(input_ids=input_ids, labels=target_ids, use_cache=False)
        neg_log_likelihood = outputs.loss.detach().float() * trg_len
        nll_sum += float(neg_log_likelihood.item())
        token_count += int(trg_len)
        window_count += 1
        if end == seq_len:
            break

    ppl = math.exp(nll_sum / max(token_count, 1))
    return {
        "ppl": float(ppl),
        "rows_used": float(rows_used),
        "windows": float(window_count),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    header = [
        "| tau | ppl | avg_dynamic_pruning_ratio | rows_used | windows |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines = list(header)
    for row in rows:
        lines.append(
            "| {tau:.4f} | {ppl:.4f} | {avg_dynamic_pruning_ratio:.4f} | {rows_used:.0f} | {windows:.0f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_run_metadata(args: argparse.Namespace, score_path: Path) -> Dict[str, object]:
    return {
        "method": "DiEP",
        "model_family": str(args.model_family),
        "model_path": str(args.model_path),
        "score_path": str(score_path),
        "split": str(args.split),
        "text_column": str(args.text_column),
        "min_text_length": int(args.min_text_length),
        "n_ctx": int(args.n_ctx),
        "calibration_num_samples": int(args.calibration_num_samples),
        "calibration_max_length": int(args.calibration_max_length),
        "calibration_split": str(args.calibration_split),
    }


def parse_tau_grid(raw: Sequence[str]) -> List[float]:
    out: List[float] = []
    for item in raw:
        for piece in str(item).replace(",", " ").split():
            if piece:
                out.append(float(piece))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DiEP dynamic expert skipping ablation on Qwen3-MoE with calibrated per-layer scores."
    )
    add_model_selection_args(parser, default_family="qwen3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tau-grid",
        nargs="+",
        default=["0.0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75", "2.0"],
        help="Grid of tau values (tau=0 keeps all top-k; tau scales the per-layer beta threshold).",
    )
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--min-text-length", type=int, default=512)
    parser.add_argument("--calibration-num-samples", type=int, default=128)
    parser.add_argument("--calibration-max-length", type=int, default=2048)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument(
        "--score-path",
        default=None,
        help="Optional explicit path to the per-layer score pickle (reused across runs).",
    )
    parser.add_argument(
        "--score-cache-dir",
        default=None,
        help="Directory to store auto-named score pickles when --score-path is not supplied.",
    )
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Fail fast when the cached score file is missing (reuse only).")
    parser.add_argument("--force-recalibrate", action="store_true",
                        help="Recompute the per-layer score even when cached.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Run only calibration (useful to prime the score cache).")
    return finalize_model_selection(parser.parse_args(), default_family="qwen3")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_cache_dir = Path(args.score_cache_dir) if args.score_cache_dir else output_dir / "score_cache"

    if args.score_path:
        score_path = Path(args.score_path)
    else:
        score_path = score_cache_path(
            score_cache_dir,
            args.model_path,
            args.calibration_num_samples,
            args.calibration_max_length,
            args.calibration_split,
        )

    tau_grid = parse_tau_grid(args.tau_grid)
    print(
        f"[DiEP] start output_dir={output_dir} tau_grid={tau_grid} score_path={score_path}",
        flush=True,
    )

    if args.skip_calibration and not score_path.exists():
        raise FileNotFoundError(
            f"--skip-calibration requires an existing score file at {score_path}"
        )

    print(f"[DiEP] loading model from {args.model_path}", flush=True)
    model, tokenizer = load_qwen3_moe(args.model_path, model_family=args.model_family)
    print("[DiEP] model load complete", flush=True)

    score_payload = load_or_compute_score(
        model=model,
        tokenizer=tokenizer,
        score_path=score_path,
        num_samples=args.calibration_num_samples,
        max_length=args.calibration_max_length,
        calibration_split=args.calibration_split,
        force_recalibrate=args.force_recalibrate,
    )
    run_metadata = build_run_metadata(args, score_path)

    write_json(output_dir / "score_summary.json", {
        **run_metadata,
        "score_path": str(score_path),
        "num_layers": score_payload.get("num_layers"),
        "num_samples_used": score_payload.get("num_samples_used"),
        "num_samples_requested": score_payload.get("num_samples_requested"),
        "calibration_split": score_payload.get("calibration_split"),
        "max_length": score_payload.get("max_length"),
        "calibration_tokens_required": score_payload.get("calibration_tokens_required"),
        "calibration_tokens_available": score_payload.get("calibration_tokens_available"),
        "calibration_rows_available": score_payload.get("calibration_rows_available"),
        "calibration_text_policy": score_payload.get("calibration_text_policy"),
        "per_layer_beta_median": {
            str(k): float(v) for k, v in score_payload["per_layer_beta"].items()
        },
        "per_layer_beta_mean": {
            str(k): float(v) for k, v in score_payload.get("per_layer_mean", {}).items()
        },
        "per_layer_token_count": {
            str(k): int(v) for k, v in score_payload.get("per_layer_token_count", {}).items()
        },
    })
    print(
        f"[DiEP] score ready: layers={score_payload.get('num_layers')} "
        f"samples_used={score_payload.get('num_samples_used')}",
        flush=True,
    )

    per_layer_beta = score_payload["per_layer_beta"]
    if args.skip_eval:
        print("[DiEP] --skip-eval set, exiting after calibration", flush=True)
        return

    rows: List[Dict[str, float]] = []
    partial_path = output_dir / "wikitext_ppl.partial.json"
    partial_md = output_dir / "wikitext_ppl.partial.md"
    for tau in tau_grid:
        runtime_stats = RuntimeStats()
        print(f"[DiEP] evaluating tau={tau:.4f}", flush=True)
        with patch_qwen3_moe_blocks_diep(
            model=model,
            per_layer_beta=per_layer_beta,
            tau=tau,
            runtime_stats=runtime_stats,
        ):
            metrics = evaluate_wikitext_ppl(
                model,
                tokenizer,
                split=args.split,
                text_column=args.text_column,
                min_text_length=args.min_text_length,
                n_ctx=args.n_ctx,
            )
        row = {
            **run_metadata,
            "tau": float(tau),
            "ppl": float(metrics["ppl"]),
            "avg_dynamic_pruning_ratio": float(runtime_stats.mean_pruning_ratio()),
            "rows_used": float(metrics["rows_used"]),
            "windows": float(metrics["windows"]),
        }
        rows.append(row)
        per_tau_dir = output_dir / "ppl_by_tau" / f"tau_{float(tau):.4f}"
        write_json(per_tau_dir / "wikitext_ppl.json", row)
        write_json(partial_path, rows)
        write_markdown(partial_md, rows)
        print(
            "[DiEP] tau={:.4f} ppl={:.4f} avg_dynamic_pruning_ratio={:.4f}".format(
                float(tau), row["ppl"], row["avg_dynamic_pruning_ratio"]
            ),
            flush=True,
        )

    write_json(output_dir / "wikitext_ppl.json", rows)
    write_markdown(output_dir / "wikitext_ppl.md", rows)
    print(f"[DiEP] done output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
