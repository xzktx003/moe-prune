from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset

from moe_prune.code.src.aimer_selector import build_aimer_keep_table_for_model
from moe_prune.code.src.amp_proxy import build_amp_table_for_model, build_router_proto_amp_table_for_model
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection
from moe_prune.code.src.model_adapter import (
    clear_hf_proxy_env,
    load_qwen3_moe,
    maybe_bf16_autocast,
    patched_model_for_ace,
    patched_model_for_gsp,
    patched_model_for_rcr,
    patched_model_for_aimer,
    patched_model_for_expert_sparsity,
    patched_model_for_score_only,
    patched_model_for_sere,
    patched_model_for_top_p,
    patched_model_for_xshare,
)
from moe_prune.code.src.runtime_pruner import RuntimeStats


class FullWikiTextPerplexity:
    def __init__(
        self,
        model,
        tokenizer,
        split: str = "test",
        text_column: str = "text",
        min_text_length: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.split = split
        self.text_column = text_column
        self.min_text_length = min_text_length
        self.text, self.num_rows = self._prepare_text()
        self._tokens_by_device: Dict[str, torch.Tensor] = {}

    def _resolve_device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        if hasattr(self.model, "hf_device_map"):
            for mapped in self.model.hf_device_map.values():
                if mapped not in ("cpu", "disk"):
                    return mapped
        return next(self.model.parameters()).device

    def _prepare_text(self):
        clear_hf_proxy_env()
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split=self.split)
        texts: List[str] = []
        used_rows = 0
        for sample in data:
            text = sample[self.text_column]
            if len(text) >= self.min_text_length:
                texts.append(" \n" if text == "" else text)
                used_rows += 1
        return "".join(texts), used_rows

    @staticmethod
    def _device_cache_key(runtime_device) -> str:
        return str(runtime_device)

    def _tokenize_for_device(self, runtime_device) -> torch.Tensor:
        cache_key = self._device_cache_key(runtime_device)
        tokens = self._tokens_by_device.get(cache_key)
        if tokens is not None:
            return tokens

        self.tokenizer.model_max_length = 2**31 - 1
        tokens = self.tokenizer(
            self.text,
            truncation=False,
            return_tensors="pt",
        ).input_ids.to(runtime_device)
        self._tokens_by_device[cache_key] = tokens
        return tokens

    def calculate(self, n_ctx: int = 512, n_batch: int = 512):
        runtime_device = self._resolve_device()
        tokens = self._tokenize_for_device(runtime_device)
        max_length = n_ctx
        stride = n_ctx
        seq_len = tokens.size(1)

        nll_sum = 0.0
        token_count = 0
        all_perplexity = []

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - begin_loc
            input_ids = tokens[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.inference_mode():
                with maybe_bf16_autocast():
                    outputs = self.model(
                        input_ids, labels=target_ids, use_cache=False)

            neg_log_likelihood = outputs.loss.detach().float() * trg_len
            nll_sum += neg_log_likelihood.item()
            token_count += trg_len
            all_perplexity.append(
                float(torch.exp(torch.tensor(nll_sum / token_count)).item()))

            if end_loc == seq_len:
                break

        return all_perplexity

    def calculate_corpus_ppl(self, n_ctx: int = 512, n_batch: int = 512) -> Dict[str, float]:
        del n_batch
        runtime_device = self._resolve_device()
        tokens = self._tokenize_for_device(runtime_device)
        max_length = n_ctx
        stride = n_ctx
        seq_len = tokens.size(1)

        nll_sum = 0.0
        token_count = 0
        window_count = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - begin_loc
            input_ids = tokens[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.inference_mode():
                with maybe_bf16_autocast():
                    outputs = self.model(
                        input_ids, labels=target_ids, use_cache=False)

            neg_log_likelihood = outputs.loss.detach().float() * trg_len
            nll_sum += neg_log_likelihood.item()
            token_count += trg_len
            window_count += 1

            if end_loc == seq_len:
                break

        ppl = math.exp(nll_sum / token_count) if token_count else 0.0
        return {
            "ppl": float(ppl),
            "windows": float(window_count),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pruned Qwen3-MoE perplexity on full WikiText-2.")
    add_model_selection_args(parser)
    parser.add_argument("--output-dir", type=str,
                        default="moe_prune/outputs-ppl")
    parser.add_argument("--taus", type=float, nargs="+",
                        default=[round(i * 0.1, 1) for i in range(11)])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--min-text-length", type=int, default=512)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-batch", type=int, default=2048)
    parser.add_argument(
        "--method",
        choices=[
            "ace",
            "gsp",
            "rcr",
            "score_only",
            "aimer",
            "expert_sparsity",
            "top_p",
            "sere",
            "xshare",
        ],
        default="ace",
    )
    parser.add_argument("--lambda-penalty", type=float, default=0.5)
    parser.add_argument("--gate-guard", type=float, default=0.5)
    parser.add_argument(
        "--score-only-moe-backend",
        choices=["torch", "triton"],
        default="triton",
        help="MoE expert backend used only when --method score_only.",
    )
    parser.add_argument("--similarity-mode",
                        choices=["fast", "exact"], default="fast")
    parser.add_argument(
        "--router-weight-centering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Center router weights within each layer before building the RCR branch (default: false).",
    )
    return finalize_model_selection(parser.parse_args())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2,
                    ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, rows: List[Dict[str, float]]) -> None:
    lines = [
        (
            "| tau | ppl | delta_vs_tau0 | active_expert_pruning_ratio "
            "| rows_used | method |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    tau0 = rows[0]["ppl"] if rows else 0.0
    for row in rows:
        delta = row["ppl"] - tau0
        lines.append(
            (
                f"| {row['tau']:.4f} | {row['ppl']:.4f} | "
                f"{delta:+.4f} | "
                f"{row['active_expert_pruning_ratio']:.4f} | "
                f"{int(row['rows_used'])} | {row['method']} |"
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_tau_dir(value: float) -> str:
    return f"tau_{float(value):.4f}"


def knob_name_for_ppl_method(method: str) -> str:
    return "beta" if method in {"score_only", "expert_sparsity"} else "tau"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_qwen3_moe(args.model_path, model_family=args.model_family)
    amp_table = (
        None
        if args.method in {
            "score_only",
            "rcr",
            "aimer",
            "expert_sparsity",
            "top_p",
            "sere",
            "xshare",
        }
        else build_amp_table_for_model(model)
    )
    proto_amp_table = None
    if args.method in {"rcr", "ace"}:
        if args.router_weight_centering:
            proto_amp_table = build_router_proto_amp_table_for_model(
                model,
                center_router_weights=True,
            )
        else:
            proto_amp_table = build_router_proto_amp_table_for_model(model)
    aimer_keep_table = (
        build_aimer_keep_table_for_model(model)
        if args.method == "aimer"
        else None
    )
    evaluator = FullWikiTextPerplexity(
        model=model,
        tokenizer=tokenizer,
        split=args.split,
        text_column=args.text_column,
        min_text_length=args.min_text_length,
    )

    rows: List[Dict[str, float]] = []
    for tau in args.taus:
        runtime_stats = RuntimeStats()
        if args.method == "score_only":
            patch_ctx = patched_model_for_score_only(
                model,
                gate_threshold=tau,
                runtime_stats=runtime_stats,
                moe_backend=args.score_only_moe_backend,
            )
        elif args.method == "expert_sparsity":
            patch_ctx = patched_model_for_expert_sparsity(
                model,
                beta=tau,
                runtime_stats=runtime_stats,
                moe_backend=args.score_only_moe_backend,
            )
        elif args.method == "top_p":
            patch_ctx = patched_model_for_top_p(
                model,
                tau=tau,
                runtime_stats=runtime_stats,
                moe_backend=args.score_only_moe_backend,
            )
        elif args.method == "sere":
            patch_ctx = patched_model_for_sere(
                model,
                tau=tau,
                similarity_mode=args.similarity_mode,
                runtime_stats=runtime_stats,
                moe_backend=args.score_only_moe_backend,
            )
        elif args.method == "xshare":
            patch_ctx = patched_model_for_xshare(
                model,
                tau=tau,
                runtime_stats=runtime_stats,
                moe_backend=args.score_only_moe_backend,
            )
        elif args.method == "gsp":
            patch_ctx = patched_model_for_gsp(
                model,
                amp_table=amp_table,
                tau=tau,
                runtime_stats=runtime_stats,
            )
        elif args.method == "ace":
            patch_ctx = patched_model_for_ace(
                model,
                slanc_amp_table=amp_table,
                proto_amp_table=proto_amp_table,
                tau=tau,
                runtime_stats=runtime_stats,
            )
        elif args.method == "aimer":
            patch_ctx = patched_model_for_aimer(
                model,
                keep_table=aimer_keep_table,
                tau=tau,
                runtime_stats=runtime_stats,
            )
        elif args.method == "rcr":
            patch_ctx = patched_model_for_rcr(
                model,
                proto_amp_table=proto_amp_table,
                tau=tau,
                runtime_stats=runtime_stats,
            )
        else:
            raise ValueError(f"Unsupported paper method: {args.method}")
        with patch_ctx:
            metrics = evaluator.calculate_corpus_ppl(
                n_ctx=args.n_ctx, n_batch=args.n_batch)
        rows.append(
            {
                "tau": float(tau),
                "ppl": float(metrics["ppl"]),
                "active_expert_pruning_ratio": float(
                    runtime_stats.mean_pruning_ratio()
                ),
                "rows_used": float(evaluator.num_rows),
                "windows": float(metrics["windows"]),
                "method": args.method,
                "gate_threshold": (
                    float(tau)
                    if args.method == "score_only"
                    else None
                ),
                "beta": (
                    float(tau)
                    if args.method == "expert_sparsity"
                    else None
                ),
            }
        )
        current_row = rows[-1]
        if args.method in {
            "ace",
            "gsp",
            "rcr",
            "aimer",
            "expert_sparsity",
            "top_p",
            "sere",
            "xshare",
        }:
            knob_name = knob_name_for_ppl_method(args.method)
            per_tau_dir = output_dir / "ppl_by_tau" / f"{knob_name}_{float(tau):.4f}"
            write_json(per_tau_dir / "wikitext_ppl.json", current_row)
            write_markdown(per_tau_dir / "wikitext_ppl.md", [current_row])
        write_json(output_dir / "wikitext_ppl.partial.json", rows)
        write_markdown(output_dir / "wikitext_ppl.partial.md", rows)

    write_json(output_dir / "wikitext_ppl.json", rows)
    write_markdown(output_dir / "wikitext_ppl.md", rows)


if __name__ == "__main__":
    main()
