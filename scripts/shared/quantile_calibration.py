"""Build quantile thresholds from the same full unpruned PPL forward path.

For ``--search-mode quantile`` we no longer run a separate sampled calibration
pass. Instead, we replay the exact evaluation corpus/windows (equivalent to a
``tau=0`` run), collect candidate scores online, and derive the target
thresholds from that global score pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from moe_prune.code.src.aimer_selector import build_aimer_keep_table_for_model
from moe_prune.code.scripts.shared.ppl_eval import FullWikiTextPerplexity
from moe_prune.code.src.amp_proxy import (
    build_amp_table_for_model,
    build_router_proto_amp_table_for_model,
)
from moe_prune.code.src.model_adapter import clear_hf_proxy_env, load_qwen3_moe, maybe_bf16_autocast
from moe_prune.code.src.expert_similarity import build_model_similarity_table
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection
from moe_prune.code.src.evalscope_search import archive_incomplete_work_dir
from moe_prune.code.src.quantile_collector import patched_model_for_quantile_collection
from moe_prune.code.src.quantile_search import (
    QUANTILE_COMPATIBLE_METHODS,
    build_threshold_table_from_global_candidates,
    candidate_collection_cache_matches,
    load_candidate_collection_chunks,
    write_candidate_collection_meta,
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Build target-rate→tau threshold table from the full unpruned evaluation forward."
    )
    add_model_selection_args(parser)
    parser.add_argument("--method", choices=list(QUANTILE_COMPATIBLE_METHODS), required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-pruning-ratios", type=float, nargs="+", required=True)
    parser.add_argument("--candidate-cache-dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--min-text-length", type=int, default=512)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-batch", type=int, default=2048)
    parser.add_argument(
        "--router-weight-centering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Center router weights within each layer for the RCR branch (default: false).",
    )
    return finalize_model_selection(parser.parse_args(list(argv) if argv is not None else None))


def _candidate_cache_signature(args) -> Dict[str, object]:
    return {
        "source": "ppl_full_unpruned_eval_forward",
        "method": args.method,
        "model_family": args.model_family,
        "model_path": args.model_path,
        "split": args.split,
        "text_column": args.text_column,
        "min_text_length": int(args.min_text_length),
        "n_ctx": int(args.n_ctx),
        "n_batch": int(args.n_batch),
        "router_weight_centering": bool(args.router_weight_centering),
    }


def _flush_chunk(
    chunks_dir: Path,
    chunk_index: int,
    candidate_scores_by_layer: Dict[int, List[torch.Tensor]],
    total_slots_by_layer: Dict[int, int],
) -> bool:
    candidate_parts = []
    for parts in candidate_scores_by_layer.values():
        if parts:
            candidate_parts.append(torch.cat(parts, dim=0))
    total_slots = int(sum(total_slots_by_layer.values()))
    if not candidate_parts:
        for layer_idx in list(candidate_scores_by_layer.keys()):
            candidate_scores_by_layer[layer_idx] = []
        for layer_idx in list(total_slots_by_layer.keys()):
            total_slots_by_layer[layer_idx] = 0
        return False
    payload = {
        "candidate_scores": torch.cat(candidate_parts, dim=0),
        "total_slots": total_slots,
    }
    chunks_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, chunks_dir / f"chunk_{chunk_index:06d}.pt")
    for layer_idx in list(candidate_scores_by_layer.keys()):
        candidate_scores_by_layer[layer_idx] = []
    for layer_idx in list(total_slots_by_layer.keys()):
        total_slots_by_layer[layer_idx] = 0
    return True


def collect_global_candidates_from_full_forward(
    model,
    tokenizer,
    *,
    method: str,
    split: str,
    text_column: str,
    min_text_length: int,
    n_ctx: int,
    n_batch: int,
    amp_tables: Dict[int, torch.Tensor] | None = None,
    proto_amp_tables: Dict[int, torch.Tensor] | None = None,
    sim_tables: Dict[int, torch.Tensor] | None = None,
    chunks_dir: Path | None = None,
) -> tuple[torch.Tensor, int]:
    del n_batch
    evaluator = FullWikiTextPerplexity(
        model=model,
        tokenizer=tokenizer,
        split=split,
        text_column=text_column,
        min_text_length=min_text_length,
    )
    runtime_device = evaluator._resolve_device()
    tokens = evaluator._tokenize_for_device(runtime_device)
    max_length = n_ctx
    stride = n_ctx
    seq_len = tokens.size(1)

    candidate_scores_by_layer: Dict[int, List[torch.Tensor]] = {}
    total_slots_by_layer: Dict[int, int] = {}
    chunk_index = 0
    with patched_model_for_quantile_collection(
        model,
        method=method,
        candidate_scores_by_layer=candidate_scores_by_layer,
        total_slots_by_layer=total_slots_by_layer,
        amp_tables=amp_tables,
        proto_amp_tables=proto_amp_tables,
        sim_tables=sim_tables,
    ):
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            input_ids = tokens[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :- (end_loc - begin_loc)] = -100
            with torch.inference_mode():
                with maybe_bf16_autocast():
                    model(input_ids, labels=target_ids, use_cache=False)
            if chunks_dir is not None and _flush_chunk(
                chunks_dir,
                chunk_index,
                candidate_scores_by_layer,
                total_slots_by_layer,
            ):
                chunk_index += 1
            if end_loc == seq_len:
                break

    if chunks_dir is not None:
        return load_candidate_collection_chunks(chunks_dir)[:2]

    all_candidates = [
        torch.cat(parts, dim=0)
        for parts in candidate_scores_by_layer.values()
        if parts
    ]
    if not all_candidates:
        raise ValueError("No candidate scores collected from full forward.")
    return torch.cat(all_candidates, dim=0), sum(total_slots_by_layer.values())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    clear_hf_proxy_env()
    model, tokenizer = load_qwen3_moe(args.model_path, model_family=args.model_family)
    model.eval()

    amp_tables = None
    proto_amp_tables = None
    sim_tables = None
    if args.method in {"gsp", "ace", "naee"}:
        amp_tables = {int(k): v for k, v in build_amp_table_for_model(model).items()}
    if args.method == "aimer":
        amp_tables = {int(k): v for k, v in build_aimer_keep_table_for_model(model).items()}
    if args.method in {"rcr", "ace"}:
        proto_amp_tables = {
            int(k): v
            for k, v in build_router_proto_amp_table_for_model(
                model,
                center_router_weights=args.router_weight_centering,
            ).items()
        }
    if args.method == "sere":
        sim_tables = {int(k): v for k, v in build_model_similarity_table(model, mode="fast").items()}

    cache_signature = _candidate_cache_signature(args)
    if args.candidate_cache_dir is not None and candidate_collection_cache_matches(args.candidate_cache_dir, cache_signature):
        global_candidates, total_slots, chunk_count = load_candidate_collection_chunks(args.candidate_cache_dir)
    else:
        if args.candidate_cache_dir is not None:
            archived = archive_incomplete_work_dir(args.candidate_cache_dir)
            if archived is not None:
                print(f"[quantile_calibration] archived incomplete cache_dir {args.candidate_cache_dir} -> {archived}")
        global_candidates, total_slots = collect_global_candidates_from_full_forward(
            model,
            tokenizer,
            method=args.method,
            split=args.split,
            text_column=args.text_column,
            min_text_length=int(args.min_text_length),
            n_ctx=int(args.n_ctx),
            n_batch=int(args.n_batch),
            amp_tables=amp_tables,
            proto_amp_tables=proto_amp_tables,
            sim_tables=sim_tables,
            chunks_dir=args.candidate_cache_dir,
        )
        if args.candidate_cache_dir is not None:
            _, _, chunk_count = load_candidate_collection_chunks(args.candidate_cache_dir)
            write_candidate_collection_meta(
                args.candidate_cache_dir,
                {
                    **cache_signature,
                    "collection_complete": True,
                    "chunk_count": int(chunk_count),
                    "total_slots": int(total_slots),
                    "total_candidates": int(global_candidates.numel()),
                },
            )
        else:
            chunk_count = 0

    threshold_table, stats_table = build_threshold_table_from_global_candidates(
        global_candidates=global_candidates,
        total_slots=total_slots,
        target_rates=args.target_pruning_ratios,
        target_is_global_slot_rate=True,
    )

    payload = {
        "method": args.method,
        "model_path": args.model_path,
        "model_family": args.model_family,
        "target_pruning_ratios": [float(x) for x in args.target_pruning_ratios],
        "split": args.split,
        "text_column": args.text_column,
        "min_text_length": int(args.min_text_length),
        "n_ctx": int(args.n_ctx),
        "n_batch": int(args.n_batch),
        "target_is_global_slot_rate": True,
        "min_keep": 1,
        "threshold_table": {f"{float(rate):.6f}": float(tau) for rate, tau in threshold_table.items()},
        "stats_table": {f"{float(rate):.6f}": stats for rate, stats in stats_table.items()},
        "global_candidate_count": int(global_candidates.numel()),
        "candidate_cache_dir": str(args.candidate_cache_dir) if args.candidate_cache_dir is not None else None,
        "chunk_count": int(chunk_count),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[quantile_calibration] wrote {args.output_path}")
    print(json.dumps({"threshold_table": payload["threshold_table"], "stats_table": payload["stats_table"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
