import argparse
import json
import math
from pathlib import Path
from typing import Any, List, Mapping, cast

import torch
from datasets import load_dataset
from loguru import logger
from transformers import AutoConfig

import models.gemma4 as gemma4_model
import models.qwen3 as qwen3_model
from models.gemma4 import load_model as load_gemma_model
from models.qwen3 import load_model as load_qwen_model
from utils import ensure_single_gpu_evaluation


def parse_taus(raw: str) -> List[float]:
    values: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def load_wikitext_text(
    split: str = "test",
    text_column: str = "text",
    min_text_length: int = 512,
) -> tuple[str, int]:
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts: List[str] = []
    used_rows = 0
    for row in data:
        row = cast(Mapping[str, Any], row)
        text = row[text_column]
        if len(text) < min_text_length:
            continue
        texts.append(" \n" if text == "" else text)
        used_rows += 1
    return "".join(texts), used_rows


def tokenize_corpus(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    tokenizer.model_max_length = 2**31 - 1
    return tokenizer(
        text,
        truncation=False,
        return_tensors="pt",
    ).input_ids.to(device)


def maybe_bf16_autocast():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", enabled=False)


def resolve_model_backend(name_or_path: str):
    config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if model_type in {"qwen3_moe", "qwen3_5_moe"}:
        return load_qwen_model, qwen3_model
    if model_type == "gemma4":
        return load_gemma_model, gemma4_model
    raise ValueError(f"Unsupported model_type={model_type} for {name_or_path}")


@torch.no_grad()
def evaluate_tau(
    model,
    counter_module,
    tokens: torch.Tensor,
    tau_text: float,
    layer_importance_path: str | None,
    n_ctx: int,
    rows_used: int,
    max_windows: int,
):
    tau = {"text": float(tau_text), "visual": 0.0}
    counter_module.SKIP_EXP_COUNT = 0
    counter_module.TOTAL_EXP_COUNT = 0

    total_nll = 0.0
    total_tokens = 0
    num_windows = 0
    seq_len = tokens.size(1)

    for begin_loc in range(0, seq_len, n_ctx):
        if max_windows > 0 and num_windows >= max_windows:
            break
        end_loc = min(begin_loc + n_ctx, seq_len)
        trg_len = end_loc - begin_loc
        input_ids = tokens[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.inference_mode():
            with maybe_bf16_autocast():
                outputs = model(
                    input_ids=input_ids,
                    labels=target_ids,
                    use_cache=False,
                    return_dict=True,
                    enable_tau_skip=True,
                    tau=tau,
                    enable_load_layer_importance=bool(layer_importance_path),
                    layer_importance_path=layer_importance_path,
                )

        neg_log_likelihood = outputs.loss.detach().float() * trg_len
        total_nll += float(neg_log_likelihood.item())
        total_tokens += int(trg_len)
        num_windows += 1

        if end_loc == seq_len:
            break

    ppl = math.exp(total_nll / max(total_tokens, 1.0))
    pruning_ratio = counter_module.SKIP_EXP_COUNT / max(counter_module.TOTAL_EXP_COUNT, 1.0)
    return {
        "tau": float(tau_text),
        "tau_text": float(tau_text),
        "pruning_ratio": float(pruning_ratio),
        "avg_dynamic_pruning_ratio": float(pruning_ratio),
        "ppl": float(ppl),
        "rows_used": float(rows_used),
        "num_windows": float(num_windows),
        "windows": float(num_windows),
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, rows) -> None:
    lines = [
        "| tau | pruning_ratio | ppl | rows_used | num_windows |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {tau:.6f} | {pruning_ratio:.6f} | {ppl:.6f} | {rows_used:.0f} | {num_windows:.0f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate official MoDES text-only tau skipping on WikiText PPL."
    )
    parser.add_argument("--name_or_path", type=str, required=True)
    parser.add_argument("--wikitext_split", type=str, default="test")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--min_text_length", type=int, default=512)
    parser.add_argument("--taus", type=str, default="0.0,0.0005,0.001,0.002,0.005")
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--max_windows", type=int, default=-1)
    parser.add_argument("--layer_importance_path", type=str, default=None)
    parser.add_argument(
        "--output_path",
        type=str,
        default="storage/search/modes_text_tau_metrics_wikitext.json",
    )
    args = parser.parse_args()

    ensure_single_gpu_evaluation()
    logger.info("Loading model from {}", args.name_or_path)
    load_model, counter_module = resolve_model_backend(args.name_or_path)
    model, tokenizer = load_model(args.name_or_path, device_map="auto")
    model.eval()

    full_text, rows_used = load_wikitext_text(
        split=args.wikitext_split,
        text_column=args.text_column,
        min_text_length=args.min_text_length,
    )
    logger.info("Loaded {} usable WikiText rows", rows_used)
    device = getattr(model, "device", next(model.parameters()).device)
    tokens = tokenize_corpus(tokenizer, full_text, device)
    logger.info("Tokenized WikiText corpus to {} tokens", tokens.size(1))

    results = []
    output_path = Path(args.output_path)
    markdown_path = output_path.with_suffix(".md")
    for tau in parse_taus(args.taus):
        logger.info("Evaluating tau={}", tau)
        row = evaluate_tau(
            model=model,
            counter_module=counter_module,
            tokens=tokens,
            tau_text=tau,
            layer_importance_path=args.layer_importance_path,
            n_ctx=args.n_ctx,
            rows_used=rows_used,
            max_windows=args.max_windows,
        )
        results.append(row)
        write_json(output_path, results)
        write_markdown(markdown_path, results)
        logger.info("tau={} result={}", tau, row)

    logger.info("Saved metrics to {}", output_path)


if __name__ == "__main__":
    main()
