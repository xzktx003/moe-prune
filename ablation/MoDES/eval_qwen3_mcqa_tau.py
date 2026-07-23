import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from datasets import Dataset, load_dataset
from loguru import logger

import models.qwen3 as qwen3_model
from models.qwen3 import load_model as load_qwen_model
from utils import ensure_single_gpu_evaluation

OPTION_RE = {"A", "B", "C", "D", "E"}


def parse_taus(raw: str) -> List[float]:
    values: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def maybe_bf16_autocast():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", enabled=False)


def build_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_mcqa_dataset(dataset_name: str) -> Dataset:
    if dataset_name == "mathqa":
        return load_dataset("math_qa", split="test")
    if dataset_name == "openbookqa":
        return load_dataset("allenai/openbookqa", "main", split="validation")
    if dataset_name == "arc_challenge":
        return load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    raise KeyError(f"Unsupported dataset: {dataset_name}")


def format_example(dataset_name: str, example: Dict) -> Tuple[str, str, List[str]]:
    if dataset_name == "mathqa":
        labels = ["A", "B", "C", "D", "E"]
        prompt = (
            "You are solving a multiple choice question.\n"
            "Return only the capital letter of the correct answer.\n\n"
            f"Question: {example['Problem']}\n"
            f"Options: {example['options']}\n"
            "Answer:"
        )
        return prompt, example["correct"].upper(), labels
    if dataset_name == "openbookqa":
        labels = list(example["choices"]["label"])
        choices = "\n".join(
            f"{label}. {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"])
        )
        prompt = (
            "You are solving a multiple choice question.\n"
            "Return only the capital letter of the correct answer.\n\n"
            f"Question: {example['question_stem']}\n"
            f"{choices}\n"
            "Answer:"
        )
        return prompt, example["answerKey"].upper(), labels
    if dataset_name == "arc_challenge":
        labels = list(example["choices"]["label"])
        choices = "\n".join(
            f"{label}. {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"])
        )
        prompt = (
            "You are solving a multiple choice question.\n"
            "Return only the capital letter of the correct answer.\n\n"
            f"Question: {example['question']}\n"
            f"{choices}\n"
            "Answer:"
        )
        return prompt, example["answerKey"].upper(), labels
    raise KeyError(f"Unsupported dataset: {dataset_name}")


def _resolve_model_device(model):
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "hf_device_map"):
        return list(model.hf_device_map.values())[0]
    return next(model.parameters()).device


def predict_choices_batched(model, tokenizer, prompts: List[str], labels: Iterable[str], tau_text: float, layer_importance_path: str | None) -> List[str | None]:
    if not prompts:
        return []
    rendered = [build_chat_prompt(tokenizer, prompt) for prompt in prompts]
    device = _resolve_model_device(model)
    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(device)

    candidate_ids = []
    normalized_labels = []
    for label in labels:
        token_ids = tokenizer.encode(f" {label.upper()}", add_special_tokens=False)
        if len(token_ids) != 1:
            candidate_ids = []
            break
        candidate_ids.append(token_ids[0])
        normalized_labels.append(label.upper())

    with torch.no_grad():
        with maybe_bf16_autocast():
            outputs = model(
                **inputs,
                use_cache=False,
                return_dict=True,
                enable_tau_skip=True,
                tau={"text": float(tau_text), "visual": 0.0},
                enable_load_layer_importance=bool(layer_importance_path),
                layer_importance_path=layer_importance_path,
            )

    if candidate_ids:
        logits = outputs.logits
        seq_lens = inputs["attention_mask"].sum(dim=-1) - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        next_token_logits = logits[batch_indices, seq_lens]
        choice_scores = next_token_logits[:, candidate_ids]
        pred_indices = choice_scores.argmax(dim=-1).tolist()
        return [normalized_labels[idx] for idx in pred_indices]

    generated = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        enable_tau_skip=True,
        tau={"text": float(tau_text), "visual": 0.0},
        enable_load_layer_importance=bool(layer_importance_path),
        layer_importance_path=layer_importance_path,
    )
    input_lengths = inputs["attention_mask"].sum(dim=-1).tolist()
    predictions: List[str | None] = []
    for row_idx, input_len in enumerate(input_lengths):
        output = tokenizer.decode(generated[row_idx][input_len:], skip_special_tokens=True)
        prediction = None
        for token in output.upper().split():
            cleaned = token.strip(".,:;()[]{}")
            if cleaned in OPTION_RE:
                prediction = cleaned
                break
        predictions.append(prediction)
    return predictions


def evaluate_dataset(model, tokenizer, dataset_name: str, tau_text: float, layer_importance_path: str | None, batch_size: int) -> Dict[str, float]:
    dataset = load_mcqa_dataset(dataset_name)
    qwen3_model.SKIP_EXP_COUNT = 0
    qwen3_model.TOTAL_EXP_COUNT = 0

    total = 0
    correct = 0
    skipped = 0
    batch_prompts: List[str] = []
    batch_gold: List[str] = []
    batch_labels: List[str] | None = None

    def _flush():
        nonlocal total, correct, skipped, batch_prompts, batch_gold, batch_labels
        if not batch_prompts or batch_labels is None:
            return
        preds = predict_choices_batched(model, tokenizer, batch_prompts, batch_labels, tau_text, layer_importance_path)
        for pred, gold in zip(preds, batch_gold):
            total += 1
            if pred is None:
                skipped += 1
                continue
            if pred == gold:
                correct += 1
        batch_prompts = []
        batch_gold = []
        batch_labels = None

    for example in dataset:
        prompt, gold, labels = format_example(dataset_name, example)
        if batch_labels is None:
            batch_labels = labels
        batch_prompts.append(prompt)
        batch_gold.append(gold)
        if len(batch_prompts) >= batch_size:
            _flush()
    _flush()
    pruning_ratio = qwen3_model.SKIP_EXP_COUNT / max(qwen3_model.TOTAL_EXP_COUNT, 1.0)
    return {
        "dataset_size": float(total),
        "correct": float(correct),
        "skipped": float(skipped),
        "accuracy": float(correct / total) if total else 0.0,
        "avg_dynamic_pruning_ratio": float(pruning_ratio),
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, rows) -> None:
    lines = [
        "| tau | mathqa | openbookqa | arc_challenge | avg_pruning_ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {tau:.6f} | {mathqa:.6f} | {openbookqa:.6f} | {arc_challenge:.6f} | {avg_pruning_ratio:.6f} |".format(**row)
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate true MoDES Qwen3 text-only tau skipping on MCQA datasets.")
    parser.add_argument("--name_or_path", type=str, required=True)
    parser.add_argument("--taus", type=str, required=True)
    parser.add_argument("--layer_importance_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="storage/search/qwen3_text_mcqa_metrics.json")
    args = parser.parse_args()

    ensure_single_gpu_evaluation()
    logger.info("Loading Qwen3 MoDES model from {}", args.name_or_path)
    model, tokenizer = load_qwen_model(args.name_or_path, device_map="auto")
    model.eval()

    output_path = Path(args.output_path)
    markdown_path = output_path.with_suffix(".md")
    rows = []
    for tau in parse_taus(args.taus):
        logger.info("Evaluating MCQA at tau={}", tau)
        per_dataset = {}
        for dataset_name in ("mathqa", "openbookqa", "arc_challenge"):
            metrics = evaluate_dataset(model, tokenizer, dataset_name, tau, args.layer_importance_path, args.batch_size)
            per_dataset[dataset_name] = metrics
        row = {
            "tau": float(tau),
            "mathqa": per_dataset["mathqa"]["accuracy"],
            "openbookqa": per_dataset["openbookqa"]["accuracy"],
            "arc_challenge": per_dataset["arc_challenge"]["accuracy"],
            "avg_pruning_ratio": float(sum(per_dataset[name]["avg_dynamic_pruning_ratio"] for name in ("mathqa", "openbookqa", "arc_challenge")) / 3.0),
            "method": "MoDES",
        }
        rows.append(row)
        write_json(output_path, rows)
        write_markdown(markdown_path, rows)
        write_json(output_path.parent / f"results_tau_{tau:.6f}.json", per_dataset)
        logger.info("tau={} row={}", tau, row)


if __name__ == "__main__":
    main()
