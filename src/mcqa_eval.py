from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from datasets import Dataset, load_dataset

from .model_adapter import clear_hf_proxy_env, maybe_bf16_autocast


OPTION_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


@dataclass
class DatasetSpec:
    name: str
    path: str
    config: str | None
    split: str


DATASET_SPECS = {
    "mathqa": DatasetSpec("mathqa", "math_qa", None, "test"),
    "openbookqa": DatasetSpec("openbookqa", "allenai/openbookqa", "main", "validation"),
    "arc_challenge": DatasetSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "validation"),
}


def load_mcqa_dataset(dataset_name: str) -> Dataset:
    clear_hf_proxy_env()
    spec = DATASET_SPECS[dataset_name]
    return load_dataset(spec.path, spec.config, split=spec.split)


def format_example(dataset_name: str, example: Dict) -> Tuple[str, str, List[str]]:
    if dataset_name == "mathqa":
        labels = ["A", "B", "C", "D", "E"]
        options_text = example["options"]
        prompt = (
            "You are solving a multiple choice question.\n"
            "Return only the capital letter of the correct answer.\n\n"
            f"Question: {example['Problem']}\n"
            f"Options: {options_text}\n"
            "Answer:"
        )
        answer = example["correct"].upper()
        return prompt, answer, labels

    if dataset_name == "openbookqa":
        labels = list(example["choices"]["label"])
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(example["choices"]["label"], example["choices"]["text"])
        )
        prompt = (
            "You are solving a multiple choice question.\n"
            "Return only the capital letter of the correct answer.\n\n"
            f"Question: {example['question_stem']}\n"
            f"{choices}\n"
            "Answer:"
        )
        answer = example["answerKey"].upper()
        return prompt, answer, labels

    if dataset_name == "arc_challenge":
        labels = list(example["choices"]["label"])
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(example["choices"]["label"], example["choices"]["text"])
        )
        prompt = (
            "You are solving a multiple choice question.\n"
            "Return only the capital letter of the correct answer.\n\n"
            f"Question: {example['question']}\n"
            f"{choices}\n"
            "Answer:"
        )
        answer = example["answerKey"].upper()
        return prompt, answer, labels

    raise KeyError(f"Unsupported dataset: {dataset_name}")


def extract_answer_label(text: str, labels: Iterable[str]) -> str | None:
    label_set = {label.upper() for label in labels}
    for match in OPTION_RE.findall(text.upper()):
        if match in label_set:
            return match
    return None


def build_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def predict_choice(model, tokenizer, prompt: str, labels: Iterable[str], max_new_tokens: int = 4) -> str | None:
    rendered_prompt = build_chat_prompt(tokenizer, prompt)

    if hasattr(model, "device"):
        device = model.device
    elif hasattr(model, "hf_device_map"):
        device = list(model.hf_device_map.values())[0]
    else:
        device = "cuda:0"

    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)

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
            if candidate_ids:
                logits = model(**inputs, use_cache=False).logits[0, -1]
                choice_scores = logits[candidate_ids]
                return normalized_labels[int(choice_scores.argmax().item())]

            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
            output = tokenizer.decode(generated[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return extract_answer_label(output, labels)


def _resolve_model_device(model):
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "hf_device_map"):
        return list(model.hf_device_map.values())[0]
    return "cuda:0"


def predict_choices_batched(
    model,
    tokenizer,
    prompts: List[str],
    labels: Iterable[str],
) -> List[str | None]:
    if not prompts:
        return []

    candidate_ids = []
    normalized_labels = []
    for label in labels:
        token_ids = tokenizer.encode(f" {label.upper()}", add_special_tokens=False)
        if len(token_ids) != 1:
            candidate_ids = []
            break
        candidate_ids.append(token_ids[0])
        normalized_labels.append(label.upper())

    device = _resolve_model_device(model)
    rendered_prompts = [build_chat_prompt(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(rendered_prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        with maybe_bf16_autocast():
            outputs = model(**inputs, use_cache=False)

    if candidate_ids:
        logits = outputs.logits
        seq_lens = inputs["attention_mask"].sum(dim=-1) - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        next_token_logits = logits[batch_indices, seq_lens]
        choice_scores = next_token_logits[:, candidate_ids]
        pred_indices = choice_scores.argmax(dim=-1).tolist()
        return [normalized_labels[idx] for idx in pred_indices]

    predictions: List[str | None] = []
    generated = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_lengths = inputs["attention_mask"].sum(dim=-1).tolist()
    for row_idx, input_len in enumerate(input_lengths):
        output = tokenizer.decode(generated[row_idx][input_len:], skip_special_tokens=True)
        predictions.append(extract_answer_label(output, labels))
    return predictions


def evaluate_mcqa_dataset(
    model,
    tokenizer,
    dataset_name: str,
    limit: int | None = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    dataset = load_mcqa_dataset(dataset_name)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

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
        preds = predict_choices_batched(model, tokenizer, batch_prompts, batch_labels)
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

    return {
        "dataset_size": float(total),
        "correct": float(correct),
        "skipped": float(skipped),
        "accuracy": float(correct / total) if total else 0.0,
    }
