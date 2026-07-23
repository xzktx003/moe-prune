from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
THIS_DIR = THIS_FILE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from moe_prune.code.src.mcqa_eval import evaluate_mcqa_dataset  # noqa: E402
from moe_prune.code.src.model_adapter import load_qwen3_moe  # noqa: E402
from moe_prune.code.src.runtime_pruner import RuntimeStats  # noqa: E402
from qwen3_diep_ablation import patch_qwen3_moe_blocks_diep  # noqa: E402


DEFAULT_DATASETS = ("mathqa", "openbookqa", "arc_challenge")


def parse_taus(raw: str) -> List[float]:
    values: List[float] = []
    for piece in str(raw).replace(",", " ").split():
        if piece:
            values.append(float(piece))
    return values


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, rows: List[Dict[str, float]]) -> None:
    lines = [
        "| tau | mathqa | openbookqa | arc_challenge | avg_accuracy | avg_dynamic_pruning_ratio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {tau:.6f} | {mathqa:.6f} | {openbookqa:.6f} | {arc_challenge:.6f} | {avg_accuracy:.6f} | {avg_dynamic_pruning_ratio:.6f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_tau(
    model,
    tokenizer,
    per_layer_beta: Dict[int, float],
    tau: float,
    datasets: Iterable[str],
    eval_limit: int | None,
    batch_size: int,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    dataset_metrics: Dict[str, Dict[str, float]] = {}
    pruning_values: List[float] = []
    for dataset_name in datasets:
        runtime_stats = RuntimeStats()
        with patch_qwen3_moe_blocks_diep(
            model=model,
            per_layer_beta=per_layer_beta,
            tau=float(tau),
            runtime_stats=runtime_stats,
        ):
            metrics = evaluate_mcqa_dataset(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                limit=eval_limit,
                batch_size=batch_size,
            )
        metrics["avg_dynamic_pruning_ratio"] = float(runtime_stats.mean_pruning_ratio())
        dataset_metrics[dataset_name] = metrics
        pruning_values.append(float(metrics["avg_dynamic_pruning_ratio"]))

    row = {
        "tau": float(tau),
        "mathqa": float(dataset_metrics["mathqa"]["accuracy"]),
        "openbookqa": float(dataset_metrics["openbookqa"]["accuracy"]),
        "arc_challenge": float(dataset_metrics["arc_challenge"]["accuracy"]),
        "avg_accuracy": float(
            sum(float(dataset_metrics[name]["accuracy"]) for name in DEFAULT_DATASETS) / len(DEFAULT_DATASETS)
        ),
        "avg_dynamic_pruning_ratio": float(sum(pruning_values) / len(pruning_values)) if pruning_values else 0.0,
        "method": "DiEP",
    }
    return dataset_metrics, row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DiEP Qwen3 tau values on MCQA datasets.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--score-path", required=True)
    parser.add_argument("--taus", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    with open(args.score_path, "rb") as handle:
        score_payload = pickle.load(handle)
    per_layer_beta = score_payload["per_layer_beta"]

    model, tokenizer = load_qwen3_moe(args.model_path)
    rows: List[Dict[str, float]] = []
    output_path = Path(args.output_path)
    markdown_path = output_path.with_suffix(".md")
    for tau in parse_taus(args.taus):
        dataset_metrics, row = evaluate_tau(
            model=model,
            tokenizer=tokenizer,
            per_layer_beta=per_layer_beta,
            tau=float(tau),
            datasets=args.datasets,
            eval_limit=args.eval_limit,
            batch_size=args.batch_size,
        )
        rows.append(row)
        write_json(output_path, rows)
        write_markdown(markdown_path, rows)
        write_json(output_path.parent / f"results_tau_{float(tau):.6f}.json", dataset_metrics)


if __name__ == "__main__":
    main()