from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from moe_prune.code.src.evalscope_search import (
    archive_incomplete_work_dir,
    binary_search_record_for_target,
    json_result_has_payload,
    normalize_knob_value,
    write_json,
    write_search_artifacts,
)
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection


REPO_ROOT = Path(__file__).resolve().parents[2]
DIEP_ROOT = REPO_ROOT / "ablation" / "DiEP"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary-search DiEP tau by target pruning ratio using MCQA evaluation.")
    add_model_selection_args(parser)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results" / "DiEP")
    parser.add_argument("--target-pruning-ratios", type=float, nargs="+", required=True)
    parser.add_argument("--tau-min", type=float, default=0.0)
    parser.add_argument("--tau-max", type=float, default=2.0)
    parser.add_argument("--search-tolerance", type=float, default=0.01)
    parser.add_argument("--max-search-steps", type=int, default=10)
    parser.add_argument("--calibration-num-samples", type=int, default=128)
    parser.add_argument("--calibration-max-length", type=int, default=2048)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--score-cache-dir", type=Path, default=None)
    parser.add_argument("--score-path", type=Path, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    return finalize_model_selection(parser.parse_args(argv))


def score_cache_path(base_dir: Path, model_path: str, num_samples: int, max_length: int, split: str) -> Path:
    fingerprint = hashlib.sha1(
        f"{Path(model_path).as_posix()}|{num_samples}|{max_length}|{split}".encode("utf-8")
    ).hexdigest()[:12]
    model_tag = Path(model_path).name or "model"
    return base_dir / f"diep_score_{model_tag}_n{num_samples}_L{max_length}_{split}_{fingerprint}.pkl"


def resolve_score_path(args: argparse.Namespace) -> Path:
    if args.score_path is not None:
        return args.score_path
    score_cache_dir = args.score_cache_dir or (args.results_root / "score_cache")
    return score_cache_path(
        score_cache_dir,
        args.model_path,
        args.calibration_num_samples,
        args.calibration_max_length,
        args.calibration_split,
    )


def ensure_score_file(args: argparse.Namespace, score_path: Path) -> None:
    if score_path.exists():
        return
    if args.skip_calibration:
        raise FileNotFoundError(f"Missing DiEP score file: {score_path}")

    cmd = [
        sys.executable,
        str(DIEP_ROOT / "qwen3_diep_ablation.py"),
        "--model-path",
        args.model_path,
        "--output-dir",
        str(args.results_root),
        "--score-path",
        str(score_path),
        "--tau-grid",
        "0.0",
        "--calibration-num-samples",
        str(args.calibration_num_samples),
        "--calibration-max-length",
        str(args.calibration_max_length),
        "--calibration-split",
        str(args.calibration_split),
        "--skip-eval",
    ]
    print("[DiEP-search] calibration ->", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(DIEP_ROOT), check=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    search_dir = args.results_root / "pruning_search"
    score_path = resolve_score_path(args)
    ensure_score_file(args, score_path)
    records_by_tau: Dict[float, Dict[str, object]] = {}
    selections = []

    def persist() -> None:
        write_search_artifacts(search_dir, list(records_by_tau.values()), selections, "tau")

    def evaluate_tau(tau_value: float) -> Dict[str, object]:
        tau_value = normalize_knob_value(tau_value)
        if tau_value in records_by_tau:
            return records_by_tau[tau_value]

        run_dir = search_dir / "eval" / f"tau_{tau_value:.6f}"
        result_path = run_dir / "results_table.json"
        if not (args.skip_completed and json_result_has_payload(result_path)):
            archived = archive_incomplete_work_dir(run_dir)
            if archived is not None:
                print(f"[DiEP-search] archived incomplete work_dir {run_dir} -> {archived}", flush=True)
            cmd = [
                sys.executable,
                str(DIEP_ROOT / "eval_qwen3_diep_tau.py"),
                "--model-path",
                args.model_path,
                "--score-path",
                str(score_path),
                "--taus",
                f"{tau_value:.6f}",
                "--batch-size",
                str(args.eval_batch_size),
                "--output-path",
                str(result_path),
            ]
            if args.eval_limit is not None:
                cmd.extend(["--eval-limit", str(args.eval_limit)])
            print("[DiEP-search] eval ->", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(DIEP_ROOT), check=True)

        row = json.loads(result_path.read_text(encoding="utf-8"))[0]
        record = {
            "tau": tau_value,
            "mathqa": float(row["mathqa"]),
            "openbookqa": float(row["openbookqa"]),
            "arc_challenge": float(row["arc_challenge"]),
            "avg_accuracy": float(row["avg_accuracy"]),
            "avg_dynamic_pruning_ratio": float(row["avg_dynamic_pruning_ratio"]),
            "run_dir": str(run_dir),
            "runtime_stats_path": str(result_path),
            "method": "DiEP",
        }
        records_by_tau[tau_value] = record
        persist()
        return record

    write_json(
        search_dir / "search_config.json",
        {
            "method": "DiEP",
            "target_pruning_ratios": [float(value) for value in args.target_pruning_ratios],
            "tau_min": float(args.tau_min),
            "tau_max": float(args.tau_max),
            "search_tolerance": float(args.search_tolerance),
            "max_search_steps": int(args.max_search_steps),
            "score_path": str(score_path),
        },
    )

    selections = [
        binary_search_record_for_target(
            evaluate_record=evaluate_tau,
            target_pruning_ratio=target,
            knob_name="tau",
            lower_bound=args.tau_min,
            upper_bound=args.tau_max,
            tolerance=args.search_tolerance,
            max_steps=args.max_search_steps,
        )
        for target in args.target_pruning_ratios
    ]
    persist()
    print(f"[DiEP-search] done search_dir={search_dir} records={len(records_by_tau)} targets={len(selections)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
