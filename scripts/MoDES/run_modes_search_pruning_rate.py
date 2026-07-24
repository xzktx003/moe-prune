from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

try:
    from moe_prune.code.src.modes_calibration import calibration_file_is_valid, calibration_path
except ModuleNotFoundError:  # pragma: no cover - direct script/test import fallback
    _HELPER_PATH = Path(__file__).resolve().parents[3] / 'code' / 'src' / 'modes_calibration.py'
    _SPEC = importlib.util.spec_from_file_location('modes_calibration_local', _HELPER_PATH)
    assert _SPEC is not None and _SPEC.loader is not None
    _MODULE = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
    calibration_file_is_valid = _MODULE.calibration_file_is_valid
    calibration_path = _MODULE.calibration_path
from moe_prune.code.src.evalscope_search import (
    archive_incomplete_work_dir,
    binary_search_record_for_target,
    json_result_has_payload,
    normalize_knob_value,
    write_json,
    write_search_artifacts,
)
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection


REPO_ROOT = Path(__file__).resolve().parents[3]
MODES_ROOT = REPO_ROOT / "code" / "ablation" / "MoDES"
DEFAULT_DATASETS = ("mathqa", "openbookqa", "arc_challenge")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary-search MoDES tau by target pruning ratio using MCQA evaluation.")
    add_model_selection_args(parser)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results" / "MoDES")
    parser.add_argument("--calibration-root", type=Path, default=REPO_ROOT / "results" / "MoDES",
                        help="Shared MoDES calibration root reused across datasets and search entrypoints.")
    parser.add_argument("--target-pruning-ratios", type=float, nargs="+", required=True)
    parser.add_argument("--tau-min", type=float, default=0.0)
    parser.add_argument("--tau-max", type=float, default=0.71)
    parser.add_argument("--search-tolerance", type=float, default=0.01)
    parser.add_argument("--max-search-steps", type=int, default=10)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--loss-type", default="kl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    return finalize_model_selection(parser.parse_args(argv))

def ensure_calibration(args: argparse.Namespace) -> Path:
    layer_importance_path = calibration_path(args.calibration_root, args.model_path, args.loss_type, args.num_samples)
    if calibration_file_is_valid(layer_importance_path):
        return layer_importance_path
    if args.skip_calibration:
        raise FileNotFoundError(f"Missing valid calibration file: {layer_importance_path}")

    cmd = [
        sys.executable,
        str(MODES_ROOT / "get_layer_importance_ddp.py"),
        "--name_or_path",
        args.model_path,
        "--save_dir",
        str(args.calibration_root / "calibration"),
        "--dataset",
        "wiki",
        "--loss_type",
        args.loss_type,
        "--batch_size",
        str(args.batch_size),
        "--num_samples",
        str(args.num_samples),
        "--max_length",
        str(args.max_length),
        "--temperature",
        str(args.temperature),
    ]
    print("[MoDES-search] calibration ->", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(MODES_ROOT), check=True)
    if not calibration_file_is_valid(layer_importance_path):
        raise RuntimeError(f"Calibration finished but did not produce a valid artifact: {layer_importance_path}")
    return layer_importance_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    search_dir = args.results_root / "pruning_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    layer_importance_path = ensure_calibration(args)
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
                print(f"[MoDES-search] archived incomplete work_dir {run_dir} -> {archived}", flush=True)
            cmd = [
                sys.executable,
                str(MODES_ROOT / "eval_qwen3_mcqa_tau.py"),
                "--name_or_path",
                args.model_path,
                "--taus",
                f"{tau_value:.6f}",
                "--layer_importance_path",
                str(layer_importance_path),
                "--batch_size",
                str(args.eval_batch_size),
                "--output_path",
                str(result_path),
            ]
            print("[MoDES-search] eval ->", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(MODES_ROOT), check=True)

        row = json.loads(result_path.read_text(encoding="utf-8"))[0]
        record = {
            "tau": tau_value,
            "mathqa": float(row["mathqa"]),
            "openbookqa": float(row["openbookqa"]),
            "arc_challenge": float(row["arc_challenge"]),
            "avg_accuracy": float((float(row["mathqa"]) + float(row["openbookqa"]) + float(row["arc_challenge"])) / 3.0),
            "avg_dynamic_pruning_ratio": float(row["avg_pruning_ratio"]),
            "run_dir": str(run_dir),
            "runtime_stats_path": str(result_path),
            "method": "MoDES",
        }
        records_by_tau[tau_value] = record
        persist()
        return record

    write_json(
        search_dir / "search_config.json",
        {
            "method": "MoDES",
            "target_pruning_ratios": [float(value) for value in args.target_pruning_ratios],
            "tau_min": float(args.tau_min),
            "tau_max": float(args.tau_max),
            "search_tolerance": float(args.search_tolerance),
            "max_search_steps": int(args.max_search_steps),
            "layer_importance_path": str(layer_importance_path),
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
    print(f"[MoDES-search] done search_dir={search_dir} records={len(records_by_tau)} targets={len(selections)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
