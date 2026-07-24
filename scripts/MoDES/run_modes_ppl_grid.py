from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from moe_prune.code.src.evalscope_search import write_json
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection

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


REPO_ROOT = Path(__file__).resolve().parents[3]
MODES_ROOT = REPO_ROOT / "code" / "ablation" / "MoDES"
DEFAULT_TAUS = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MoDES Qwen3 PPL on an explicit tau grid.")
    parser.add_argument("taus", nargs="*", type=float, default=list(DEFAULT_TAUS))
    add_model_selection_args(parser)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results" / "MoDES")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=-1)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--loss-type", default="kl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--skip-calibration", action="store_true")
    return finalize_model_selection(parser.parse_args(argv))


def format_tau(value: float) -> str:
    return f"{float(value):.12g}"


def format_tau_dir(value: float) -> str:
    return f"tau_{float(value):.4f}"


def write_single_markdown(path: Path, row: dict) -> None:
    lines = [
        "| tau | pruning_ratio | ppl | rows_used | num_windows |",
        "| --- | --- | --- | --- | --- |",
        "| {tau:.6f} | {pruning_ratio:.6f} | {ppl:.6f} | {rows_used:.0f} | {num_windows:.0f} |".format(**row),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_tau_outputs(output_dir: Path) -> None:
    top_level_json = output_dir / "wikitext_ppl.json"
    if not top_level_json.exists():
        return

    payload = json.loads(top_level_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return

    for row in payload:
        tau_value = float(row["tau"])
        per_tau_dir = output_dir / "ppl_by_tau" / format_tau_dir(tau_value)
        write_json(per_tau_dir / "wikitext_ppl.json", row)
        write_single_markdown(per_tau_dir / "wikitext_ppl.md", row)


def ensure_calibration(args: argparse.Namespace) -> Path:
    layer_importance_path = calibration_path(args.results_root, args.model_path, args.loss_type, args.num_samples)
    if calibration_file_is_valid(layer_importance_path):
        print(f"[MoDES-ppl-grid] reuse calibration {layer_importance_path}", flush=True)
        return layer_importance_path
    if args.skip_calibration:
        raise FileNotFoundError(f"Missing valid calibration file: {layer_importance_path}")

    cmd = [
        sys.executable,
        str(MODES_ROOT / "get_layer_importance_ddp.py"),
        "--name_or_path",
        args.model_path,
        "--save_dir",
        str(args.results_root / "calibration"),
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
    print("[MoDES-ppl-grid] calibration ->", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(MODES_ROOT), check=True)
    if not calibration_file_is_valid(layer_importance_path):
        raise RuntimeError(f"Calibration finished but did not produce a valid artifact: {layer_importance_path}")
    return layer_importance_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    layer_importance_path = ensure_calibration(args)
    output_dir = args.results_root / "ppl"
    output_dir.mkdir(parents=True, exist_ok=True)
    tau_values = [float(tau) for tau in args.taus]
    tau_csv = ",".join(format_tau(tau) for tau in tau_values)

    (output_dir / "grid_config.json").write_text(
        json.dumps(
            {
                "method": "MoDES",
                "model_path": args.model_path,
                "tau_values": tau_values,
                "layer_importance_path": str(layer_importance_path),
                "n_ctx": int(args.n_ctx),
                "max_windows": int(args.max_windows),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(MODES_ROOT / "eval_qwen3_text_tau.py"),
        "--name_or_path",
        args.model_path,
        "--taus",
        tau_csv,
        "--n_ctx",
        str(args.n_ctx),
        "--layer_importance_path",
        str(layer_importance_path),
        "--output_path",
        str(output_dir / "wikitext_ppl.json"),
    ]
    if args.max_windows > 0:
        cmd.extend(["--max_windows", str(args.max_windows)])
    print("[MoDES-ppl-grid] eval ->", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(MODES_ROOT), check=True)
    write_per_tau_outputs(output_dir)
    print(f"[MoDES-ppl-grid] done output_dir={output_dir} tau_count={len(tau_values)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
