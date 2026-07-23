from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
DEFAULT_OUTPUT_ROOT = Path("moe_prune/results/score_only_ppl_ablation")
DEFAULT_BASELINE_JSON = Path("moe_prune/results/runs/outputs-ppl-tau0/wikitext_ppl.json")
DEFAULT_DEVICE = "0"
DEFAULT_RUNNER_TEMPLATE = (
    "python -m moe_prune.code.scripts.shared.ppl_eval "
    "--model-family {model_family} "
    "--model-path {model_path} "
    "--method score_only "
    "--taus {tau} "
    "--output-dir {output_dir}"
)


@dataclass(frozen=True)
class DeviceAssignment:
    device: str
    taus: tuple[float, ...]


@dataclass(frozen=True)
class ThresholdSummaryRow:
    tau: float
    ppl: float | None
    active_expert_pruning_ratio: float | None
    delta_vs_tau0: float | None
    output_dir: str
    result_json: str | None
    log_path: str | None
    status: str


def normalize_tau(value: float) -> float:
    return round(float(value), 6)


def format_tau(tau: float) -> str:
    return f"{normalize_tau(tau):.2f}"


def tau_label(tau: float) -> str:
    return format_tau(tau).replace(".", "p")


def tau_output_dir(output_root: Path, tau: float) -> Path:
    return output_root / f"tau_{tau_label(tau)}"


def plan_device_assignments(
    taus: Sequence[float],
    devices: Sequence[str],
    max_thresholds_per_gpu: int,
) -> list[DeviceAssignment]:
    normalized_taus = [normalize_tau(tau) for tau in taus]
    clean_devices = [device.strip() for device in devices if device.strip()]
    if not normalized_taus:
        raise ValueError("At least one tau is required.")
    if not clean_devices:
        raise ValueError("At least one CUDA device is required.")
    if max_thresholds_per_gpu < 1:
        raise ValueError("max_thresholds_per_gpu must be >= 1.")

    minimum_devices = math.ceil(len(normalized_taus) / max_thresholds_per_gpu)
    if minimum_devices > len(clean_devices):
        raise ValueError(
            "Need at least "
            f"{minimum_devices} devices for {len(normalized_taus)} thresholds with "
            f"max_thresholds_per_gpu={max_thresholds_per_gpu}, got {len(clean_devices)}."
        )

    used_device_count = min(len(clean_devices), max(minimum_devices, 1))
    base, remainder = divmod(len(normalized_taus), used_device_count)
    counts = [base + (1 if idx < remainder else 0) for idx in range(used_device_count)]

    assignments: list[DeviceAssignment] = []
    start = 0
    for device, count in zip(clean_devices[:used_device_count], counts):
        chunk = tuple(normalized_taus[start : start + count])
        if chunk:
            assignments.append(DeviceAssignment(device=device, taus=chunk))
        start += count
    return assignments


def load_baseline_ppl(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return None
    return float(rows[0].get("ppl"))


def render_command(
    runner_template: str,
    *,
    tau: float,
    model_family: str,
    model_path: str,
    output_dir: Path,
) -> str:
    substitutions = {
        "tau": format_tau(tau),
        "tau_label": tau_label(tau),
        "model_family": shlex.quote(model_family),
        "model_family_raw": model_family,
        "model_path": shlex.quote(model_path),
        "model_path_raw": model_path,
        "output_dir": shlex.quote(str(output_dir)),
        "output_dir_raw": str(output_dir),
    }
    return runner_template.format(**substitutions)


def build_plan_payload(
    assignments: Sequence[DeviceAssignment],
    *,
    output_root: Path,
    model_family: str,
    model_path: str,
    runner_template: str,
) -> dict:
    return {
        "thresholds": [tau for assignment in assignments for tau in assignment.taus],
        "output_root": str(output_root),
        "model_family": model_family,
        "runner_template": runner_template,
        "assignments": [
            {
                **asdict(assignment),
                "commands": [
                    {
                        "tau": tau,
                        "output_dir": str(tau_output_dir(output_root, tau)),
                        "command": render_command(
                            runner_template,
                            tau=tau,
                            model_family=model_family,
                            model_path=model_path,
                            output_dir=tau_output_dir(output_root, tau),
                        ),
                    }
                    for tau in assignment.taus
                ],
            }
            for assignment in assignments
        ],
    }


def write_plan_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Score-only WikiText PPL ablation execution plan",
        "",
        f"- Output root: `{payload['output_root']}`",
        f"- Thresholds: {', '.join(format_tau(tau) for tau in payload['thresholds'])}",
        f"- Runner template: `{payload['runner_template']}`",
        "- Status: plan prepared; real execution should wait for the lane-1 score-only entrypoint/commit.",
        "",
        "## Device assignments",
        "",
        "| device | taus | output_dirs |",
        "| --- | --- | --- |",
    ]
    for assignment in payload["assignments"]:
        output_dirs = "<br>".join(command["output_dir"] for command in assignment["commands"])
        lines.append(
            f"| {assignment['device']} | {', '.join(format_tau(t) for t in assignment['taus'])} | {output_dirs} |"
        )

    lines.extend(["", "## Commands", ""])
    for assignment in payload["assignments"]:
        lines.append(f"### device {assignment['device']}")
        lines.append("")
        for command in assignment["commands"]:
            lines.append(f"- tau={format_tau(command['tau'])}: `{command['command']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_threshold_row(path: Path, tau: float) -> dict | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    normalized_tau = normalize_tau(tau)
    for row in rows:
        if normalize_tau(row.get("tau", normalized_tau)) == normalized_tau:
            return row
    return rows[0] if rows else None


def collect_summary_rows(
    taus: Sequence[float],
    *,
    output_root: Path,
    baseline_json: Path | None,
) -> list[ThresholdSummaryRow]:
    baseline_ppl = load_baseline_ppl(baseline_json)
    rows: list[ThresholdSummaryRow] = []
    for tau in taus:
        result_dir = tau_output_dir(output_root, tau)
        result_json = result_dir / "wikitext_ppl.json"
        log_path = output_root / "logs" / f"tau_{tau_label(tau)}.log"
        payload = read_threshold_row(result_json, tau)
        if payload is None:
            rows.append(
                ThresholdSummaryRow(
                    tau=normalize_tau(tau),
                    ppl=None,
                    active_expert_pruning_ratio=None,
                    delta_vs_tau0=None,
                    output_dir=str(result_dir),
                    result_json=None,
                    log_path=str(log_path) if log_path.exists() else None,
                    status="pending",
                )
            )
            continue

        ppl = float(payload.get("ppl")) if payload.get("ppl") is not None else None
        delta_vs_tau0 = None
        if ppl is not None and baseline_ppl is not None:
            delta_vs_tau0 = round(ppl - baseline_ppl, 6)
        rows.append(
            ThresholdSummaryRow(
                tau=normalize_tau(tau),
                ppl=ppl,
                active_expert_pruning_ratio=(
                    float(payload.get("active_expert_pruning_ratio"))
                    if payload.get("active_expert_pruning_ratio") is not None
                    else None
                ),
                delta_vs_tau0=delta_vs_tau0,
                output_dir=str(result_dir),
                result_json=str(result_json),
                log_path=str(log_path) if log_path.exists() else None,
                status="done",
            )
        )
    return rows


def write_summary_markdown(path: Path, rows: Sequence[ThresholdSummaryRow]) -> None:
    lines = [
        "# Score-only WikiText PPL ablation summary",
        "",
        "| tau | ppl | delta_vs_tau0 | active_expert_pruning_ratio | status | artifact_dir | result_json |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_tau(row.tau),
                    "" if row.ppl is None else f"{row.ppl:.4f}",
                    "" if row.delta_vs_tau0 is None else f"{row.delta_vs_tau0:+.4f}",
                    ""
                    if row.active_expert_pruning_ratio is None
                    else f"{row.active_expert_pruning_ratio:.4f}",
                    row.status,
                    row.output_dir,
                    row.result_json or "",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


class _DeviceWorker(threading.Thread):
    def __init__(
        self,
        *,
        device: str,
        taus: Sequence[float],
        output_root: Path,
        runner_template: str,
        model_family: str,
        model_path: str,
        cwd: Path,
        shared_env: Dict[str, str],
    ) -> None:
        super().__init__(daemon=True)
        self.device = device
        self.taus = list(taus)
        self.output_root = output_root
        self.runner_template = runner_template
        self.model_family = model_family
        self.model_path = model_path
        self.cwd = cwd
        self.shared_env = shared_env
        self.error: str | None = None

    def run(self) -> None:
        env = os.environ.copy()
        env.update(self.shared_env)
        env["CUDA_VISIBLE_DEVICES"] = self.device
        logs_dir = self.output_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        for tau in self.taus:
            result_dir = tau_output_dir(self.output_root, tau)
            result_dir.mkdir(parents=True, exist_ok=True)
            command = render_command(
                self.runner_template,
                tau=tau,
                model_family=self.model_family,
                model_path=self.model_path,
                output_dir=result_dir,
            )
            log_path = logs_dir / f"tau_{tau_label(tau)}.log"
            with log_path.open("w", encoding="utf-8") as handle:
                handle.write(f"[device {self.device}] {command}\n")
                handle.flush()
                completed = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.cwd,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if completed.returncode != 0:
                self.error = (
                    f"device {self.device} failed for tau={format_tau(tau)}; "
                    f"see {log_path}"
                )
                return


def run_plan(
    assignments: Sequence[DeviceAssignment],
    *,
    output_root: Path,
    runner_template: str,
    model_family: str,
    model_path: str,
    cwd: Path,
    env: Dict[str, str],
) -> None:
    workers = [
        _DeviceWorker(
            device=assignment.device,
            taus=assignment.taus,
            output_root=output_root,
            runner_template=runner_template,
            model_family=model_family,
            model_path=model_path,
            cwd=cwd,
            shared_env=env,
        )
        for assignment in assignments
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    failures = [worker.error for worker in workers if worker.error]
    if failures:
        raise RuntimeError("; ".join(failures))


def parse_devices(raw: str) -> list[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def build_single_device_assignment(
    taus: Sequence[float],
    raw_devices: str,
) -> list[DeviceAssignment]:
    devices = parse_devices(raw_devices)
    if not devices:
        raise ValueError("At least one CUDA device is required.")
    if len(devices) != 1:
        raise ValueError(
            "Official score-only ablation is single-GPU only; pass exactly one CUDA device "
            f"(got: {', '.join(devices)})."
        )
    return [DeviceAssignment(device=devices[0], taus=tuple(normalize_tau(tau) for tau in taus))]


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    add_model_selection_args(parser)
    parser.add_argument("--devices", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--max-thresholds-per-gpu",
        type=int,
        default=2,
        help="Deprecated compatibility flag; official score-only ablation now runs all taus on one GPU.",
    )
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--runner-template",
        type=str,
        default=DEFAULT_RUNNER_TEMPLATE,
        help=(
            "Shell command template with placeholders {tau}, {tau_label}, {model_path}, "
            "{model_path_raw}, {output_dir}, and {output_dir_raw}."
        ),
    )
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)


def cmd_plan(args: argparse.Namespace) -> int:
    assignments = build_single_device_assignment(args.taus, args.devices)
    payload = build_plan_payload(
        assignments,
        output_root=args.output_root,
        model_family=args.model_family,
        model_path=args.model_path,
        runner_template=args.runner_template,
    )
    write_json(args.output_root / "execution_plan.json", payload)
    write_plan_markdown(args.output_root / "execution_plan.md", payload)
    summary_rows = collect_summary_rows(args.taus, output_root=args.output_root, baseline_json=args.baseline_json)
    write_json(args.output_root / "summary.json", [asdict(row) for row in summary_rows])
    write_summary_markdown(args.output_root / "summary.md", summary_rows)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    assignments = build_single_device_assignment(args.taus, args.devices)
    payload = build_plan_payload(
        assignments,
        output_root=args.output_root,
        model_family=args.model_family,
        model_path=args.model_path,
        runner_template=args.runner_template,
    )
    write_json(args.output_root / "execution_plan.json", payload)
    write_plan_markdown(args.output_root / "execution_plan.md", payload)

    env = {
        key: value
        for key, value in {
            "HF_HUB_OFFLINE": str(args.hf_hub_offline),
            "HF_DATASETS_OFFLINE": str(args.hf_datasets_offline),
            "TRANSFORMERS_OFFLINE": str(args.transformers_offline),
        }.items()
        if value is not None
    }
    run_plan(
        assignments,
        output_root=args.output_root,
        runner_template=args.runner_template,
        model_family=args.model_family,
        model_path=args.model_path,
        cwd=Path.cwd(),
        env=env,
    )
    summary_rows = collect_summary_rows(args.taus, output_root=args.output_root, baseline_json=args.baseline_json)
    write_json(args.output_root / "summary.json", [asdict(row) for row in summary_rows])
    write_summary_markdown(args.output_root / "summary.md", summary_rows)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    summary_rows = collect_summary_rows(args.taus, output_root=args.output_root, baseline_json=args.baseline_json)
    write_json(args.output_root / "summary.json", [asdict(row) for row in summary_rows])
    write_summary_markdown(args.output_root / "summary.md", summary_rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, run, and summarize the score-only WikiText PPL ablation on one GPU."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Write the execution plan and current summary placeholders.")
    add_shared_arguments(plan_parser)
    plan_parser.set_defaults(handler=cmd_plan)

    run_parser = subparsers.add_parser("run", help="Execute the per-device tau plan and collate results.")
    add_shared_arguments(run_parser)
    run_parser.add_argument("--hf-hub-offline", type=int, default=1)
    run_parser.add_argument("--hf-datasets-offline", type=int, default=1)
    run_parser.add_argument("--transformers-offline", type=int, default=1)
    run_parser.set_defaults(handler=cmd_run)

    summarize_parser = subparsers.add_parser("summarize", help="Refresh summary artifacts from existing per-tau outputs.")
    summarize_parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    summarize_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    summarize_parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    summarize_parser.set_defaults(handler=cmd_summarize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = finalize_model_selection(parser.parse_args(argv))
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
