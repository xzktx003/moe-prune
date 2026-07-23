from __future__ import annotations

import json
from pathlib import Path

from moe_prune.code.scripts.score_only.score_only_ppl_ablation import (
    build_single_device_assignment,
    collect_summary_rows,
    main,
    plan_device_assignments,
    render_command,
    tau_output_dir,
)


def test_plan_device_assignments_balances_thresholds_across_minimum_devices() -> None:
    assignments = plan_device_assignments(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        ["0", "1", "2", "3", "4"],
        max_thresholds_per_gpu=2,
    )
    assert [(assignment.device, assignment.taus) for assignment in assignments] == [
        ("0", (0.05, 0.10)),
        ("1", (0.15, 0.20)),
        ("2", (0.25, 0.30)),
        ("3", (0.35, 0.40)),
    ]


def test_build_single_device_assignment_rejects_multi_gpu_requests() -> None:
    try:
        build_single_device_assignment([0.05, 0.10], "6,7")
    except ValueError as exc:
        assert "single-GPU only" in str(exc)
    else:
        raise AssertionError("expected multi-GPU score-only plan to be rejected")


def test_render_command_expands_shell_safe_placeholders(tmp_path: Path) -> None:
    command = render_command(
        "python run.py --tau {tau} --model {model_path} --out {output_dir}",
        tau=0.15,
        model_path="/models/Qwen 3",
        model_family="qwen3",
        output_dir=tmp_path / "tau_0p15",
    )
    assert "--tau 0.15" in command
    assert "'/models/Qwen 3'" in command
    assert str(tmp_path / "tau_0p15") in command


def test_collect_summary_rows_reads_finished_and_pending_thresholds(tmp_path: Path) -> None:
    output_root = tmp_path / "score_only"
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([{"tau": 0.0, "ppl": 8.80}]), encoding="utf-8")

    completed_dir = tau_output_dir(output_root, 0.05)
    completed_dir.mkdir(parents=True, exist_ok=True)
    (completed_dir / "wikitext_ppl.json").write_text(
        json.dumps(
            [
                {
                    "tau": 0.05,
                    "ppl": 8.95,
                    "active_expert_pruning_ratio": 0.12,
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = collect_summary_rows([0.05, 0.10], output_root=output_root, baseline_json=baseline_path)
    assert rows[0].status == "done"
    assert rows[0].delta_vs_tau0 == 0.15
    assert rows[1].status == "pending"
    assert rows[1].result_json is None


def test_plan_command_writes_expected_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    exit_code = main(
        [
            "plan",
            "--model-path",
            "/models/Qwen3-30B-A3B-Instruct-2507",
            "--devices",
            "7",
            "--output-root",
            str(output_root),
        ]
    )
    assert exit_code == 0
    plan_payload = json.loads((output_root / "execution_plan.json").read_text(encoding="utf-8"))
    assert plan_payload["thresholds"] == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    assert [assignment["device"] for assignment in plan_payload["assignments"]] == ["7"]
    assert (output_root / "execution_plan.md").exists()
    assert (output_root / "summary.json").exists()
    assert (output_root / "summary.md").exists()
