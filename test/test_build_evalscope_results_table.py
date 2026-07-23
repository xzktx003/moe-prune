from __future__ import annotations

import json
from pathlib import Path

from moe_prune.code.src import build_evalscope_results_table as builder


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def test_collect_rows_reads_evalscope_reports_and_runtime_stats(tmp_path: Path) -> None:
    run_dir = tmp_path / 'ACE_tau0.200'
    _write_json(
        run_dir / 'runtime_stats.json',
        {'tau': 0.2, 'avg_dynamic_pruning_ratio': 0.37},
    )
    _write_json(
        run_dir / 'reports' / 'Qwen3' / 'arc.json',
        {'dataset_name': 'arc', 'score': 0.61},
    )
    _write_json(
        run_dir / 'reports' / 'Qwen3' / 'openbookqa.json',
        {'dataset_name': 'openbookqa', 'score': 0.72},
    )
    _write_json(
        run_dir / 'reports' / 'Qwen3' / 'math_qa.json',
        {'dataset_name': 'math_qa', 'score': 0.53},
    )

    rows = builder.collect_rows(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row['tau'] == 0.2
    assert row['avg_dynamic_pruning_ratio'] == 0.37
    assert row['arc_challenge'] == 0.61
    assert row['avg_accuracy'] == 0.61
    assert all(row[key] is None for key in builder.DATASET_KEYS if key != 'arc_challenge')


def test_write_results_emits_method_matrix_inputs(tmp_path: Path) -> None:
    rows = [
        {
            'tau': 0.2,
            **{key: None for key in builder.DATASET_KEYS},
            'arc_challenge': 0.61,
            'avg_accuracy': 0.61,
            'avg_dynamic_pruning_ratio': 0.37,
        }
    ]

    builder.write_results(tmp_path, rows, knob_name='tau')

    payload = json.loads((tmp_path / 'results_table.json').read_text(encoding='utf-8'))
    assert payload[0]['tau'] == 0.2
    assert payload[0]['arc_challenge'] == 0.61
    assert payload[0]['avg_dynamic_pruning_ratio'] == 0.37
    assert '| tau | ' + ' | '.join(builder.DATASET_KEYS) + ' | avg_accuracy | avg_dynamic_pruning_ratio |' in (
        tmp_path / 'results_table.md'
    ).read_text(encoding='utf-8')