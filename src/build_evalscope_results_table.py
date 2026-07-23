from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Optional

from moe_prune.code.src.dataset_registry import DATASET_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME_MAP = {
    spec.evalscope_benchmark: spec.metric_key
    for spec in DATASET_REGISTRY.values()
    if spec.kind == 'evalscope' and spec.evalscope_benchmark
}
DATASET_NAME_MAP.update({
    'arc_easy': 'arc_easy',
    'arc-e': 'arc_easy',
    'arc-c': 'arc_challenge',
})
DATASET_KEYS = tuple(
    spec.metric_key
    for spec in DATASET_REGISTRY.values()
    if spec.kind == 'evalscope'
)


def normalize_knob(value: float) -> float:
    return round(float(value), 6)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build results_table.json from evalscope per-run report directories.',
    )
    parser.add_argument('--method', required=True)
    parser.add_argument('--knob-name', default='tau')
    parser.add_argument('--evalscope-root', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    return parser.parse_args(argv)


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def infer_knob_value(run_dir: Path, knob_name: str) -> float:
    stats_path = run_dir / 'runtime_stats.json'
    if stats_path.exists():
        payload = load_json(stats_path)
        if knob_name in payload and payload[knob_name] is not None:
            return normalize_knob(payload[knob_name])

    match = re.search(rf'{re.escape(knob_name)}([0-9.]+)', run_dir.name)
    if not match:
        raise ValueError(f'Could not infer {knob_name} from {run_dir}')
    return normalize_knob(float(match.group(1)))


def load_runtime_pruning_ratio(run_dir: Path) -> Optional[float]:
    stats_path = run_dir / 'runtime_stats.json'
    if not stats_path.exists():
        return None
    payload = load_json(stats_path)
    value = payload.get('avg_dynamic_pruning_ratio')
    return None if value is None else float(value)


def iter_report_files(run_dir: Path) -> Iterable[Path]:
    reports_root = run_dir / 'reports'
    if not reports_root.exists():
        return []
    model_dirs = [path for path in sorted(reports_root.iterdir()) if path.is_dir()]
    if not model_dirs:
        return []
    return sorted(model_dirs[0].glob('*.json'))


def build_row(run_dir: Path, knob_name: str) -> Dict[str, Optional[float]]:
    row: Dict[str, Optional[float]] = {
        knob_name: infer_knob_value(run_dir, knob_name),
        'avg_dynamic_pruning_ratio': load_runtime_pruning_ratio(run_dir),
    }
    for dataset_key in DATASET_KEYS:
        row[dataset_key] = None

    for report_path in iter_report_files(run_dir):
        payload = load_json(report_path)
        dataset_name = payload.get('dataset_name')
        normalized_name = DATASET_NAME_MAP.get(dataset_name)
        if normalized_name is None:
            continue
        row[normalized_name] = float(payload.get('score', 0.0))

    scores = [row[key] for key in DATASET_KEYS if row[key] is not None]
    row['avg_accuracy'] = float(fmean(scores)) if scores else None
    return row


def collect_rows(evalscope_root: Path, knob_name: str = 'tau') -> List[Dict[str, Optional[float]]]:
    rows = []
    if not evalscope_root.exists():
        return rows
    for run_dir in sorted(path for path in evalscope_root.iterdir() if path.is_dir()):
        if not (run_dir / 'reports').exists():
            continue
        rows.append(build_row(run_dir, knob_name=knob_name))
    return sorted(rows, key=lambda row: float(row[knob_name]))


def build_markdown(rows: List[Dict[str, Optional[float]]], knob_name: str) -> str:
    lines = [
        '| ' + ' | '.join([knob_name, *DATASET_KEYS, 'avg_accuracy', 'avg_dynamic_pruning_ratio']) + ' |',
        '| ' + ' | '.join(['---'] * (len(DATASET_KEYS) + 3)) + ' |',
    ]
    for row in rows:
        values = [f"{float(row[knob_name]):.4f}"]
        values.extend(
            '' if row[dataset_key] is None else f"{float(row[dataset_key]):.4f}"
            for dataset_key in DATASET_KEYS
        )
        values.append('' if row['avg_accuracy'] is None else f"{float(row['avg_accuracy']):.4f}")
        values.append(
            ''
            if row['avg_dynamic_pruning_ratio'] is None
            else f"{float(row['avg_dynamic_pruning_ratio']):.4f}"
        )
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def write_results(output_dir: Path, rows: List[Dict[str, Optional[float]]], knob_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [{knob_name: row[knob_name], **{k: row[k] for k in DATASET_KEYS},
                'avg_dynamic_pruning_ratio': row['avg_dynamic_pruning_ratio']}
               for row in rows]
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    markdown_text = build_markdown(rows, knob_name=knob_name)
    (output_dir / 'results_table.partial.json').write_text(json_text, encoding='utf-8')
    (output_dir / 'results_table.partial.md').write_text(markdown_text, encoding='utf-8')
    (output_dir / 'results_table.json').write_text(json_text, encoding='utf-8')
    (output_dir / 'results_table.md').write_text(markdown_text, encoding='utf-8')


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    evalscope_root = args.evalscope_root or (REPO_ROOT / 'results' / args.method / 'evalscope')
    output_dir = args.output_dir or (REPO_ROOT / 'results' / args.method)
    rows = collect_rows(evalscope_root=evalscope_root, knob_name=args.knob_name)
    write_results(output_dir=output_dir, rows=rows, knob_name=args.knob_name)
    print(f'[build_evalscope_results_table] rows={len(rows)} output_dir={output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
