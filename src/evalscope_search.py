from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from moe_prune.code.src.dataset_registry import DATASET_REGISTRY


DATASET_KEYS = tuple(
    spec.metric_key
    for spec in DATASET_REGISTRY.values()
    if spec.kind == 'evalscope'
)
DEFAULT_REPORT_FILES = {
    spec.evalscope_benchmark: spec.evalscope_report_filename
    for spec in DATASET_REGISTRY.values()
    if spec.kind == 'evalscope' and spec.evalscope_benchmark and spec.evalscope_report_filename
}


def knob_name_for_method(method: str) -> str:
    return 'beta' if str(method).lower() in {'naee', 'score_only', 'expert_sparsity'} else 'tau'


def normalize_knob_value(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def method_output_dir_name(method: str) -> str:
    normalized = str(method).lower()
    if normalized == 'naee':
        return 'NAEE'
    if normalized == 'score_only':
        return 'score_only'
    if normalized == 'diep':
        return 'DiEP'
    if normalized == 'modes':
        return 'MoDES'
    if normalized == 'aimer':
        return 'AIMER'
    if normalized == 'expert_sparsity':
        return 'ExpertSparsity'
    if normalized == 'top_p':
        return 'TopP'
    if normalized == 'sere':
        return 'SERE'
    if normalized == 'xshare':
        return 'XShare'
    return str(method)


def default_eval_batch_size(method: str) -> int:
    normalized = str(method).lower()
    if normalized == 'modes':
        return 1
    return 8


def report_filename_for_dataset(dataset_name: str) -> str:
    return DEFAULT_REPORT_FILES.get(dataset_name, f'{dataset_name}.json')


def _rerun_archive_suffix() -> str:
    return '.stale_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def archive_incomplete_work_dir(work_dir: Path) -> Optional[Path]:
    if not work_dir.exists():
        return None
    suffix = _rerun_archive_suffix()
    archived = work_dir.with_name(work_dir.name + suffix)
    counter = 1
    while archived.exists():
        archived = work_dir.with_name(f'{work_dir.name}{suffix}_{counter}')
        counter += 1
    work_dir.rename(archived)
    return archived


def remove_empty_dirs(root: Path, *, remove_root: bool = False) -> int:
    if not root.exists() or not root.is_dir():
        return 0

    removed = 0
    for path in sorted((candidate for candidate in root.rglob('*') if candidate.is_dir()),
                       key=lambda candidate: len(candidate.parts),
                       reverse=True):
        try:
            path.rmdir()
        except OSError:
            continue
        removed += 1

    if remove_root:
        try:
            root.rmdir()
        except OSError:
            return removed
        return removed + 1

    return removed


def json_result_has_payload(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return False
    if isinstance(payload, list):
        return bool(payload)
    if isinstance(payload, dict):
        return bool(payload)
    return False


def is_evalscope_run_complete(run_dir: Path, datasets: Sequence[str]) -> bool:
    if not (run_dir / 'runtime_stats.json').exists():
        return False
    reports_root = run_dir / 'reports'
    if not reports_root.exists():
        return False
    model_dirs = sorted(path for path in reports_root.iterdir() if path.is_dir())
    if not model_dirs:
        return False
    report_dir = model_dirs[0]
    for dataset_name in datasets:
        if not (report_dir / report_filename_for_dataset(dataset_name)).exists():
            return False
    return True


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding='utf-8')


def normalize_search_record(
    row: Mapping[str, object],
    knob_name: str,
    run_dir: Path,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        knob_name: float(row[knob_name]),
        'run_dir': str(run_dir),
        'runtime_stats_path': str(run_dir / 'runtime_stats.json'),
        'avg_accuracy': row.get('avg_accuracy'),
        'avg_dynamic_pruning_ratio': row.get('avg_dynamic_pruning_ratio'),
    }
    for dataset_key in DATASET_KEYS:
        record[dataset_key] = row.get(dataset_key)
    return record


def build_search_records_markdown(
    records: Sequence[Mapping[str, object]],
    knob_name: str,
) -> str:
    headers = [knob_name, *DATASET_KEYS, 'avg_accuracy', 'avg_dynamic_pruning_ratio']
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for record in records:
        values = [f'{float(record[knob_name]):.6f}']
        values.extend(_fmt(record.get(dataset_key)) for dataset_key in DATASET_KEYS)
        values.append(_fmt(record.get('avg_accuracy')))
        values.append(_fmt(record.get('avg_dynamic_pruning_ratio')))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return ''
    return f'{float(value):.6f}'


def build_target_candidates(
    records: Iterable[Mapping[str, object]],
    target_pruning_ratio: float,
    knob_name: str,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for record in records:
        prune = record.get('avg_dynamic_pruning_ratio')
        if prune is None:
            continue
        avg_accuracy = record.get('avg_accuracy')
        pruning_gap = round(abs(float(prune) - float(target_pruning_ratio)), 12)
        candidates.append(
            {
                **record,
                'pruning_gap': pruning_gap,
                'accuracy_sort': float(avg_accuracy) if avg_accuracy is not None else float('-inf'),
                knob_name: float(record[knob_name]),
            }
        )
    candidates.sort(
        key=lambda record: (
            float(record['pruning_gap']),
            -float(record['accuracy_sort']),
            float(record[knob_name]),
        )
    )
    for record in candidates:
        record.pop('accuracy_sort', None)
    return candidates


def select_record_for_target(
    records: Iterable[Mapping[str, object]],
    target_pruning_ratio: float,
    knob_name: str,
) -> Dict[str, object]:
    candidates = build_target_candidates(records, target_pruning_ratio, knob_name)
    if not candidates:
        raise ValueError('No search records with avg_dynamic_pruning_ratio were available.')

    selected = candidates[0]
    return {
        'target_pruning_ratio': float(target_pruning_ratio),
        'knob_name': knob_name,
        'selected_knob_value': float(selected[knob_name]),
        'selected_record': selected,
        'candidates': candidates,
        'evaluation_reused_from_search': True,
    }


def binary_search_record_for_target(
    evaluate_record: Callable[[float], Mapping[str, object]],
    target_pruning_ratio: float,
    knob_name: str,
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
    max_steps: int,
) -> Dict[str, object]:
    lower = normalize_knob_value(min(lower_bound, upper_bound))
    upper = normalize_knob_value(max(lower_bound, upper_bound))
    if max_steps < 1:
        raise ValueError('max_steps must be >= 1')

    evaluated: Dict[float, Mapping[str, object]] = {}
    trace: List[Dict[str, object]] = []

    def _evaluate(knob_value: float, step: int, note: str, interval_low: float, interval_high: float) -> Mapping[str, object]:
        normalized = normalize_knob_value(knob_value)
        record = evaluated.get(normalized)
        if record is None:
            record = evaluate_record(normalized)
            evaluated[normalized] = record
        trace.append(
            {
                'step': int(step),
                'note': note,
                'lower_bound': float(interval_low),
                'upper_bound': float(interval_high),
                knob_name: float(record[knob_name]),
                'avg_accuracy': record.get('avg_accuracy'),
                'avg_dynamic_pruning_ratio': record.get('avg_dynamic_pruning_ratio'),
                'run_dir': record.get('run_dir'),
            }
        )
        return record

    lower_record = _evaluate(lower, step=0, note='lower_bound', interval_low=lower, interval_high=upper)
    upper_record = lower_record if upper == lower else _evaluate(
        upper,
        step=0,
        note='upper_bound',
        interval_low=lower,
        interval_high=upper,
    )

    lower_prune = lower_record.get('avg_dynamic_pruning_ratio')
    upper_prune = upper_record.get('avg_dynamic_pruning_ratio')
    status = 'target_out_of_range'
    target_in_range = False

    if lower_prune is not None and upper_prune is not None:
        min_prune = min(float(lower_prune), float(upper_prune))
        max_prune = max(float(lower_prune), float(upper_prune))
        target_in_range = min_prune <= float(target_pruning_ratio) <= max_prune

        if target_in_range:
            increasing = float(upper_prune) >= float(lower_prune)
            low = lower
            high = upper

            for step in range(1, max_steps + 1):
                current_best = select_record_for_target(evaluated.values(), target_pruning_ratio, knob_name)
                if float(current_best['selected_record']['pruning_gap']) <= float(tolerance):
                    status = 'tolerance_reached'
                    break

                mid = normalize_knob_value((low + high) / 2.0)
                if mid <= low or mid >= high:
                    status = 'search_interval_exhausted'
                    break

                mid_record = _evaluate(
                    mid,
                    step=step,
                    note='midpoint',
                    interval_low=low,
                    interval_high=high,
                )
                mid_prune = mid_record.get('avg_dynamic_pruning_ratio')
                if mid_prune is None:
                    status = 'missing_pruning_ratio'
                    break

                go_right = float(mid_prune) < float(target_pruning_ratio) if increasing else float(mid_prune) > float(target_pruning_ratio)
                if go_right:
                    low = mid
                    trace[-1]['decision'] = 'raise_lower_bound'
                else:
                    high = mid
                    trace[-1]['decision'] = 'lower_upper_bound'
            else:
                status = 'max_steps_reached'

    selection = select_record_for_target(evaluated.values(), target_pruning_ratio, knob_name)
    selection.update(
        {
            'search_mode': 'binary',
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'tolerance': float(tolerance),
            'max_search_steps': int(max_steps),
            'target_in_range': bool(target_in_range),
            'search_status': status,
            'trace': trace,
        }
    )
    return selection


def build_selected_targets_markdown(
    selections: Sequence[Mapping[str, object]],
    knob_name: str,
) -> str:
    headers = ['target_pruning_ratio', knob_name, 'avg_accuracy', 'avg_dynamic_pruning_ratio', 'pruning_gap']
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for selection in selections:
        selected = selection['selected_record']
        lines.append(
            '| {target:.6f} | {knob:.6f} | {avg_acc} | {avg_prune} | {gap:.6f} |'.format(
                target=float(selection['target_pruning_ratio']),
                knob=float(selected[knob_name]),
                avg_acc=_fmt(selected.get('avg_accuracy')),
                avg_prune=_fmt(selected.get('avg_dynamic_pruning_ratio')),
                gap=float(selected['pruning_gap']),
            )
        )
    return '\n'.join(lines)


def selection_row(selection: Mapping[str, object], knob_name: str) -> Dict[str, object]:
    selected = selection['selected_record']
    return {
        'target_pruning_ratio': float(selection['target_pruning_ratio']),
        knob_name: float(selected[knob_name]),
        'avg_accuracy': selected.get('avg_accuracy'),
        'avg_dynamic_pruning_ratio': selected.get('avg_dynamic_pruning_ratio'),
        'pruning_gap': selected.get('pruning_gap'),
        'run_dir': selected.get('run_dir'),
    }


def write_search_artifacts(
    search_dir: Path,
    records: Sequence[Mapping[str, object]],
    selections: Sequence[Mapping[str, object]],
    knob_name: str,
) -> None:
    ordered_records = sorted(records, key=lambda record: float(record[knob_name]))
    write_json(search_dir / 'search_records.json', list(ordered_records))
    write_text(search_dir / 'search_records.md', build_search_records_markdown(ordered_records, knob_name))

    for record in ordered_records:
        knob_value = float(record[knob_name])
        filename = f'{knob_name}_{knob_value:.6f}.json'
        write_json(search_dir / 'by_knob' / filename, record)
        write_text(
            search_dir / 'by_knob' / filename.replace('.json', '.md'),
            build_search_records_markdown([record], knob_name),
        )

    selected_rows = [selection_row(selection, knob_name) for selection in selections]
    write_json(search_dir / 'selected_targets.json', selected_rows)
    write_text(search_dir / 'selected_targets.md', build_selected_targets_markdown(selections, knob_name))

    for selection in selections:
        target = float(selection['target_pruning_ratio'])
        knob_value = float(selection['selected_record'][knob_name])
        filename = f'target_prune_{target:.6f}_selected_{knob_name}_{knob_value:.6f}.json'
        write_json(search_dir / 'targets' / filename, selection)
        write_text(
            search_dir / 'targets' / filename.replace('.json', '.md'),
            build_selected_targets_markdown([selection], knob_name),
        )
