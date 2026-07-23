from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

from .evalscope_search import normalize_knob_value, write_json, write_text


def _fmt(value: object) -> str:
    if value is None:
        return ''
    return f'{float(value):.6f}'


def build_search_records_markdown(
    records: Sequence[Mapping[str, object]],
    knob_name: str,
) -> str:
    headers = [knob_name, 'ppl', 'avg_dynamic_pruning_ratio', 'rows_used', 'windows']
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for record in records:
        lines.append(
            '| {knob:.6f} | {ppl} | {avg_prune} | {rows_used} | {windows} |'.format(
                knob=float(record[knob_name]),
                ppl=_fmt(record.get('ppl')),
                avg_prune=_fmt(record.get('avg_dynamic_pruning_ratio')),
                rows_used=_fmt(record.get('rows_used')),
                windows=_fmt(record.get('windows')),
            )
        )
    return '\n'.join(lines)


def build_target_candidates(
    records: Iterable[Mapping[str, object]],
    target_pruning_ratio: float,
    knob_name: str,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for record in records:
        prune = record.get('avg_dynamic_pruning_ratio')
        ppl = record.get('ppl')
        if prune is None:
            continue
        pruning_gap = round(abs(float(prune) - float(target_pruning_ratio)), 12)
        candidates.append(
            {
                **record,
                'pruning_gap': pruning_gap,
                'ppl_sort': float(ppl) if ppl is not None else float('inf'),
                knob_name: float(record[knob_name]),
            }
        )
    candidates.sort(
        key=lambda record: (
            float(record['pruning_gap']),
            float(record['ppl_sort']),
            float(record[knob_name]),
        )
    )
    for record in candidates:
        record.pop('ppl_sort', None)
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
                'ppl': record.get('ppl'),
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
    headers = ['target_pruning_ratio', knob_name, 'ppl', 'avg_dynamic_pruning_ratio', 'pruning_gap']
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for selection in selections:
        selected = selection['selected_record']
        lines.append(
            '| {target:.6f} | {knob:.6f} | {ppl} | {avg_prune} | {gap:.6f} |'.format(
                target=float(selection['target_pruning_ratio']),
                knob=float(selected[knob_name]),
                ppl=_fmt(selected.get('ppl')),
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
        'ppl': selected.get('ppl'),
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

    selected_rows = [selection_row(selection, knob_name) for selection in selections]
    write_json(search_dir / 'selected_targets.json', selected_rows)
    write_text(search_dir / 'selected_targets.md', build_selected_targets_markdown(selections, knob_name))
    for selection in selections:
        target = float(selection['target_pruning_ratio'])
        selected = selection['selected_record']
        filename = f'target_prune_{target:.6f}_selected_{knob_name}_{float(selected[knob_name]):.6f}.json'
        write_json(search_dir / 'targets' / filename, selection)