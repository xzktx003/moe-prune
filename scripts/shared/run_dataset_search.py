"""Dataset-driven evalscope tau/beta search.

Drives a single (method, model, dataset) search. Results are staged as::

    results/<method>/<model_name>/<dataset>_search/
        search_config.json
        search_records.json
        by_knob/<knob>_<value>.json     # per-knob cached record (reuse key)
        selected_targets.json
        eval/<knob>_<value>/            # evalscope work_dir
            runtime_stats.json
            reports/<model>/<dataset>.json

On every target, we first inspect ``by_knob/`` for an existing cached
record. If the record is valid (``avg_dynamic_pruning_ratio`` and the
dataset metric are present), the corresponding knob value is reused and
no subprocess is launched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from moe_prune.code.scripts.shared.run_evalscope_eval import REPO_ROOT, SUPPORTED_METHODS
from moe_prune.code.src.dataset_registry import (
    EVALSCOPE_DATASETS,
    DatasetSpec,
    dataset_search_dir,
    default_evalscope_generation_max_tokens,
    default_knob_bounds_for_dataset,
    model_name_for_path,
    require_dataset,
)
from moe_prune.code.src.evalscope_search import (
    archive_incomplete_work_dir,
    binary_search_record_for_target,
    default_eval_batch_size,
    knob_name_for_method,
    method_output_dir_name,
    normalize_knob_value,
    remove_empty_dirs,
    select_record_for_target,
    write_json,
    write_search_artifacts,
)
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection
from moe_prune.code.src.quantile_search import (
    QUANTILE_COMPATIBLE_METHODS,
    build_threshold_table_from_global_candidates,
    candidate_collection_cache_matches,
    load_candidate_collection_chunks,
    write_candidate_collection_meta,
)


SEARCHABLE_METHODS = tuple(method for method in sorted(SUPPORTED_METHODS) if method != 'none')
DATASET_QUANTILE_METHODS = tuple(method for method in SEARCHABLE_METHODS if method in QUANTILE_COMPATIBLE_METHODS)


def default_search_mode_for_method(method: str) -> str:
    return 'quantile' if str(method).lower() in DATASET_QUANTILE_METHODS else 'binary'


def _resolved_dataset_hub(args: argparse.Namespace, spec: DatasetSpec) -> str:
    return 'modelscope' if spec.name == 'livecodebench' else args.dataset_hub


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Per-dataset evalscope tau/beta search.')
    add_model_selection_args(parser)
    parser.add_argument('--method', choices=SEARCHABLE_METHODS, required=True)
    parser.add_argument(
        '--dataset', required=True, choices=EVALSCOPE_DATASETS,
        help='Single evalscope dataset to drive the search.',
    )
    parser.add_argument('--search-mode', choices=['binary', 'grid', 'quantile'], default=None)
    parser.add_argument('--tau-grid', type=float, nargs='*', default=None)
    parser.add_argument('--beta-grid', type=float, nargs='*', default=None)
    parser.add_argument('--tau-min', type=float, default=None)
    parser.add_argument('--tau-max', type=float, default=None)
    parser.add_argument('--beta-min', type=float, default=None)
    parser.add_argument('--beta-max', type=float, default=None)
    parser.add_argument('--search-tolerance', type=float, default=0.01)
    parser.add_argument('--max-search-steps', type=int, default=10)
    parser.add_argument('--target-pruning-ratios', type=float, nargs='+', required=True)
    parser.add_argument('--lambda-penalty', type=float, default=0.5)
    parser.add_argument('--gamma-keep', type=float, default=0.5)
    parser.add_argument('--similarity-mode', default='fast')
    parser.add_argument('--score-path', default=None,
                        help='DiEP per-layer-beta pickle (required when method=diep).')
    parser.add_argument('--layer-importance-path', default=None,
                        help='MoDES layer-importance pickle (optional; auto-resolved if omitted).')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max samples per dataset (None = full; required spec is full-dataset).')
    parser.add_argument('--eval-batch-size', type=int, default=None)
    parser.add_argument('--precision', default='bfloat16')
    parser.add_argument('--generation-max-tokens', type=int, default=None,
                        help='Override search-time evalscope cap; defaults to a dataset-aware shorter limit.')
    parser.add_argument('--dataset-hub', default='huggingface', choices=['huggingface', 'modelscope'])
    parser.add_argument('--results-root', type=Path, default=None)
    parser.add_argument('--search-dir', type=Path, default=None)
    parser.add_argument('--skip-completed', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--router-weight-centering',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Center router weights within each layer for the RCR branch (default: false).',
    )
    args = finalize_model_selection(parser.parse_args(argv))
    if args.search_mode is None:
        args.search_mode = default_search_mode_for_method(args.method)
    return args


def _resolve_bounds(args: argparse.Namespace, knob_name: str) -> tuple[float, float]:
    default_lo, default_hi = default_knob_bounds_for_dataset(args.dataset, args.method)
    if knob_name == 'beta':
        lo = default_lo if args.beta_min is None else float(args.beta_min)
        hi = default_hi if args.beta_max is None else float(args.beta_max)
    else:
        lo = default_lo if args.tau_min is None else float(args.tau_min)
        hi = default_hi if args.tau_max is None else float(args.tau_max)
    return lo, hi


def _build_search_dir(args: argparse.Namespace) -> Path:
    if args.search_dir is not None:
        return args.search_dir
    results_root = args.results_root or (REPO_ROOT / 'results')
    return dataset_search_dir(
        results_root,
        method_output_dir_name(args.method),
        model_name_for_path(args.model_path),
        args.dataset,
    )


def _per_knob_run_dir(search_dir: Path, knob_name: str, knob_value: float) -> Path:
    return search_dir / 'eval' / f'{knob_name}_{float(knob_value):.6f}'


def _per_knob_cache_path(search_dir: Path, knob_name: str, knob_value: float) -> Path:
    return search_dir / 'by_knob' / f'{knob_name}_{float(knob_value):.6f}.json'


def _load_report_score(work_dir: Path, spec: DatasetSpec) -> Optional[float]:
    reports_root = work_dir / 'reports'
    if not reports_root.exists():
        return None
    for model_dir in sorted(path for path in reports_root.iterdir() if path.is_dir()):
        report_name = spec.evalscope_report_filename or f'{spec.evalscope_benchmark}.json'
        report_path = model_dir / report_name
        if not report_path.exists():
            continue
        try:
            payload = json.loads(report_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        score = payload.get('score')
        if score is None:
            score = payload.get('metric_score') or payload.get('accuracy')
        if score is not None:
            return float(score)
    return None


def _load_pruning_ratio(work_dir: Path) -> Optional[float]:
    stats_path = work_dir / 'runtime_stats.json'
    if not stats_path.exists():
        return None
    try:
        payload = json.loads(stats_path.read_text(encoding='utf-8'))
    except Exception:
        return None
    value = payload.get('avg_dynamic_pruning_ratio')
    return None if value is None else float(value)


def _record_is_valid(record: Optional[dict], spec: DatasetSpec) -> bool:
    if not record:
        return False
    if record.get('avg_dynamic_pruning_ratio') is None:
        return False
    return record.get(spec.metric_key) is not None


def _build_record(
    knob_name: str,
    knob_value: float,
    spec: DatasetSpec,
    work_dir: Path,
) -> dict:
    score = _load_report_score(work_dir, spec)
    prune = _load_pruning_ratio(work_dir)
    return {
        knob_name: float(knob_value),
        'run_dir': str(work_dir),
        'runtime_stats_path': str(work_dir / 'runtime_stats.json'),
        'dataset': spec.name,
        'metric_key': spec.metric_key,
        spec.metric_key: score,
        'avg_accuracy': score,
        'avg_dynamic_pruning_ratio': prune,
    }


def _build_eval_command(
    args: argparse.Namespace,
    knob_name: str,
    knob_value: float,
    work_dir: Path,
    spec: DatasetSpec,
) -> List[str]:
    eval_batch_size = args.eval_batch_size or default_eval_batch_size(args.method)
    dataset_hub = _resolved_dataset_hub(args, spec)
    generation_max_tokens = (
        int(args.generation_max_tokens)
        if args.generation_max_tokens is not None
        else default_evalscope_generation_max_tokens(spec.name)
    )
    command = [
        sys.executable, '-m', 'moe_prune.code.scripts.shared.run_evalscope_eval',
        '--model-family', args.model_family,
        '--model-path', args.model_path,
        '--method', args.method,
        f'--{knob_name}', str(float(knob_value)),
        '--datasets', spec.evalscope_benchmark,
        '--eval-batch-size', str(eval_batch_size),
        '--precision', args.precision,
        '--generation-max-tokens', str(generation_max_tokens),
        '--dataset-hub', dataset_hub,
        '--work-dir', str(work_dir),
    ]
    if spec.evalscope_subset and spec.evalscope_benchmark == 'arc':
        command.extend(['--arc-subset', spec.evalscope_subset])
    if args.limit is not None:
        command.extend(['--limit', str(args.limit)])
    if knob_name == 'tau':
        if args.method == 'diep':
            if not args.score_path:
                raise SystemExit('method=diep requires --score-path')
            command.extend(['--score-path', str(args.score_path)])
        elif args.method == 'modes':
            if args.layer_importance_path:
                command.extend(['--layer-importance-path', str(args.layer_importance_path)])
        else:
            command.extend([
                '--lambda-penalty', str(args.lambda_penalty),
                '--gamma-keep', str(args.gamma_keep),
                '--similarity-mode', args.similarity_mode,
            ])
    if args.method in {'ace', 'rcr'} and args.router_weight_centering:
        command.append('--router-weight-centering')
    return command


def _build_quantile_collect_command(
    args: argparse.Namespace,
    work_dir: Path,
    chunks_dir: Path,
    spec: DatasetSpec,
) -> List[str]:
    eval_batch_size = args.eval_batch_size or default_eval_batch_size(args.method)
    dataset_hub = _resolved_dataset_hub(args, spec)
    generation_max_tokens = (
        int(args.generation_max_tokens)
        if args.generation_max_tokens is not None
        else default_evalscope_generation_max_tokens(spec.name)
    )
    command = [
        sys.executable, '-m', 'moe_prune.code.scripts.shared.run_evalscope_eval',
        '--model-family', args.model_family,
        '--model-path', args.model_path,
        '--method', 'none',
        '--quantile-collect-method', args.method,
        '--quantile-collect-dir', str(chunks_dir),
        '--datasets', spec.evalscope_benchmark,
        '--eval-batch-size', str(eval_batch_size),
        '--precision', args.precision,
        '--generation-max-tokens', str(generation_max_tokens),
        '--dataset-hub', dataset_hub,
        '--work-dir', str(work_dir),
    ]
    if spec.evalscope_subset and spec.evalscope_benchmark == 'arc':
        command.extend(['--arc-subset', spec.evalscope_subset])
    if args.limit is not None:
        command.extend(['--limit', str(args.limit)])
    if args.method in {'ace', 'rcr'} and args.router_weight_centering:
        command.append('--router-weight-centering')
    return command


def _run_quantile_collection(
    args: argparse.Namespace,
    search_dir: Path,
    spec: DatasetSpec,
) -> dict:
    output_path = search_dir / 'threshold_table.json'
    collect_work_dir = search_dir / 'quantile_eval'
    chunks_dir = search_dir / 'quantile_collect_chunks'
    dataset_hub = _resolved_dataset_hub(args, spec)
    generation_max_tokens = (
        int(args.generation_max_tokens)
        if args.generation_max_tokens is not None
        else default_evalscope_generation_max_tokens(spec.name)
    )
    cache_signature = {
        'source': 'evalscope_full_unpruned_generate',
        'method': args.method,
        'model_family': args.model_family,
        'model_path': args.model_path,
        'dataset': args.dataset,
        'limit': args.limit,
        'precision': args.precision,
        'generation_max_tokens': generation_max_tokens,
        'dataset_hub': dataset_hub,
        'router_weight_centering': bool(args.router_weight_centering),
    }
    if not candidate_collection_cache_matches(chunks_dir, cache_signature):
        archived = archive_incomplete_work_dir(collect_work_dir)
        if archived is not None:
            print(f'[run_dataset_search] archived incomplete work_dir {collect_work_dir} -> {archived}', flush=True)
        archived_chunks = archive_incomplete_work_dir(chunks_dir)
        if archived_chunks is not None:
            print(f'[run_dataset_search] archived incomplete work_dir {chunks_dir} -> {archived_chunks}', flush=True)

        command = _build_quantile_collect_command(args, collect_work_dir, chunks_dir, spec)
        print('[run_dataset_search] quantile collect ->', ' '.join(command), flush=True)
        subprocess.run(command, check=True, cwd=str(REPO_ROOT))

    global_candidates, total_slots, chunk_count = load_candidate_collection_chunks(chunks_dir)
    threshold_table, stats_table = build_threshold_table_from_global_candidates(
        global_candidates=global_candidates,
        total_slots=total_slots,
        target_rates=args.target_pruning_ratios,
        target_is_global_slot_rate=True,
    )
    payload = {
        'method': args.method,
        'model_family': args.model_family,
        'model_path': args.model_path,
        'dataset': args.dataset,
        'limit': args.limit,
        'precision': args.precision,
        'generation_max_tokens': generation_max_tokens,
        'quantile_source': 'evalscope_full_unpruned_generate',
        'target_is_global_slot_rate': True,
        'target_pruning_ratios': [float(v) for v in args.target_pruning_ratios],
        'threshold_table': {f'{float(rate):.6f}': float(tau) for rate, tau in threshold_table.items()},
        'stats_table': {f'{float(rate):.6f}': stats for rate, stats in stats_table.items()},
        'candidate_cache_dir': str(chunks_dir),
        'chunk_count': int(chunk_count),
    }
    write_candidate_collection_meta(
        chunks_dir,
        {
            **cache_signature,
            'collection_complete': True,
            'chunk_count': int(chunk_count),
            'total_slots': int(total_slots),
            'total_candidates': int(global_candidates.numel()),
        },
    )
    write_json(output_path, payload)
    return payload


def run_search(args: argparse.Namespace) -> int:
    spec = require_dataset(args.dataset)
    if spec.kind != 'evalscope':
        raise SystemExit(f'--dataset={args.dataset} is not an evalscope dataset; use run_ppl_search for PPL.')
    dataset_hub = _resolved_dataset_hub(args, spec)

    knob_name = knob_name_for_method(args.method)
    search_dir = _build_search_dir(args)

    records_by_knob: Dict[float, dict] = {}
    selections: List[dict] = []

    def persist_partial() -> None:
        write_search_artifacts(search_dir, list(records_by_knob.values()), selections, knob_name)

    def evaluate(knob_value: float) -> dict:
        normalized = normalize_knob_value(knob_value)
        existing = records_by_knob.get(normalized)
        if existing is not None:
            return existing

        cache_path = _per_knob_cache_path(search_dir, knob_name, normalized)
        if args.skip_completed and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding='utf-8'))
            except Exception:
                cached = None
            if _record_is_valid(cached, spec):
                records_by_knob[normalized] = cached
                persist_partial()
                return cached

        work_dir = _per_knob_run_dir(search_dir, knob_name, normalized)
        if args.skip_completed:
            reuse_record = _build_record(knob_name, normalized, spec, work_dir)
            if _record_is_valid(reuse_record, spec):
                remove_empty_dirs(work_dir)
                records_by_knob[normalized] = reuse_record
                persist_partial()
                return reuse_record

        command = _build_eval_command(args, knob_name, normalized, work_dir, spec)
        archived = archive_incomplete_work_dir(work_dir)
        if archived is not None:
            print(f'[run_dataset_search] archived incomplete work_dir {work_dir} -> {archived}', flush=True)
        print('[run_dataset_search] run ->', ' '.join(command), flush=True)
        subprocess.run(command, check=True, cwd=str(REPO_ROOT))
        remove_empty_dirs(work_dir)

        record = _build_record(knob_name, normalized, spec, work_dir)
        records_by_knob[normalized] = record
        persist_partial()
        return record

    lower, upper = _resolve_bounds(args, knob_name)
    write_json(
        search_dir / 'search_config.json',
        {
            'method': args.method,
            'model_family': args.model_family,
            'model_path': args.model_path,
            'dataset': args.dataset,
            'knob_name': knob_name,
            'search_mode': args.search_mode,
            'lower_bound': lower,
            'upper_bound': upper,
            'search_tolerance': args.search_tolerance,
            'max_search_steps': args.max_search_steps,
            'target_pruning_ratios': [float(v) for v in args.target_pruning_ratios],
            'limit': args.limit,
            'generation_max_tokens': (
                int(args.generation_max_tokens)
                if args.generation_max_tokens is not None
                else default_evalscope_generation_max_tokens(spec.name)
            ),
            'precision': args.precision,
            'dataset_hub': dataset_hub,
            'quantile_source': 'evalscope_full_unpruned_generate' if args.search_mode == 'quantile' else None,
        },
    )

    if args.search_mode == 'grid':
        grid = args.beta_grid if knob_name == 'beta' else args.tau_grid
        if not grid:
            raise SystemExit(f'--{knob_name}-grid required for grid mode')
        for value in sorted({float(v) for v in grid}):
            evaluate(value)
        selections = [
            select_record_for_target(records_by_knob.values(), target, knob_name)
            for target in args.target_pruning_ratios
        ]
    elif args.search_mode == 'quantile':
        if args.method not in DATASET_QUANTILE_METHODS:
            raise SystemExit(
                f'--search-mode quantile only supports methods in {DATASET_QUANTILE_METHODS}, '
                f'got method={args.method}. Use --search-mode binary instead.'
            )
        calibration_payload = _run_quantile_collection(args, search_dir, spec)
        threshold_table = calibration_payload.get('threshold_table', {})
        selections = []
        for target in args.target_pruning_ratios:
            key = f'{float(target):.6f}'
            if key not in threshold_table:
                raise SystemExit(
                    f'threshold_table missing entry for target_pruning_ratio={target}; '
                    f'available keys={list(threshold_table.keys())}'
                )
            knob_value = float(threshold_table[key])
            record = evaluate(knob_value)
            actual_prune = record.get('avg_dynamic_pruning_ratio')
            pruning_gap = (
                abs(float(actual_prune) - float(target))
                if actual_prune is not None
                else None
            )
            record_with_gap = dict(record)
            record_with_gap['pruning_gap'] = pruning_gap
            selections.append({
                'target_pruning_ratio': float(target),
                'selected_record': record_with_gap,
                'search_mode': 'quantile',
                'derived_knob': float(knob_value),
                'stats': calibration_payload.get('stats_table', {}).get(key),
            })
    else:
        selections = [
            binary_search_record_for_target(
                evaluate_record=evaluate,
                target_pruning_ratio=target,
                knob_name=knob_name,
                lower_bound=lower,
                upper_bound=upper,
                tolerance=args.search_tolerance,
                max_steps=args.max_search_steps,
            )
            for target in args.target_pruning_ratios
        ]

    persist_partial()
    print(
        f'[run_dataset_search] search_dir={search_dir} '
        f'records={len(records_by_knob)} targets={len(selections)} mode={args.search_mode}',
        flush=True,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    return run_search(args)


if __name__ == '__main__':
    raise SystemExit(main())
