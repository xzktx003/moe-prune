from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from moe_prune.code.scripts.shared.run_evalscope_eval import REPO_ROOT, SUPPORTED_METHODS
from moe_prune.code.src import build_evalscope_results_table as results_builder
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection
from moe_prune.code.src.evalscope_search import (
    archive_incomplete_work_dir,
    binary_search_record_for_target,
    default_eval_batch_size,
    is_evalscope_run_complete,
    knob_name_for_method,
    method_output_dir_name,
    normalize_knob_value,
    normalize_search_record,
    select_record_for_target,
    write_json,
    write_search_artifacts,
)
from moe_prune.code.src.quantile_search import (
    QUANTILE_COMPATIBLE_METHODS,
    build_threshold_table_from_global_candidates,
    candidate_collection_cache_matches,
    load_candidate_collection_chunks,
    write_candidate_collection_meta,
)


SEARCHABLE_METHODS = tuple(method for method in sorted(SUPPORTED_METHODS) if method != 'none')
EVALSCOPE_QUANTILE_METHODS = tuple(method for method in SEARCHABLE_METHODS if method in QUANTILE_COMPATIBLE_METHODS)


def default_search_mode_for_method(method: str) -> str:
    return 'quantile' if str(method).lower() in EVALSCOPE_QUANTILE_METHODS else 'binary'


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Search tau/beta values by target pruning ratio using full evalscope test evaluation.',
    )
    add_model_selection_args(parser)
    parser.add_argument('--method', choices=SEARCHABLE_METHODS, required=True)
    parser.add_argument('--search-mode', choices=['binary', 'grid', 'quantile'], default=None)
    parser.add_argument('--tau-grid', type=float, nargs='*', default=None)
    parser.add_argument('--beta-grid', type=float, nargs='*', default=None)
    parser.add_argument('--tau-min', type=float, default=0.0)
    parser.add_argument('--tau-max', type=float, default=1.0)
    parser.add_argument('--beta-min', type=float, default=0.0)
    parser.add_argument('--beta-max', type=float, default=1.0)
    parser.add_argument('--search-tolerance', type=float, default=0.01)
    parser.add_argument('--max-search-steps', type=int, default=10)
    parser.add_argument('--target-pruning-ratios', type=float, nargs='+', required=True)
    parser.add_argument('--lambda-penalty', type=float, default=0.5)
    parser.add_argument('--gamma-keep', type=float, default=0.5)
    parser.add_argument('--similarity-mode', default='fast')
    parser.add_argument('--datasets', nargs='+', default=['arc', 'math_qa', 'openbookqa'])
    parser.add_argument('--arc-subset', default='ARC-Challenge')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--eval-batch-size', type=int, default=None)
    parser.add_argument('--precision', default='bfloat16')
    parser.add_argument('--generation-max-tokens', type=int, default=32)
    parser.add_argument('--dataset-hub', default='huggingface', choices=['huggingface', 'modelscope'])
    parser.add_argument('--score-path', default=None, help='DiEP calibrated score JSON passed to evalscope eval.')
    parser.add_argument('--layer-importance-path', default=None, help='MoDES layer-importance pickle passed to evalscope eval.')
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


def resolve_knob_grid(args: argparse.Namespace) -> tuple[str, List[float]]:
    knob_name = knob_name_for_method(args.method)
    grid = args.beta_grid if knob_name == 'beta' else args.tau_grid
    if not grid:
        required_flag = '--beta-grid' if knob_name == 'beta' else '--tau-grid'
        raise SystemExit(f'{required_flag} is required for method={args.method}')
    normalized = sorted({float(value) for value in grid})
    return knob_name, normalized


def resolve_search_bounds(args: argparse.Namespace, knob_name: str) -> tuple[float, float]:
    if knob_name == 'beta':
        return float(args.beta_min), float(args.beta_max)
    return float(args.tau_min), float(args.tau_max)


def build_search_dir(args: argparse.Namespace) -> Path:
    if args.search_dir is not None:
        return args.search_dir
    method_dir = method_output_dir_name(args.method)
    return REPO_ROOT / 'results' / method_dir / 'search'


def build_search_run_dir(search_dir: Path, method: str, knob_name: str, knob_value: float) -> Path:
    method_dir = method_output_dir_name(method)
    run_name = f'search_{method_dir}_{knob_name}{float(knob_value):.6f}'
    return search_dir / 'evalscope' / run_name


def build_evalscope_command(
    args: argparse.Namespace,
    knob_name: str,
    knob_value: float,
    work_dir: Path,
) -> List[str]:
    eval_batch_size = args.eval_batch_size or default_eval_batch_size(args.method)
    command = [
        sys.executable,
        '-m',
        'moe_prune.code.scripts.shared.run_evalscope_eval',
        '--model-family',
        args.model_family,
        '--model-path',
        args.model_path,
        '--method',
        args.method,
        f'--{knob_name}',
        str(float(knob_value)),
        '--datasets',
        *args.datasets,
        '--arc-subset',
        args.arc_subset,
        '--eval-batch-size',
        str(eval_batch_size),
        '--precision',
        args.precision,
        '--generation-max-tokens',
        str(args.generation_max_tokens),
        '--dataset-hub',
        args.dataset_hub,
        '--work-dir',
        str(work_dir),
    ]
    if args.limit is not None:
        command.extend(['--limit', str(args.limit)])
    if knob_name == 'tau':
        command.extend([
            '--lambda-penalty', str(args.lambda_penalty),
            '--gamma-keep', str(args.gamma_keep),
            '--similarity-mode', args.similarity_mode,
        ])
    if args.method == 'diep':
        if not args.score_path:
            raise SystemExit('--score-path is required for method=diep')
        command.extend(['--score-path', str(args.score_path)])
    if args.method == 'modes' and args.layer_importance_path:
        command.extend(['--layer-importance-path', str(args.layer_importance_path)])
    if args.method in {'ace', 'rcr'} and args.router_weight_centering:
        command.append('--router-weight-centering')
    return command


def build_evalscope_quantile_collect_command(
    args: argparse.Namespace,
    work_dir: Path,
    chunks_dir: Path,
) -> List[str]:
    eval_batch_size = args.eval_batch_size or default_eval_batch_size(args.method)
    command = [
        sys.executable,
        '-m',
        'moe_prune.code.scripts.shared.run_evalscope_eval',
        '--model-family',
        args.model_family,
        '--model-path',
        args.model_path,
        '--method',
        'none',
        '--quantile-collect-method',
        args.method,
        '--quantile-collect-dir',
        str(chunks_dir),
        '--datasets',
        *args.datasets,
        '--arc-subset',
        args.arc_subset,
        '--eval-batch-size',
        str(eval_batch_size),
        '--precision',
        args.precision,
        '--generation-max-tokens',
        str(args.generation_max_tokens),
        '--dataset-hub',
        args.dataset_hub,
        '--work-dir',
        str(work_dir),
    ]
    if args.limit is not None:
        command.extend(['--limit', str(args.limit)])
    if args.method in {'ace', 'rcr'} and args.router_weight_centering:
        command.append('--router-weight-centering')
    return command


def run_evalscope_quantile_collection(args: argparse.Namespace, search_dir: Path) -> dict:
    output_path = search_dir / 'threshold_table.json'
    collect_work_dir = search_dir / 'evalscope_quantile_collect'
    chunks_dir = search_dir / 'quantile_collect_chunks'
    cache_signature = {
        'source': 'evalscope_full_unpruned_generate',
        'method': args.method,
        'model_family': args.model_family,
        'model_path': args.model_path,
        'datasets': list(args.datasets),
        'arc_subset': args.arc_subset,
        'limit': args.limit,
        'precision': args.precision,
        'generation_max_tokens': int(args.generation_max_tokens),
        'dataset_hub': args.dataset_hub,
    }
    if not candidate_collection_cache_matches(chunks_dir, cache_signature):
        archived = archive_incomplete_work_dir(collect_work_dir)
        if archived is not None:
            print(f'[run_evalscope_search] archived incomplete work_dir {collect_work_dir} -> {archived}', flush=True)
        archived_chunks = archive_incomplete_work_dir(chunks_dir)
        if archived_chunks is not None:
            print(f'[run_evalscope_search] archived incomplete work_dir {chunks_dir} -> {archived_chunks}', flush=True)

        command = build_evalscope_quantile_collect_command(args, collect_work_dir, chunks_dir)
        print('[run_evalscope_search] quantile collect ->', ' '.join(command), flush=True)
        subprocess.run(command, check=True)

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
        'datasets': list(args.datasets),
        'arc_subset': args.arc_subset,
        'limit': args.limit,
        'precision': args.precision,
        'generation_max_tokens': int(args.generation_max_tokens),
        'quantile_source': 'evalscope_full_unpruned_generate',
        'target_is_global_slot_rate': True,
        'target_pruning_ratios': [float(value) for value in args.target_pruning_ratios],
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


def run_evalscope_search(args: argparse.Namespace) -> int:
    knob_name = knob_name_for_method(args.method)
    search_dir = build_search_dir(args)
    search_dir.mkdir(parents=True, exist_ok=True)
    records_by_knob: dict[float, dict] = {}
    selections = []

    def persist_partial() -> None:
        write_search_artifacts(
            search_dir,
            list(records_by_knob.values()),
            selections,
            knob_name,
        )

    def evaluate_knob_value(knob_value: float) -> dict:
        normalized = normalize_knob_value(knob_value)
        existing = records_by_knob.get(normalized)
        if existing is not None:
            return existing

        work_dir = build_search_run_dir(search_dir, args.method, knob_name, normalized)
        if not (args.skip_completed and is_evalscope_run_complete(work_dir, args.datasets)):
            archived = archive_incomplete_work_dir(work_dir)
            if archived is not None:
                print(f'[run_evalscope_search] archived incomplete work_dir {work_dir} -> {archived}', flush=True)
            command = build_evalscope_command(args, knob_name, normalized, work_dir)
            print('[run_evalscope_search] run ->', ' '.join(command), flush=True)
            subprocess.run(command, check=True)

        row = results_builder.build_row(work_dir, knob_name=knob_name)
        record = normalize_search_record(row, knob_name, work_dir)
        records_by_knob[normalized] = record
        persist_partial()
        return record

    write_json(
        search_dir / 'search_config.json',
        {
            'method': args.method,
            'knob_name': knob_name,
            'search_mode': args.search_mode,
            'knob_grid': args.beta_grid if knob_name == 'beta' else args.tau_grid,
            'lower_bound': resolve_search_bounds(args, knob_name)[0],
            'upper_bound': resolve_search_bounds(args, knob_name)[1],
            'search_tolerance': args.search_tolerance,
            'max_search_steps': args.max_search_steps,
            'target_pruning_ratios': [float(value) for value in args.target_pruning_ratios],
            'datasets': list(args.datasets),
            'limit': args.limit,
            'precision': args.precision,
            'score_path': args.score_path,
            'layer_importance_path': args.layer_importance_path,
            'quantile_source': 'evalscope_full_unpruned_generate' if args.search_mode == 'quantile' else None,
        },
    )

    if args.search_mode == 'grid':
        _, knob_grid = resolve_knob_grid(args)
        for knob_value in knob_grid:
            evaluate_knob_value(knob_value)
        selections = [
            select_record_for_target(records_by_knob.values(), target_pruning_ratio=target, knob_name=knob_name)
            for target in args.target_pruning_ratios
        ]
    elif args.search_mode == 'quantile':
        if args.method not in EVALSCOPE_QUANTILE_METHODS:
            raise SystemExit(
                f'--search-mode quantile only supports methods in {EVALSCOPE_QUANTILE_METHODS}, '
                f'got method={args.method}. Use --search-mode binary instead.'
            )
        calibration_payload = run_evalscope_quantile_collection(args, search_dir)
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
            record = evaluate_knob_value(knob_value)
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
        lower_bound, upper_bound = resolve_search_bounds(args, knob_name)
        selections = [
            binary_search_record_for_target(
                evaluate_record=evaluate_knob_value,
                target_pruning_ratio=target,
                knob_name=knob_name,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                tolerance=args.search_tolerance,
                max_steps=args.max_search_steps,
            )
            for target in args.target_pruning_ratios
        ]

    persist_partial()
    print(f'[run_evalscope_search] search_dir={search_dir} records={len(records_by_knob)} targets={len(selections)} mode={args.search_mode}')
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    return run_evalscope_search(args)


if __name__ == '__main__':
    raise SystemExit(main())
