from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection
from moe_prune.code.src.dataset_registry import PPL_DATASETS, dataset_search_dir, model_name_for_path
from moe_prune.code.src.evalscope_search import (
    archive_incomplete_work_dir,
    json_result_has_payload,
    knob_name_for_method,
    method_output_dir_name,
    normalize_knob_value,
    write_json,
)
from moe_prune.code.src.naee_output_paths import naee_ppl_result_path, resolve_naee_output_dir
from moe_prune.code.src.ppl_search import binary_search_record_for_target, select_record_for_target, write_search_artifacts
from moe_prune.code.src.quantile_search import QUANTILE_COMPATIBLE_METHODS


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCHABLE_METHODS = (
    'ace',
    'gsp',
    'rcr',
    'naee',
    'score_only',
    'diep',
    'modes',
    'aimer',
    'expert_sparsity',
    'top_p',
    'sere',
    'xshare',
)


def default_search_mode_for_method(method: str) -> str:
    return 'quantile' if str(method).lower() in QUANTILE_COMPATIBLE_METHODS else 'binary'


def default_tau_bounds_for_method(method: str) -> tuple[float, float]:
    normalized = str(method).lower()
    if normalized in {'ace', 'gsp', 'rcr', 'aimer'}:
        return 0.0, 0.4
    if normalized == 'diep':
        return 0.0, 2.0
    if normalized == 'modes':
        return 0.0, 0.71
    return 0.0, 1.0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Search tau/beta values by target pruning ratio using full WikiText PPL evaluation.',
    )
    add_model_selection_args(parser)
    parser.add_argument('--method', choices=SEARCHABLE_METHODS, required=True)
    parser.add_argument('--search-mode', choices=['binary', 'grid', 'quantile'], default=None)
    parser.add_argument('--tau-grid', type=float, nargs='*', default=None)
    parser.add_argument('--beta-grid', type=float, nargs='*', default=None)
    parser.add_argument('--tau-min', type=float, default=None)
    parser.add_argument('--tau-max', type=float, default=None)
    parser.add_argument('--beta-min', type=float, default=0.0)
    parser.add_argument('--beta-max', type=float, default=1.0)
    parser.add_argument('--search-tolerance', type=float, default=0.01)
    parser.add_argument('--max-search-steps', type=int, default=10)
    parser.add_argument('--target-pruning-ratios', type=float, nargs='+', required=True)
    parser.add_argument('--lambda-penalty', type=float, default=0.5)
    parser.add_argument('--gamma-keep', type=float, default=0.5)
    parser.add_argument('--similarity-mode', default='fast')
    parser.add_argument('--split', default='test')
    parser.add_argument('--text-column', default='text')
    parser.add_argument('--min-text-length', type=int, default=512)
    parser.add_argument('--n-ctx', type=int, default=2048)
    parser.add_argument('--n-batch', type=int, default=2048)
    parser.add_argument('--layer-importance-path', default=None, help='MoDES layer-importance pickle path.')
    parser.add_argument('--score-path', default=None, help='DiEP calibrated score cache path.')
    parser.add_argument('--score-cache-dir', default=None, help='DiEP score cache directory.')
    parser.add_argument('--results-root', type=Path, default=None)
    parser.add_argument('--search-dir', type=Path, default=None)
    parser.add_argument(
        '--dataset', choices=PPL_DATASETS, default=None,
        help='When set (e.g. wikitext), results stage under <method>/<model>/<dataset>_search/.',
    )
    parser.add_argument('--skip-calibration', action=argparse.BooleanOptionalAction, default=False)
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
    default_lower, default_upper = default_tau_bounds_for_method(args.method)
    lower = default_lower if args.tau_min is None else float(args.tau_min)
    upper = default_upper if args.tau_max is None else float(args.tau_max)
    return lower, upper


def build_results_root(args: argparse.Namespace) -> Path:
    if args.results_root is not None:
        return args.results_root
    return REPO_ROOT / 'results' / method_output_dir_name(args.method)


def build_search_dir(args: argparse.Namespace, results_root: Path) -> Path:
    if args.search_dir is not None:
        return args.search_dir
    if args.dataset:
        return dataset_search_dir(
            results_root.parent if results_root.name == method_output_dir_name(args.method) else results_root,
            method_output_dir_name(args.method),
            model_name_for_path(args.model_path),
            args.dataset,
        )
    return results_root / 'ppl_search'


def build_search_run_dir(search_dir: Path, knob_name: str, knob_value: float) -> Path:
    return search_dir / 'eval' / f'{knob_name}_{float(knob_value):.6f}'


def format_naee_tau_dir(value: float) -> str:
    return f'tau_{float(value):.4f}'


def resolve_method_ppl_result_path(run_dir: Path, knob_name: str, knob_value: float) -> Path:
    per_knob_path = run_dir / 'ppl_by_tau' / f'{knob_name}_{float(knob_value):.4f}' / 'wikitext_ppl.json'
    legacy_tau_path = run_dir / 'ppl_by_tau' / format_naee_tau_dir(knob_value) / 'wikitext_ppl.json'
    legacy_path = run_dir / 'wikitext_ppl.json'
    if per_knob_path.exists():
        return per_knob_path
    if legacy_tau_path.exists():
        return legacy_tau_path
    return legacy_path


def build_method_command(
    args: argparse.Namespace,
    knob_name: str,
    knob_value: float,
    run_dir: Path,
    results_root: Path,
) -> List[str]:
    if args.method in {'naee', 'score_only'}:
        output_dir = resolve_naee_output_dir(results_root, method=args.method, model_path=args.model_path)
        command = [
            sys.executable,
            '-m',
            'moe_prune.code.scripts.NAEE.run_naee_ablation',
            '--method',
            args.method,
            '--model-family',
            args.model_family,
            '--model-path',
            args.model_path,
            '--output-dir',
            str(output_dir),
            '--beta-grid',
            str(float(knob_value)),
            '--skip-eval',
            '--ppl-split',
            args.split,
            '--ppl-text-column',
            args.text_column,
            '--ppl-min-text-length',
            str(args.min_text_length),
            '--ppl-n-ctx',
            str(args.n_ctx),
            '--ppl-n-batch',
            str(args.n_batch),
        ]
        if args.skip_calibration:
            command.append('--skip-calibration')
        return command

    if args.method == 'diep':
        command = [
            sys.executable,
            str(REPO_ROOT / 'ablation' / 'DiEP' / 'qwen3_diep_ablation.py'),
            '--model-family',
            args.model_family,
            '--model-path',
            args.model_path,
            '--output-dir',
            str(run_dir),
            '--tau-grid',
            str(float(knob_value)),
            '--split',
            args.split,
            '--text-column',
            args.text_column,
            '--min-text-length',
            str(args.min_text_length),
            '--n-ctx',
            str(args.n_ctx),
            '--calibration-num-samples',
            '128',
            '--calibration-max-length',
            '2048',
            '--calibration-split',
            'train',
        ]
        if args.score_path:
            command.extend(['--score-path', str(args.score_path)])
        if args.score_cache_dir:
            command.extend(['--score-cache-dir', str(args.score_cache_dir)])
        if args.skip_calibration:
            command.append('--skip-calibration')
        return command

    if args.method == 'modes':
        command = [
            sys.executable,
            str(REPO_ROOT / 'ablation' / 'MoDES' / 'eval_qwen3_text_tau.py'),
            '--name_or_path',
            args.model_path,
            '--wikitext_split',
            args.split,
            '--text_column',
            args.text_column,
            '--min_text_length',
            str(args.min_text_length),
            '--taus',
            str(float(knob_value)),
            '--n_ctx',
            str(args.n_ctx),
            '--output_path',
            str(run_dir / 'wikitext_ppl.json'),
        ]
        if args.layer_importance_path:
            command.extend(['--layer_importance_path', str(args.layer_importance_path)])
        return command

    command = [
        sys.executable,
        '-m',
        'moe_prune.code.scripts.shared.ppl_eval',
        '--model-family',
        args.model_family,
        '--model-path',
        args.model_path,
        '--output-dir',
        str(run_dir),
        '--method',
        args.method,
        '--split',
        args.split,
        '--text-column',
        args.text_column,
        '--min-text-length',
        str(args.min_text_length),
        '--n-ctx',
        str(args.n_ctx),
        '--n-batch',
        str(args.n_batch),
        '--taus',
        str(float(knob_value)),
    ]
    if args.method in {'ace', 'rcr'} and getattr(args, 'router_weight_centering', False):
        command.append('--router-weight-centering')
    return command


def load_single_row(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f'Expected at least one row in {path}')
        return dict(payload[0])
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f'Unsupported payload type in {path}: {type(payload)!r}')


def normalize_ppl_record(
    row: dict,
    method: str,
    knob_name: str,
    knob_value: float,
    run_dir: Path,
    result_path: Path,
) -> dict:
    prune = row.get('avg_dynamic_pruning_ratio')
    if prune is None:
        prune = row.get('active_expert_pruning_ratio')
    if prune is None:
        prune = row.get('pruning_ratio')
    windows = row.get('windows')
    if windows is None:
        windows = row.get('num_windows')
    return {
        knob_name: float(knob_value),
        'ppl': float(row['ppl']),
        'avg_dynamic_pruning_ratio': float(prune) if prune is not None else None,
        'rows_used': float(row['rows_used']) if row.get('rows_used') is not None else None,
        'windows': float(windows) if windows is not None else None,
        'method': method,
        'run_dir': str(run_dir),
        'runtime_stats_path': str(result_path),
    }


def run_quantile_calibration_subprocess(args: argparse.Namespace, search_dir: Path) -> dict:
    """Invoke the quantile calibration script; returns the loaded payload."""

    output_path = search_dir / 'threshold_table.json'
    cache_dir = search_dir / 'quantile_collect_chunks'
    command = [
        sys.executable,
        '-m',
        'moe_prune.code.scripts.shared.quantile_calibration',
        '--method',
        args.method,
        '--model-family',
        args.model_family,
        '--model-path',
        args.model_path,
        '--output-path',
        str(output_path),
        '--candidate-cache-dir',
        str(cache_dir),
        '--target-pruning-ratios',
        *[str(float(r)) for r in args.target_pruning_ratios],
        '--split',
        args.split,
        '--text-column',
        args.text_column,
        '--min-text-length',
        str(int(args.min_text_length)),
        '--n-ctx',
        str(int(args.n_ctx)),
        '--n-batch',
        str(int(args.n_batch)),
    ]
    if args.method in {'ace', 'rcr'} and getattr(args, 'router_weight_centering', False):
        command.append('--router-weight-centering')
    print('[run_ppl_search] quantile calibration ->', ' '.join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))

    return json.loads(output_path.read_text(encoding='utf-8'))


def run_ppl_search(args: argparse.Namespace) -> int:
    knob_name = knob_name_for_method(args.method)
    results_root = build_results_root(args)
    search_dir = build_search_dir(args, results_root)
    search_dir.mkdir(parents=True, exist_ok=True)
    records_by_knob: dict[float, dict] = {}
    selections = []

    def persist_partial() -> None:
        write_search_artifacts(search_dir, list(records_by_knob.values()), selections, knob_name)

    def evaluate_knob_value(knob_value: float) -> dict:
        normalized = normalize_knob_value(knob_value)
        existing = records_by_knob.get(normalized)
        if existing is not None:
            return existing

        run_dir = build_search_run_dir(search_dir, knob_name, normalized)
        if args.method in {'naee', 'score_only'}:
            output_dir = resolve_naee_output_dir(results_root, method=args.method, model_path=args.model_path)
            result_path = naee_ppl_result_path(output_dir, normalized)
            record_run_dir = result_path.parent
        else:
            result_path = resolve_method_ppl_result_path(run_dir, knob_name, normalized)
            record_run_dir = result_path.parent if result_path.name == 'wikitext_ppl.json' else run_dir

        if not (args.skip_completed and json_result_has_payload(result_path)):
            archived = archive_incomplete_work_dir(record_run_dir)
            if archived is not None:
                print(f'[run_ppl_search] archived incomplete work_dir {record_run_dir} -> {archived}', flush=True)
            command = build_method_command(args, knob_name, normalized, run_dir, results_root)
            print('[run_ppl_search] run ->', ' '.join(command), flush=True)
            subprocess.run(command, check=True, cwd=str(REPO_ROOT))
            if args.method not in {'naee', 'score_only'}:
                result_path = resolve_method_ppl_result_path(run_dir, knob_name, normalized)
                record_run_dir = result_path.parent

        row = load_single_row(result_path)
        record = normalize_ppl_record(row, args.method, knob_name, normalized, record_run_dir, result_path)
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
            'split': args.split,
            'text_column': args.text_column,
            'min_text_length': args.min_text_length,
            'n_ctx': args.n_ctx,
            'n_batch': args.n_batch,
            'score_path': args.score_path,
            'score_cache_dir': args.score_cache_dir,
            'layer_importance_path': args.layer_importance_path,
            'quantile_source': 'full_unpruned_eval_forward',
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
        if args.method not in QUANTILE_COMPATIBLE_METHODS:
            raise SystemExit(
                f'--search-mode quantile only supports methods in {QUANTILE_COMPATIBLE_METHODS}, '
                f'got method={args.method}. Use --search-mode binary instead.'
            )
        calibration_payload = run_quantile_calibration_subprocess(args, search_dir)
        threshold_table = calibration_payload.get('threshold_table', {})
        selections = []
        for target in args.target_pruning_ratios:
            key = f'{float(target):.6f}'
            if key not in threshold_table:
                raise SystemExit(
                    f'threshold_table missing entry for target_pruning_ratio={target}; '
                    f'available keys={list(threshold_table.keys())}'
                )
            tau_value = float(threshold_table[key])
            record = evaluate_knob_value(tau_value)
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
                'derived_tau': float(tau_value),
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
    print(
        f'[run_ppl_search] search_dir={search_dir} records={len(records_by_knob)} targets={len(selections)} mode={args.search_mode}',
        flush=True,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    return run_ppl_search(args)


if __name__ == '__main__':
    raise SystemExit(main())
