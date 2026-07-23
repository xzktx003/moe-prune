from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from moe_prune.code.src.evalscope_search import archive_incomplete_work_dir, json_result_has_payload, normalize_knob_value, write_json
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection
from moe_prune.code.src.ppl_search import binary_search_record_for_target, write_search_artifacts


REPO_ROOT = Path(__file__).resolve().parents[2]
DIEP_ROOT = REPO_ROOT / 'ablation' / 'DiEP'


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Binary-search DiEP tau by target pruning ratio using WikiText PPL evaluation.')
    add_model_selection_args(parser)
    parser.add_argument('--results-root', type=Path, default=REPO_ROOT / 'results' / 'DiEP')
    parser.add_argument('--target-pruning-ratios', type=float, nargs='+', required=True)
    parser.add_argument('--tau-min', type=float, default=0.0)
    parser.add_argument('--tau-max', type=float, default=2.0)
    parser.add_argument('--search-tolerance', type=float, default=0.01)
    parser.add_argument('--max-search-steps', type=int, default=10)
    parser.add_argument('--calibration-num-samples', type=int, default=128)
    parser.add_argument('--calibration-max-length', type=int, default=2048)
    parser.add_argument('--calibration-split', default='train')
    parser.add_argument('--score-cache-dir', type=Path, default=None)
    parser.add_argument('--score-path', type=Path, default=None)
    parser.add_argument('--split', default='test')
    parser.add_argument('--text-column', default='text')
    parser.add_argument('--min-text-length', type=int, default=512)
    parser.add_argument('--n-ctx', type=int, default=2048)
    parser.add_argument('--skip-calibration', action='store_true')
    parser.add_argument('--skip-completed', action=argparse.BooleanOptionalAction, default=True)
    return finalize_model_selection(parser.parse_args(argv))


def score_cache_path(base_dir: Path, model_path: str, num_samples: int, max_length: int, split: str) -> Path:
    fingerprint = hashlib.sha1(
        f'{Path(model_path).as_posix()}|{num_samples}|{max_length}|{split}'.encode('utf-8')
    ).hexdigest()[:12]
    model_tag = Path(model_path).name or 'model'
    return base_dir / f'diep_score_{model_tag}_n{num_samples}_L{max_length}_{split}_{fingerprint}.pkl'


def resolve_score_path(args: argparse.Namespace) -> Path:
    if args.score_path is not None:
        return args.score_path
    score_cache_dir = args.score_cache_dir or (args.results_root / 'calibration' / 'score_cache')
    return score_cache_path(
        score_cache_dir,
        args.model_path,
        args.calibration_num_samples,
        args.calibration_max_length,
        args.calibration_split,
    )


def ensure_score_file(args: argparse.Namespace, score_path: Path) -> None:
    if score_path.exists():
        return
    if args.skip_calibration:
        raise FileNotFoundError(f'Missing DiEP score file: {score_path}')
    calibration_dir = args.results_root / 'calibration'

    cmd = [
        sys.executable,
        str(DIEP_ROOT / 'qwen3_diep_ablation.py'),
        '--model-family',
        args.model_family,
        '--model-path',
        args.model_path,
        '--output-dir',
        str(calibration_dir),
        '--score-path',
        str(score_path),
        '--tau-grid',
        '0.0',
        '--calibration-num-samples',
        str(args.calibration_num_samples),
        '--calibration-max-length',
        str(args.calibration_max_length),
        '--calibration-split',
        str(args.calibration_split),
        '--skip-eval',
    ]
    print('[DiEP-ppl-search] calibration ->', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(DIEP_ROOT), check=True)


def load_row(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f'Expected at least one row in {path}')
        return dict(payload[0])
    return dict(payload)


def result_metadata_matches(path: Path, args: argparse.Namespace, score_path: Path) -> bool:
    if not json_result_has_payload(path):
        return False
    try:
        row = load_row(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    expected = {
        'method': 'DiEP',
        'model_family': str(args.model_family),
        'model_path': str(args.model_path),
        'score_path': str(score_path),
        'split': str(args.split),
        'text_column': str(args.text_column),
        'min_text_length': int(args.min_text_length),
        'n_ctx': int(args.n_ctx),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            return False
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.results_root = args.results_root.resolve()
    if args.score_cache_dir is not None:
        args.score_cache_dir = args.score_cache_dir.resolve()
    if args.score_path is not None:
        args.score_path = args.score_path.resolve()
    search_dir = args.results_root / 'wikitext_search'
    score_path = resolve_score_path(args)
    score_path = score_path.resolve()
    ensure_score_file(args, score_path)
    records_by_tau: Dict[float, Dict[str, object]] = {}
    selections = []

    def persist() -> None:
        write_search_artifacts(search_dir, list(records_by_tau.values()), selections, 'tau')

    def evaluate_tau(tau_value: float) -> Dict[str, object]:
        tau_value = normalize_knob_value(tau_value)
        if tau_value in records_by_tau:
            return records_by_tau[tau_value]

        run_dir = search_dir / 'eval' / f'tau_{tau_value:.6f}'
        result_path = run_dir / 'wikitext_ppl.json'
        if not (args.skip_completed and result_metadata_matches(result_path, args, score_path)):
            archived = archive_incomplete_work_dir(run_dir)
            if archived is not None:
                print(f'[DiEP-ppl-search] archived incomplete work_dir {run_dir} -> {archived}', flush=True)
            cmd = [
                sys.executable,
                str(DIEP_ROOT / 'qwen3_diep_ablation.py'),
                '--model-family',
                args.model_family,
                '--model-path',
                args.model_path,
                '--output-dir',
                str(run_dir),
                '--score-path',
                str(score_path),
                '--tau-grid',
                f'{tau_value:.6f}',
                '--split',
                args.split,
                '--text-column',
                args.text_column,
                '--min-text-length',
                str(args.min_text_length),
                '--n-ctx',
                str(args.n_ctx),
                '--skip-calibration',
            ]
            print('[DiEP-ppl-search] eval ->', ' '.join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(DIEP_ROOT), check=True)

        row = load_row(result_path)
        record = {
            'method': 'DiEP',
            'model_family': str(args.model_family),
            'model_path': str(args.model_path),
            'score_path': str(score_path),
            'split': args.split,
            'text_column': args.text_column,
            'min_text_length': int(args.min_text_length),
            'n_ctx': int(args.n_ctx),
            'tau': tau_value,
            'ppl': float(row['ppl']),
            'avg_dynamic_pruning_ratio': float(row['avg_dynamic_pruning_ratio']),
            'rows_used': float(row['rows_used']) if row.get('rows_used') is not None else None,
            'windows': float(row['windows']) if row.get('windows') is not None else None,
            'run_dir': str(run_dir),
            'runtime_stats_path': str(result_path),
        }
        records_by_tau[tau_value] = record
        persist()
        return record

    write_json(
        search_dir / 'search_config.json',
        {
            'method': 'DiEP',
            'model_family': str(args.model_family),
            'model_path': str(args.model_path),
            'target_pruning_ratios': [float(value) for value in args.target_pruning_ratios],
            'tau_min': float(args.tau_min),
            'tau_max': float(args.tau_max),
            'search_tolerance': float(args.search_tolerance),
            'max_search_steps': int(args.max_search_steps),
            'score_path': str(score_path),
            'split': args.split,
            'text_column': args.text_column,
            'min_text_length': args.min_text_length,
            'n_ctx': args.n_ctx,
        },
    )

    selections = [
        binary_search_record_for_target(
            evaluate_record=evaluate_tau,
            target_pruning_ratio=target,
            knob_name='tau',
            lower_bound=args.tau_min,
            upper_bound=args.tau_max,
            tolerance=args.search_tolerance,
            max_steps=args.max_search_steps,
        )
        for target in args.target_pruning_ratios
    ]
    persist()
    print(f'[DiEP-ppl-search] done search_dir={search_dir} records={len(records_by_tau)} targets={len(selections)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
