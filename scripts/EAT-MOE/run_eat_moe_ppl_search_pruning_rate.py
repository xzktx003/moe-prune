from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from moe_prune.code.src.evalscope_search import archive_incomplete_work_dir, json_result_has_payload, normalize_knob_value, write_json
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection
from moe_prune.code.src.ppl_search import binary_search_record_for_target, write_search_artifacts


REPO_ROOT = Path(__file__).resolve().parents[3]
EAT_ROOT = REPO_ROOT / 'code' / 'ablation' / 'EAT-MOE'


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Binary-search EAT-MOE tau by target pruning ratio using WikiText PPL evaluation.')
    add_model_selection_args(parser)
    parser.add_argument('--results-root', type=Path, default=REPO_ROOT / 'results' / 'EAT-MOE')
    parser.add_argument('--target-pruning-ratios', type=float, nargs='+', required=True)
    parser.add_argument('--tau-min', type=float, default=0.01)
    parser.add_argument('--tau-max', type=float, default=0.5)
    parser.add_argument('--search-tolerance', type=float, default=0.01)
    parser.add_argument('--max-search-steps', type=int, default=10)
    parser.add_argument('--min-threshold', type=float, default=0.01)
    parser.add_argument('--max-threshold', type=float, default=0.5)
    parser.add_argument('--n-ctx', type=int, default=2048)
    parser.add_argument('--split', default='test')
    parser.add_argument('--text-column', default='text')
    parser.add_argument('--min-text-length', type=int, default=512)
    parser.add_argument('--wikitext-row-limit', type=int, default=None)
    parser.add_argument('--skip-completed', action=argparse.BooleanOptionalAction, default=True)
    return finalize_model_selection(parser.parse_args(argv))


def load_row(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f'Expected at least one row in {path}')
        return dict(payload[0])
    return dict(payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    search_dir = args.results_root / 'ppl_search'
    search_dir.mkdir(parents=True, exist_ok=True)
    records_by_tau: Dict[float, Dict[str, object]] = {}
    selections = []

    def persist() -> None:
        write_search_artifacts(search_dir, list(records_by_tau.values()), selections, 'tau')

    def evaluate_tau(tau_value: float) -> Dict[str, object]:
        tau_value = normalize_knob_value(tau_value)
        if tau_value in records_by_tau:
            return records_by_tau[tau_value]

        run_dir = search_dir / 'eval' / f'tau_{tau_value:.6f}'
        result_path = run_dir / 'results_table.json'
        if not (args.skip_completed and json_result_has_payload(result_path)):
            archived = archive_incomplete_work_dir(run_dir)
            if archived is not None:
                print(f'[EAT-MOE-ppl-search] archived incomplete work_dir {run_dir} -> {archived}', flush=True)
            cmd = [
                sys.executable,
                str(EAT_ROOT / 'qwen3_eat_moe_ablation.py'),
                '--model-family',
                args.model_family,
                '--model-path',
                args.model_path,
                '--output-dir',
                str(run_dir),
                '--tau-grid',
                f'{tau_value:.6f}',
                '--min-threshold',
                str(args.min_threshold),
                '--max-threshold',
                str(args.max_threshold),
                '--n-ctx',
                str(args.n_ctx),
                '--split',
                args.split,
                '--text-column',
                args.text_column,
                '--min-text-length',
                str(args.min_text_length),
                '--skip-eval',
            ]
            if args.wikitext_row_limit is not None:
                cmd.extend(['--wikitext-row-limit', str(args.wikitext_row_limit)])
            print('[EAT-MOE-ppl-search] eval ->', ' '.join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(EAT_ROOT), check=True)

        row = load_row(result_path)
        record = {
            'tau': tau_value,
            'ppl': float(row['wikitext_ppl']),
            'avg_dynamic_pruning_ratio': float(row['avg_dynamic_pruning_ratio']),
            'rows_used': None,
            'windows': None,
            'run_dir': str(run_dir),
            'runtime_stats_path': str(result_path),
            'method': 'EAT-MOE',
        }
        records_by_tau[tau_value] = record
        persist()
        return record

    write_json(
        search_dir / 'search_config.json',
        {
            'method': 'EAT-MOE',
            'target_pruning_ratios': [float(value) for value in args.target_pruning_ratios],
            'tau_min': float(args.tau_min),
            'tau_max': float(args.tau_max),
            'search_tolerance': float(args.search_tolerance),
            'max_search_steps': int(args.max_search_steps),
            'n_ctx': args.n_ctx,
            'split': args.split,
            'text_column': args.text_column,
            'min_text_length': args.min_text_length,
            'wikitext_row_limit': args.wikitext_row_limit,
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
    print(f'[EAT-MOE-ppl-search] done search_dir={search_dir} records={len(records_by_tau)} targets={len(selections)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
