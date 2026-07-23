"""Rebuild selected target artefacts from an existing search_records.json.

Useful when a long-running search already produced reusable per-knob records but
``selected_targets.json`` was not written (for example, interrupted orchestration
or an older target list). This is a pure post-processing step and does not run
new evaluations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from moe_prune.code.src.evalscope_search import knob_name_for_method as evalscope_knob_name_for_method
from moe_prune.code.src.evalscope_search import (
    select_record_for_target as select_evalscope_record_for_target,
)
from moe_prune.code.src.evalscope_search import write_search_artifacts as write_evalscope_search_artifacts
from moe_prune.code.src.ppl_search import (
    select_record_for_target as select_ppl_record_for_target,
)
from moe_prune.code.src.ppl_search import write_search_artifacts as write_ppl_search_artifacts


DEFAULT_TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Rebuild selected_targets.json from search_records.json.')
    parser.add_argument('--search-dir', type=Path, required=True)
    parser.add_argument('--kind', choices=('ppl', 'evalscope'), required=True)
    parser.add_argument('--method', required=True, help='Used to infer tau vs beta.')
    parser.add_argument('--target-pruning-ratios', nargs='+', type=float, default=DEFAULT_TARGETS)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    search_dir = args.search_dir.resolve()
    records_path = search_dir / 'search_records.json'
    if not records_path.exists():
        raise SystemExit(f'search_records.json not found under {search_dir}')

    payload = json.loads(records_path.read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f'search_records.json under {search_dir} is empty or not a JSON list')

    knob_name = evalscope_knob_name_for_method(args.method)
    if knob_name not in payload[0]:
        raise SystemExit(f'Could not find knob field {knob_name!r} in {records_path}')

    if args.kind == 'ppl':
        selections = [
            select_ppl_record_for_target(payload, target_pruning_ratio=target, knob_name=knob_name)
            for target in args.target_pruning_ratios
        ]
        write_ppl_search_artifacts(search_dir, payload, selections, knob_name=knob_name)
    else:
        selections = [
            select_evalscope_record_for_target(payload, target_pruning_ratio=target, knob_name=knob_name)
            for target in args.target_pruning_ratios
        ]
        write_evalscope_search_artifacts(search_dir, payload, selections, knob_name=knob_name)

    print(
        f'[rebuild_selected_targets] search_dir={search_dir} '
        f'kind={args.kind} method={args.method} targets={len(selections)}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
