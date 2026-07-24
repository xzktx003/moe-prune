"""EvalScope driver for end-to-end ACE evaluation.

This script wraps ``evalscope.run_task`` with the ACE output layout::

    results/ACE/evalscope/ACE_tau{tau}/...

Example:
    python -m moe_prune.code.scripts.shared.run_evalscope_eval \
        --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --tau 0.2 \
        --datasets arc math_qa openbookqa \
        --limit 32

For ``arc`` the subset defaults to ``ARC-Challenge`` (matching the paper plan).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# Side-effect imports: register custom model API + openbookqa benchmark.
from moe_prune.code.src import evalscope_adapter  # noqa: F401
from moe_prune.code.src import evalscope_openbookqa  # noqa: F401
from moe_prune.code.src.model_families import add_model_selection_args, finalize_model_selection


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'results'


def default_attention_implementation(model_family: str) -> str:
    return 'eager' if args_family_is_gemma4(model_family) else 'flash_attention_2'


def args_family_is_gemma4(model_family: str) -> bool:
    return str(model_family).strip().lower() == 'gemma4'


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='ACE EvalScope driver.')
    add_model_selection_args(p)
    p.add_argument('--tau', type=float, required=True)
    p.add_argument('--datasets', nargs='+', default=['arc', 'math_qa', 'openbookqa'])
    p.add_argument('--arc-subset', default='ARC-Challenge')
    p.add_argument('--limit', type=int, default=None,
                   help='Max samples per dataset (None = full).')
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--precision', default='bfloat16')
    p.add_argument('--generation-max-tokens', type=int, default=8192,
                   help='Cap generated answer length (8k per spec).')
    p.add_argument('--dataset-hub', default='huggingface',
                   choices=['huggingface', 'modelscope'])
    p.add_argument('--output-root', type=Path,
                   default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--work-dir', type=Path, default=None,
                   help='Override the exact evalscope work_dir instead of deriving it from output_root/run_name.')
    p.add_argument('--run-name', default=None,
                   help='Override work_dir name; default = ACE_tau{tau}.')
    p.add_argument(
        '--router-weight-centering',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Center router weights within each layer for the RCR branch (default: false).',
    )
    return finalize_model_selection(p.parse_args(argv))


def build_task(args: argparse.Namespace):
    from evalscope import TaskConfig

    model_args: Dict[str, Any] = {
        'prune_tau': args.tau,
        'precision': args.precision,
        'attn_implementation': default_attention_implementation(args.model_family),
    }
    if str(args.model_family).strip().lower() in ('qwen3.6', 'qwen3_6', 'qwen3_5', 'qwen3.5'):
        model_args['enable_thinking'] = False
    model_args['prune_router_weight_centering'] = bool(args.router_weight_centering)
    knob_tag = f'tau{args.tau:.3f}'

    dataset_args: Dict[str, Dict[str, Any]] = {}
    if 'arc' in args.datasets:
        dataset_args['arc'] = {'subset_list': [args.arc_subset]}

    run_name = args.run_name or f'ACE_{knob_tag}'
    work_dir = args.work_dir or (args.output_root / 'ACE' / 'evalscope' / run_name)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_args['prune_stats_path'] = str(work_dir / 'runtime_stats.json')

    return TaskConfig(
        model=args.model_path,
        eval_type='ace_moe',
        model_args=model_args,
        generation_config={
            'max_tokens': args.generation_max_tokens,
            'do_sample': False,
        },
        datasets=list(args.datasets),
        dataset_args=dataset_args,
        dataset_hub=args.dataset_hub,
        eval_batch_size=args.eval_batch_size,
        limit=args.limit,
        work_dir=str(work_dir),
        no_timestamp=True,
    )


def main() -> int:
    from evalscope import run_task

    args = parse_args()
    task = build_task(args)
    print(f'[run_evalscope_eval] work_dir={task.work_dir}')
    print(f'[run_evalscope_eval] model_args={task.model_args}')
    print(f'[run_evalscope_eval] datasets={task.datasets} hub={task.dataset_hub} limit={task.limit}')
    result = run_task(task)
    print('[run_evalscope_eval] done ->', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
