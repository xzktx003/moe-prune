"""EvalScope driver for ACE, its paper ablations, and reported baselines.

This script wraps ``evalscope.run_task`` with a uniform output layout::

    results/{method}/evalscope/{tau_or_beta}/...

Example:
    python -m moe_prune.code.scripts.shared.run_evalscope_eval \
        --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --method ace --tau 0.2 \
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


SUPPORTED_METHODS = {
    'none', 'ace', 'gsp', 'rcr', 'naee',
    'score_only', 'diep', 'modes', 'aimer', 'expert_sparsity', 'top_p',
    'sere', 'xshare',
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'results'
METHOD_OUTPUT_DIRS = {
    'none': 'baseline',
    'ace': 'ACE',
    'gsp': 'GSP',
    'rcr': 'RCR',
    'naee': 'NAEE',
    'score_only': 'score_only',
    'diep': 'DiEP',
    'modes': 'MoDES',
    'aimer': 'AIMER',
    'expert_sparsity': 'ExpertSparsity',
    'top_p': 'TopP',
    'sere': 'SERE',
    'xshare': 'XShare',
}


def default_attention_implementation(model_family: str) -> str:
    return 'eager' if args_family_is_gemma4(model_family) else 'flash_attention_2'


def args_family_is_gemma4(model_family: str) -> bool:
    return str(model_family).strip().lower() == 'gemma4'


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Qwen3-MoE evalscope driver.')
    add_model_selection_args(p)
    p.add_argument('--method', choices=sorted(SUPPORTED_METHODS), default='none')
    p.add_argument('--tau', type=float, default=None)
    p.add_argument('--beta', type=float, default=None)
    p.add_argument('--lambda-penalty', type=float, default=0.5)
    p.add_argument('--gamma-keep', type=float, default=0.5)
    p.add_argument('--similarity-mode', default='fast')
    p.add_argument('--score-path', default=None,
                   help='DiEP per-layer-beta pickle (required for method=diep).')
    p.add_argument('--layer-importance-path', default=None,
                   help='MoDES layer-importance pickle (optional; auto-resolved if omitted).')
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
                   help='Override work_dir name; default = {method}_tau{tau}.')
    p.add_argument('--quantile-collect-method', default=None,
                   help='When set, run an unpruned evalscope pass while collecting quantile scores for this method.')
    p.add_argument('--quantile-collect-dir', type=Path, default=None,
                   help='Directory where evalscope quantile score chunks are written.')
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
        'prune_method': args.method,
        'precision': args.precision,
        'attn_implementation': default_attention_implementation(args.model_family),
    }
    if str(args.model_family).strip().lower() in ('qwen3.6', 'qwen3_6', 'qwen3_5', 'qwen3.5'):
        model_args['enable_thinking'] = False
    if args.quantile_collect_method is not None:
        model_args['prune_quantile_collect_method'] = args.quantile_collect_method
    if args.quantile_collect_dir is not None:
        model_args['prune_quantile_collect_dir'] = str(args.quantile_collect_dir)
    model_args['prune_router_weight_centering'] = bool(args.router_weight_centering)
    knob_tag = 'baseline'
    if args.method != 'none':
        if args.method in {'naee', 'score_only', 'expert_sparsity'}:
            if args.beta is None:
                raise SystemExit(f'--beta required for method={args.method}')
            model_args['prune_beta'] = args.beta
            knob_tag = f'beta{args.beta:.3f}'
        elif args.method == 'diep':
            if args.tau is None:
                raise SystemExit('--tau required for method=diep')
            if args.score_path is None:
                raise SystemExit('--score-path required for method=diep')
            model_args['prune_tau'] = args.tau
            model_args['prune_score_path'] = args.score_path
            knob_tag = f'tau{args.tau:.3f}'
        elif args.method == 'modes':
            if args.tau is None:
                raise SystemExit('--tau required for method=modes')
            model_args['prune_tau'] = args.tau
            if args.layer_importance_path is not None:
                model_args['prune_layer_importance_path'] = args.layer_importance_path
            knob_tag = f'tau{args.tau:.3f}'
        elif args.method in {'gsp', 'rcr', 'ace', 'aimer', 'top_p', 'sere', 'xshare'}:
            if args.tau is None:
                raise SystemExit(f'--tau required for method={args.method}')
            model_args['prune_tau'] = args.tau
            if args.method == 'sere':
                model_args['prune_similarity_mode'] = args.similarity_mode
            knob_tag = f'tau{args.tau:.3f}'
        else:
            raise SystemExit(f'Unsupported paper method: {args.method}')

    dataset_args: Dict[str, Dict[str, Any]] = {}
    if 'arc' in args.datasets:
        dataset_args['arc'] = {'subset_list': [args.arc_subset]}

    method_dir = METHOD_OUTPUT_DIRS[args.method]
    run_name = args.run_name or f'{method_dir}_{knob_tag}'
    work_dir = args.work_dir or (args.output_root / method_dir / 'evalscope' / run_name)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_args['prune_stats_path'] = str(work_dir / 'runtime_stats.json')

    return TaskConfig(
        model=args.model_path,
        eval_type='qwen3_moe_pruned',
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
