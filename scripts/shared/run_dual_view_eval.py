"""Backward-compatible entry point for ACE dataset search."""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]


def run():
    parser = argparse.ArgumentParser(description='Run ACE dataset search.')
    parser.add_argument('--model-family', default='qwen3')
    parser.add_argument('--model-path', default='Qwen/Qwen3-30B-A3B-Instruct-2507')
    parser.add_argument('--dataset', default='piqa')
    parser.add_argument('--target-pruning-ratios', nargs='+', type=float,
                        default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--search-mode', default='quantile')
    parser.add_argument('--max-search-steps', type=int, default=6)
    parser.add_argument('--search-tolerance', type=float, default=0.01)
    parser.add_argument('--skip-completed', type=int, default=1)
    parser.add_argument(
        '--router-weight-centering',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Center router weights within each layer for ACE RCR (default: false).',
    )
    args = parser.parse_args()

    print(f"Running ACE on {args.dataset}...")
    
    cmd = [
        sys.executable,
        '-m', 'moe_prune.code.scripts.shared.run_dataset_search',
        '--model-family', args.model_family,
        '--model-path', args.model_path,
        '--method', 'ace',
        '--dataset', args.dataset,
    ]
    cmd += ['--target-pruning-ratios'] + [str(t) for t in args.target_pruning_ratios]
    cmd += ['--search-mode', args.search_mode]
    cmd += ['--max-search-steps', str(args.max_search_steps)]
    cmd += ['--search-tolerance', str(args.search_tolerance)]
    cmd += ['--skip-completed' if args.skip_completed else '--no-skip-completed']
    if args.router_weight_centering:
        cmd.append('--router-weight-centering')
    
    print(f"Executing: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode

if __name__ == '__main__':
    raise SystemExit(run())
