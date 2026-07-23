#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
WORKSPACE_ROOT=$REPO_ROOT
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../shared/model_family.sh"

PYTHON_BIN=${PYTHON_BIN:-python}
parse_model_family_cli_args "$@"
set -- "${MODEL_CLI_ARGS[@]}"
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL_PATH")}
RESULTS_ROOT=${RESULTS_ROOT:-$WORKSPACE_ROOT/results/DiEP/$MODEL_TAG}

if [[ $# -lt 3 ]]; then
    echo "usage: CUDA_VISIBLE_DEVICES=<gpu> $0 <target1[,target2...]> <tau_min> <tau_max> [max_steps] [tolerance]" >&2
    exit 1
fi

TARGETS_CSV=$1
TAU_MIN=$2
TAU_MAX=$3
MAX_SEARCH_STEPS=${4:-10}
SEARCH_TOLERANCE=${5:-0.01}

IFS=',' read -r -a TARGET_VALUES <<< "$TARGETS_CSV"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m moe_prune.code.scripts.DiEP.run_diep_ppl_search_pruning_rate \
    --model-family "$MODEL_FAMILY" \
    --model-path "$MODEL_PATH" \
    --results-root "$RESULTS_ROOT" \
    --target-pruning-ratios "${TARGET_VALUES[@]}" \
    --tau-min "$TAU_MIN" \
    --tau-max "$TAU_MAX" \
    --max-search-steps "$MAX_SEARCH_STEPS" \
    --search-tolerance "$SEARCH_TOLERANCE"
