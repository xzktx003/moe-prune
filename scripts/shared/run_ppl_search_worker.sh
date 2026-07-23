#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_ROOT=$REPO_ROOT
# shellcheck source=/dev/null
source "$SCRIPT_DIR/model_family.sh"

PYTHON_BIN=${PYTHON_BIN:-python}
parse_model_family_cli_args "$@"
set -- "${MODEL_CLI_ARGS[@]}"
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
SEARCH_MODE_INPUT=${SEARCH_MODE:-auto}

if [[ $# -lt 2 ]]; then
    echo "usage: CUDA_VISIBLE_DEVICES=<gpu> SEARCH_MODE=binary   $0 <method> <target1[,target2...]> <knob_min> <knob_max> [max_steps] [tolerance]" >&2
    echo "   or: CUDA_VISIBLE_DEVICES=<gpu> SEARCH_MODE=grid     $0 <method> <target1[,target2...]> <knob1> [knob2 ...]" >&2
    echo "   or: CUDA_VISIBLE_DEVICES=<gpu> SEARCH_MODE=quantile $0 <method> <target1[,target2...]>" >&2
    echo "   or: CUDA_VISIBLE_DEVICES=<gpu> SEARCH_MODE=auto     $0 <method> <target1[,target2...]>  # default" >&2
    exit 1
fi

METHOD=$1
TARGETS_CSV=$2
shift 2

case "$METHOD" in
    ace|gsp|rcr|aimer|top_p|sere)
        GRID_FLAG=--tau-grid
        RANGE_MIN_FLAG=--tau-min
        RANGE_MAX_FLAG=--tau-max
        ;;
    naee|expert_sparsity)
        GRID_FLAG=--beta-grid
        RANGE_MIN_FLAG=--beta-min
        RANGE_MAX_FLAG=--beta-max
        ;;
    *)
        echo "unsupported method: $METHOD" >&2
        exit 1
        ;;
esac

default_search_mode_for_method() {
    case "$1" in
        ace|gsp|rcr|naee|score_only|aimer|expert_sparsity|top_p|sere)
            printf '%s\n' quantile
            ;;
        *)
            printf '%s\n' binary
            ;;
    esac
}

SEARCH_MODE=$SEARCH_MODE_INPUT
if [[ -z "$SEARCH_MODE" || "$SEARCH_MODE" == "auto" ]]; then
    SEARCH_MODE=$(default_search_mode_for_method "$METHOD")
fi

IFS=',' read -r -a TARGET_VALUES <<< "$TARGETS_CSV"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SKIP_FLAG=--skip-completed
if [[ "$SKIP_COMPLETED" != "1" ]]; then
    SKIP_FLAG=--no-skip-completed
fi

EXTRA_SEARCH_ARGS=(--search-mode "$SEARCH_MODE")
if [[ "$SEARCH_MODE" == "binary" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "binary mode requires <knob_min> <knob_max> [max_steps] [tolerance]" >&2
        exit 1
    fi
    KNOB_MIN=$1
    KNOB_MAX=$2
    MAX_SEARCH_STEPS=${3:-10}
    SEARCH_TOLERANCE=${4:-0.01}
    EXTRA_SEARCH_ARGS+=(
        "$RANGE_MIN_FLAG" "$KNOB_MIN"
        "$RANGE_MAX_FLAG" "$KNOB_MAX"
        --max-search-steps "$MAX_SEARCH_STEPS"
        --search-tolerance "$SEARCH_TOLERANCE"
    )
elif [[ "$SEARCH_MODE" == "grid" ]]; then
    KNOB_VALUES=("$@")
    EXTRA_SEARCH_ARGS+=("$GRID_FLAG" "${KNOB_VALUES[@]}")
elif [[ "$SEARCH_MODE" == "quantile" ]]; then
    :
else
    echo "unsupported SEARCH_MODE: $SEARCH_MODE" >&2
    exit 1
fi

"$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_ppl_search \
    --model-family "$MODEL_FAMILY" \
    --model-path "$MODEL_PATH" \
    --method "$METHOD" \
    --target-pruning-ratios "${TARGET_VALUES[@]}" \
    "${EXTRA_SEARCH_ARGS[@]}" \
    --n-ctx 2048 \
    --n-batch 2048 \
    "$SKIP_FLAG"
