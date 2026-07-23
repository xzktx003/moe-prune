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
MODEL_REPORT_TAG=${MODEL_REPORT_TAG:-$(basename "$MODEL_PATH")}
GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-32}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

if [[ $# -lt 2 ]]; then
    echo "usage: CUDA_VISIBLE_DEVICES=<gpu> $0 <method> <tau1> [tau2 ...]" >&2
    exit 1
fi

METHOD=$1
shift

case "$METHOD" in
    ace|gsp|rcr|top_p|sere|xshare)
        EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
        KNOB_FLAG=--tau
        ;;
    naee)
        EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
        KNOB_FLAG=--beta
        ;;
    score_only)
        EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
        KNOB_FLAG=--beta
        ;;
    diep)
        EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
        KNOB_FLAG=--tau
        : "${SCORE_PATH:?SCORE_PATH is required for method=diep}"
        ;;
    modes)
        EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
        KNOB_FLAG=--tau
        ;;
    aimer)
        EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
        KNOB_FLAG=--tau
        ;;
    *)
        echo "unsupported method: $METHOD" >&2
        exit 1
        ;;
esac

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

method_output_dir() {
    if [[ "$METHOD" == "naee" ]]; then
        echo "$WORKSPACE_ROOT/results/NAEE"
        return
    fi
    if [[ "$METHOD" == "score_only" ]]; then
        echo "$WORKSPACE_ROOT/results/score_only"
        return
    fi
    if [[ "$METHOD" == "diep" ]]; then
        echo "$WORKSPACE_ROOT/results/DiEP"
        return
    fi
    if [[ "$METHOD" == "modes" ]]; then
        echo "$WORKSPACE_ROOT/results/MoDES"
        return
    fi
    if [[ "$METHOD" == "aimer" ]]; then
        echo "$WORKSPACE_ROOT/results/AIMER"
        return
    fi
    echo "$WORKSPACE_ROOT/results/$METHOD"
}

run_dir_for_knob() {
    local knob_value=$1
    local method_dir
    method_dir=$(method_output_dir)
    if [[ "$KNOB_FLAG" == "--beta" ]]; then
        local beta_method_dir
        beta_method_dir=$(basename "$method_dir")
        printf '%s/evalscope/%s_beta%.3f' "$method_dir" "$beta_method_dir" "$knob_value"
        return
    fi
    printf '%s/evalscope/%s_tau%.3f' "$method_dir" "$METHOD" "$knob_value"
}

is_run_complete() {
    local run_dir=$1
    [[ -f "$run_dir/runtime_stats.json" ]] || return 1
    [[ -f "$run_dir/reports/$MODEL_REPORT_TAG/arc.json" ]] || return 1
    [[ -f "$run_dir/reports/$MODEL_REPORT_TAG/math_qa.json" ]] || return 1
    [[ -f "$run_dir/reports/$MODEL_REPORT_TAG/openbookqa.json" ]] || return 1
}

for knob_value in "$@"; do
    run_dir=$(run_dir_for_knob "$knob_value")
    if [[ "$SKIP_COMPLETED" == "1" ]] && is_run_complete "$run_dir"; then
        echo "[evalscope_worker] skip completed method=$METHOD knob=$knob_value run_dir=$run_dir"
        continue
    fi

    echo "[evalscope_worker] method=$METHOD knob=$knob_value gpu=${CUDA_VISIBLE_DEVICES:-unset} batch=$EVAL_BATCH_SIZE"
    extra_args=()
    if [[ "$METHOD" == "diep" ]]; then
        extra_args+=(--score-path "$SCORE_PATH")
    fi
    if [[ "$METHOD" == "modes" && -n "${LAYER_IMPORTANCE_PATH:-}" ]]; then
        extra_args+=(--layer-importance-path "$LAYER_IMPORTANCE_PATH")
    fi

    "$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_evalscope_eval \
        --model-family "$MODEL_FAMILY" \
        --model-path "$MODEL_PATH" \
        --method "$METHOD" \
        "$KNOB_FLAG" "$knob_value" \
        --datasets arc math_qa openbookqa \
        --eval-batch-size "$EVAL_BATCH_SIZE" \
        --generation-max-tokens "$GENERATION_MAX_TOKENS" \
        --dataset-hub huggingface \
        --output-root "$WORKSPACE_ROOT/results" \
        "${extra_args[@]}"

    build_method="$METHOD"
    build_knob_args=()
    if [[ "$METHOD" == "naee" ]]; then
        build_method=NAEE
        build_knob_args=(--knob-name beta)
    elif [[ "$METHOD" == "score_only" ]]; then
        build_knob_args=(--knob-name beta)
    elif [[ "$METHOD" == "diep" ]]; then
        build_method=DiEP
    elif [[ "$METHOD" == "modes" ]]; then
        build_method=MoDES
    elif [[ "$METHOD" == "aimer" ]]; then
        build_method=AIMER
    fi
    "$PYTHON_BIN" -m moe_prune.code.src.build_evalscope_results_table \
        --method "$build_method" \
        "${build_knob_args[@]}"
done
