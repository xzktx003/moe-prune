#!/usr/bin/env bash
# Run the full (model × method × dataset × pruning-ratio) search matrix.
#
# `run_full_matrix.sh` is the top-level scheduler. It picks the GPU, log path,
# and execution order, then dispatches each (model, method, dataset) cell to a
# method-specific script under code/scripts/<method>/.
# In parallel mode, each method owns one GPU and advances independently through
# its own model×dataset queue; in serial mode, everything runs on SINGLE_GPU.
# Each (model, method, dataset) combo writes output to
#   results/logs/<method>__<model>__<dataset>.log
#
# Datasets: wikitext aime25 piqa math500 arc-e arc-c gpqa humaneval livecodebench
# Methods : method1 modes method3 score_only naee diep   (mapped to GPUs 0..5; avoid GPU 6/7)
#
# Reuse is automatic: every knob visited is cached under
# results/<method>/<model>/<dataset>_search/by_knob/; previous valid records
# are read before any subprocess is launched.
#
# Usage:
#   bash code/scripts/shared/run_full_matrix.sh                  # default full matrix
#   MODELS='qwen3.6' DATASETS='wikitext aime25 piqa math500' bash ...    # subset
#   EXECUTION_MODE=serial SINGLE_GPU=0 bash code/scripts/shared/run_full_matrix.sh
#       # run all methods sequentially on one GPU
#
# NOTE: A full run across 3 models × 6 methods × 6 datasets × 6 ratios with
# full-dataset evalscope at 8k tokens takes many GPU days. This script is
# meant to be launched with `nohup ... &` and monitored via logs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
WORKSPACE_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/matrix_common.sh"

PYTHON_BIN=${PYTHON_BIN:-python}
MODELS=${MODELS:-"qwen3 qwen3.6 gemma4"}
METHODS=${METHODS:-"method1 method3 score_only naee diep modes"}
DATASETS=${DATASETS:-"wikitext piqa math500 arc-e arc-c gpqa humaneval livecodebench"}
TARGETS=${TARGETS:-"0.1 0.2 0.3 0.4 0.5 0.6"}
MAX_STEPS=${MAX_STEPS:-8}
TOLERANCE=${TOLERANCE:-0.005}
GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-2048}
EXECUTION_MODE=${EXECUTION_MODE:-parallel}
SINGLE_GPU=${SINGLE_GPU:-0}

LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"
export PYTHON_BIN TARGETS MAX_STEPS TOLERANCE GENERATION_MAX_TOKENS REPO_ROOT WORKSPACE_ROOT

case "$EXECUTION_MODE" in
    parallel|serial) ;;
    *)
        echo "[error] EXECUTION_MODE must be parallel or serial: $EXECUTION_MODE" >&2
        exit 1
        ;;
esac

gpu_for_method() {
    case "$1" in
        method1) echo 1 ;;
        method2) echo 0 ;;
        method3) echo 2 ;;
        method4) echo 2 ;;
        naee)    echo 3 ;;
        diep)    echo 4 ;;
        score_only) echo 5 ;;
        modes)   echo 6 ;;
        aimer)   echo 7 ;;
        top_p_aimer) echo 7 ;;
        *) echo 0 ;;
    esac
}

method_script_for() {
    case "$1" in
        method1)
            echo "$REPO_ROOT/code/scripts/method1/run_method1_matrix_cell.sh"
            ;;
        method2)
            echo "$REPO_ROOT/code/scripts/method2/run_method2_matrix_cell.sh"
            ;;
        method3)
            echo "$REPO_ROOT/code/scripts/method3/run_method3_matrix_cell.sh"
            ;;
        method4)
            echo "$REPO_ROOT/code/scripts/method4/run_method4_matrix_cell.sh"
            ;;
        naee)
            echo "$REPO_ROOT/code/scripts/NAEE/run_naee_matrix_cell.sh"
            ;;
        score_only)
            echo "$REPO_ROOT/code/scripts/score_only/run_score_only_matrix_cell.sh"
            ;;
        diep)
            echo "$REPO_ROOT/code/scripts/DiEP/run_diep_matrix_cell.sh"
            ;;
        modes)
            echo "$REPO_ROOT/code/scripts/MoDES/run_modes_matrix_cell.sh"
            ;;
        aimer)
            echo "$REPO_ROOT/code/scripts/AIMER/run_aimer_matrix_cell.sh"
            ;;
        top_p_aimer)
            echo "$REPO_ROOT/code_v2/scripts/run_top_p_aimer_matrix_cell.sh"
            ;;
        *)
            return 1
            ;;
    esac
}

launch_one() {
    local method="$1" model="$2" dataset="$3"
    local gpu="${4:-$(gpu_for_method "$method")}"
    local model_dir
    model_dir=$(matrix_model_tag_for "$model")
    local log="$LOG_DIR/${method}__${model_dir}__${dataset}.log"
    local script_path
    if ! script_path=$(method_script_for "$method"); then
        echo "[skip] unsupported method=$method model=$model dataset=$dataset" | tee -a "$log"
        return 0
    fi
    : >"$log"

    echo "[launch] gpu=$gpu method=$method model=$model dataset=$dataset script=$script_path log=$log"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$script_path" "$model" "$dataset"
    ) >>"$log" 2>&1
}

launch_method_queue() {
    local method="$1"
    local gpu="$2"

    for model in $MODELS; do
        for dataset in $DATASETS; do
            launch_one "$method" "$model" "$dataset" "$gpu"
        done
    done
}

case "$EXECUTION_MODE" in
    serial)
        for method in $METHODS; do
            launch_method_queue "$method" "$SINGLE_GPU"
        done
        ;;
    parallel)
        pids=()
        for method in $METHODS; do
            launch_method_queue "$method" "$(gpu_for_method "$method")" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || true
        done
        ;;
esac

echo "[run_full_matrix] all passes scheduled."
