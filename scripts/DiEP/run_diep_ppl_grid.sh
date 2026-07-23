#!/usr/bin/env bash
# DiEP PPL grid worker - assumes the per-layer score file already exists
# (use run_diep_qwen3_ppl_pipeline.sh first). Takes tau values as positional
# arguments and evaluates them one by one, matching the pattern of
# run_naee_ppl_grid.sh.
#
# Usage:
#   bash moe_prune/code/scripts/DiEP/run_diep_ppl_grid.sh 0.0 0.5 1.0 1.5
#   TAU_GRID="0.0 1.0" bash moe_prune/code/scripts/DiEP/run_diep_ppl_grid.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_ROOT=$REPO_ROOT
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../shared/model_family.sh"

PYTHON_BIN=${PYTHON_BIN:-python}
parse_model_family_cli_args "$@"
set -- "${MODEL_CLI_ARGS[@]}"
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL_PATH")}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE_ROOT/results/DiEP/$MODEL_TAG/wikitext_search}
SCORE_CACHE_DIR=${SCORE_CACHE_DIR:-$WORKSPACE_ROOT/results/DiEP/$MODEL_TAG/calibration/score_cache}
SCORE_PATH=${SCORE_PATH:-}
LOG_ROOT=${LOG_ROOT:-$OUTPUT_DIR/logs}
N_CTX=${N_CTX:-2048}
CALIB_SAMPLES=${CALIB_SAMPLES:-128}
CALIB_MAX_LEN=${CALIB_MAX_LEN:-2048}
CALIB_SPLIT=${CALIB_SPLIT:-train}

if [[ $# -eq 0 ]]; then
    if [[ -n "${TAU_GRID:-}" ]]; then
        # shellcheck disable=SC2086
        set -- ${TAU_GRID}
    else
        set -- 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0
    fi
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[diep_ppl_grid] tau_values=$* gpu=${CUDA_VISIBLE_DEVICES:-unset}"
for tau in "$@"; do
    tau_dir=$(printf "tau_%.4f" "$tau")
    log_dir="$LOG_ROOT/$tau_dir"
    log_file="$log_dir/diep_ppl.log"
    mkdir -p "$log_dir"

    {
        echo "[diep_ppl_grid] start tau=$tau tau_dir=$tau_dir gpu=${CUDA_VISIBLE_DEVICES:-unset}"
        date -u +"[diep_ppl_grid] utc_start=%Y-%m-%dT%H:%M:%SZ"
    } | tee -a "$log_file"

    cmd=("$PYTHON_BIN" "$REPO_ROOT/ablation/DiEP/qwen3_diep_ablation.py"
        --model-family "$MODEL_FAMILY"
        --model-path "$MODEL_PATH"
        --output-dir "$OUTPUT_DIR"
        --score-cache-dir "$SCORE_CACHE_DIR"
        --tau-grid "$tau"
        --n-ctx "$N_CTX"
        --calibration-num-samples "$CALIB_SAMPLES"
        --calibration-max-length "$CALIB_MAX_LEN"
        --calibration-split "$CALIB_SPLIT")
    if [[ -n "$SCORE_PATH" ]]; then
        cmd+=(--score-path "$SCORE_PATH")
    fi

    "${cmd[@]}" 2>&1 | tee -a "$log_file"

    {
        date -u +"[diep_ppl_grid] utc_end=%Y-%m-%dT%H:%M:%SZ"
        echo "[diep_ppl_grid] done tau=$tau log=$log_file"
    } | tee -a "$log_file"
done
