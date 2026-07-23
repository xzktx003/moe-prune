#!/usr/bin/env bash
# DiEP dynamic-skipping PPL pipeline on Qwen3-MoE (WikiText-2 PPL over a tau grid).
#
# Steps:
#   1. Ensure the per-layer DiEP score file exists (calibrate once if missing,
#      reused on subsequent runs).
#   2. Sweep WikiText-2 PPL for the provided tau grid.
#
# Usage (from repo root):
#   bash moe_prune/code/scripts/DiEP/run_diep_qwen3_ppl_pipeline.sh
#   TAU_GRID="0.0 0.5 1.0 1.5 2.0" bash moe_prune/code/scripts/DiEP/run_diep_qwen3_ppl_pipeline.sh
#
# Env overrides:
#   PYTHON_BIN, MODEL_PATH, OUTPUT_DIR, SCORE_PATH, SCORE_CACHE_DIR,
#   TAU_GRID, N_CTX, CALIB_SAMPLES, CALIB_MAX_LEN, CALIB_SPLIT.
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
TAU_GRID=${TAU_GRID:-"0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0"}
N_CTX=${N_CTX:-2048}
CALIB_SAMPLES=${CALIB_SAMPLES:-128}
CALIB_MAX_LEN=${CALIB_MAX_LEN:-2048}
CALIB_SPLIT=${CALIB_SPLIT:-train}

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cmd=("$PYTHON_BIN" "$REPO_ROOT/ablation/DiEP/qwen3_diep_ablation.py"
    --model-family "$MODEL_FAMILY"
    --model-path "$MODEL_PATH"
    --output-dir "$OUTPUT_DIR"
    --score-cache-dir "$SCORE_CACHE_DIR"
    --tau-grid ${TAU_GRID}
    --n-ctx "$N_CTX"
    --calibration-num-samples "$CALIB_SAMPLES"
    --calibration-max-length "$CALIB_MAX_LEN"
    --calibration-split "$CALIB_SPLIT")

if [[ -n "$SCORE_PATH" ]]; then
    cmd+=(--score-path "$SCORE_PATH")
fi

echo "[diep_ppl_pipeline] ${cmd[*]}"
exec "${cmd[@]}"
