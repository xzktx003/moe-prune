#!/usr/bin/env bash
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
OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE_ROOT/results/EAT-MOE/$MODEL_TAG/full_eval_nctx2048}

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$REPO_ROOT/ablation/EAT-MOE/qwen3_eat_moe_ablation.py" \
    --model-family "$MODEL_FAMILY" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --tau-grid 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 \
    --min-threshold 0.01 \
    --max-threshold 0.50 \
    --n-ctx 2048 \
    --datasets openbookqa arc_challenge mathqa \
    --eval-batch-size 8
