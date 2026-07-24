#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
WORKSPACE_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../shared/model_family.sh"

PYTHON_BIN=${PYTHON_BIN:-python}
parse_model_family_cli_args "$@"
set -- "${MODEL_CLI_ARGS[@]}"
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL_PATH")}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKSPACE_ROOT/results/MoDES/$MODEL_TAG}

if [[ $# -eq 0 ]]; then
    set -- 0.0 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1
fi

export PYTHONPATH="$WORKSPACE_ROOT"

echo "[modes_ppl_grid] taus=$* gpu=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$PYTHON_BIN" -m moe_prune.code.scripts.MoDES.run_modes_ppl_grid \
    --model-family "$MODEL_FAMILY" \
    --model-path "$MODEL_PATH" \
    --results-root "$OUTPUT_ROOT" \
    --n-ctx 2048 \
    --num-samples 128 \
    --batch-size 1 \
    --max-length 2048 \
    "$@"
