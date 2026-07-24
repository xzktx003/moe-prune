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
OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE_ROOT/results/method2/$MODEL_TAG}
LAMBDA_PENALTY=${LAMBDA_PENALTY:-0.5}
SIMILARITY_MODE=${SIMILARITY_MODE:-fast}

if [[ $# -eq 0 ]]; then
    set -- 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[method2_ppl_grid] taus=$* gpu=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$PYTHON_BIN" -m moe_prune.code.scripts.shared.ppl_eval \
    --model-family "$MODEL_FAMILY" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --method method2 \
    --lambda-penalty "$LAMBDA_PENALTY" \
    --similarity-mode "$SIMILARITY_MODE" \
    --n-ctx 2048 \
    --n-batch 2048 \
    --taus "$@"
