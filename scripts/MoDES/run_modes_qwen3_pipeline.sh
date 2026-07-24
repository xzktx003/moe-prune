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

export PYTHONPATH="$WORKSPACE_ROOT"

exec "$PYTHON_BIN" "$REPO_ROOT/ablation/MoDES/run_qwen3_text_pipeline.py" \
    --name_or_path "$MODEL_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --num_samples 128 \
    --batch_size 1 \
    --max_length 2048 \
    --search_targets 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --grid_num 100 \
    --text_tau_min 0.0 \
    --text_tau_max 0.71 \
    --visual_tau_min 0.0 \
    --visual_tau_max 0.0
