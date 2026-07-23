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
OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE_ROOT/results/score_only}

if [[ $# -eq 0 ]]; then
    if [[ -n "${BETA_GRID:-}" ]]; then
        # shellcheck disable=SC2086
        set -- ${BETA_GRID}
    else
        set -- 0.0 0.05 0.1 0.15 0.2 0.3
    fi
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m moe_prune.code.scripts.NAEE.run_naee_ablation \
    --method score_only \
    --model-family "$MODEL_FAMILY" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --beta-grid "$@" \
    --skip-eval
