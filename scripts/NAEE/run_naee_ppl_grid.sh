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
OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE_ROOT/results/NAEE/$MODEL_TAG}
LOG_ROOT=${LOG_ROOT:-$OUTPUT_DIR/logs}

if [[ $# -eq 0 ]]; then
    set -- 0.0 0.2 0.25 0.3 0.35 0.37 0.4 0.45 0.5
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[naee_ppl_grid] betas=$* gpu=${CUDA_VISIBLE_DEVICES:-unset}"
for beta in "$@"; do
    tau_dir=$(printf "tau_%.4f" "$beta")
    log_dir="$LOG_ROOT/$tau_dir"
    log_file="$log_dir/naee_ppl.log"
    mkdir -p "$log_dir"

    {
        echo "[naee_ppl_grid] start beta=$beta tau_dir=$tau_dir gpu=${CUDA_VISIBLE_DEVICES:-unset}"
        date -u +"[naee_ppl_grid] utc_start=%Y-%m-%dT%H:%M:%SZ"
    } | tee -a "$log_file"

    "$PYTHON_BIN" -m moe_prune.code.scripts.NAEE.run_naee_ablation \
        --model-family "$MODEL_FAMILY" \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --beta-grid "$beta" \
        --skip-calibration \
        --skip-eval \
        --ppl-n-ctx 2048 \
        --ppl-n-batch 2048 2>&1 | tee -a "$log_file"

    {
        date -u +"[naee_ppl_grid] utc_end=%Y-%m-%dT%H:%M:%SZ"
        echo "[naee_ppl_grid] done beta=$beta log=$log_file"
    } | tee -a "$log_file"
done
