#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
PARTIAL_MATRIX_DIR=${PARTIAL_MATRIX_DIR:-results/final_reports/qwen3moe_method_matrix_partial}
PARTIAL_REPORT_PATH=${PARTIAL_REPORT_PATH:-docs/prd/6method_final_report.md}

export PYTHONPATH="$WORKSPACE_ROOT"

rebuild_evalscope_table_if_present() {
    local method_name=$1
    local evalscope_dir=$2
    if [[ -d "$evalscope_dir" ]]; then
        echo "[refresh_6method_stage_outputs] rebuild evalscope table method=$method_name"
        "$PYTHON_BIN" -m moe_prune.code.src.build_evalscope_results_table --method "$method_name"
    else
        echo "[refresh_6method_stage_outputs] skip evalscope table method=$method_name (missing $evalscope_dir)"
    fi
}

cd "$REPO_ROOT"

rebuild_evalscope_table_if_present method1 "$REPO_ROOT/results/method1/evalscope"
rebuild_evalscope_table_if_present method2 "$REPO_ROOT/results/method2/evalscope"
rebuild_evalscope_table_if_present method3 "$REPO_ROOT/results/method3/evalscope"
rebuild_evalscope_table_if_present NAEE "$REPO_ROOT/results/NAEE/evalscope"

echo "[refresh_6method_stage_outputs] rebuild partial matrix"
"$PYTHON_BIN" -m moe_prune.code.src.build_qwen3moe_full_matrix \
    --repo-root "$REPO_ROOT" \
    --allow-partial \
    --output-dir "$PARTIAL_MATRIX_DIR"

echo "[refresh_6method_stage_outputs] rebuild stage report"
"$PYTHON_BIN" -m moe_prune.code.src.build_6method_final_report \
    --repo-root "$REPO_ROOT" \
    --matrix-json "$PARTIAL_MATRIX_DIR/6method_matrix.json" \
    --output "$PARTIAL_REPORT_PATH"

echo "[refresh_6method_stage_outputs] done"
