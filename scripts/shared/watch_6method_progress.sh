#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
MAX_LOOPS=${MAX_LOOPS:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

MATRIX_JSON=${MATRIX_JSON:-$REPO_ROOT/results/final_reports/qwen3moe_method_matrix_partial/6method_matrix.json}
REFRESH_SCRIPT=${REFRESH_SCRIPT:-$SCRIPT_DIR/refresh_6method_stage_outputs.sh}

is_all_complete() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

matrix_json = os.environ["MATRIX_JSON"]
required = {"method1", "method2", "method3", "NAEE", "EAT-MOE", "MoDES"}

if not os.path.exists(matrix_json):
    print("status=missing_matrix")
    sys.exit(1)

with open(matrix_json, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

rows = payload.get("rows") or []
present = {str(row.get("method")) for row in rows}
missing = sorted(required - present)

incomplete = []
for row in rows:
    method = row.get("method")
    if method not in required:
        continue
    ppl = row.get("ppl")
    avg = row.get("avg_accuracy")
    if ppl is None or avg is None:
        incomplete.append(method)

if missing or incomplete:
    print("status=incomplete")
    print("missing_methods=" + ",".join(missing))
    print("incomplete_methods=" + ",".join(sorted(set(incomplete))))
    sys.exit(1)

print("status=complete")
sys.exit(0)
PY
}

loop_index=0
echo "[watch_6method_progress] start interval=${INTERVAL_SECONDS}s max_loops=${MAX_LOOPS} matrix=${MATRIX_JSON}"

while true; do
    loop_index=$((loop_index + 1))
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[watch_6method_progress] loop=${loop_index} ts=${ts} refresh_begin"

    bash "$REFRESH_SCRIPT"

    echo "[watch_6method_progress] loop=${loop_index} refresh_done status_check"
    if MATRIX_JSON="$MATRIX_JSON" is_all_complete; then
        echo "[watch_6method_progress] all methods complete, exit loop=${loop_index}"
        exit 0
    fi

    echo "[watch_6method_progress] running_processes"
    if command -v rg >/dev/null 2>&1; then
        ps -ef | rg -i 'run_.*(method|naee|eat|modes)|evalscope|run_naee_ablation|qwen3_eat_moe_ablation|run_qwen3_text_pipeline' || true
    else
        ps -ef | grep -Ei 'run_.*(method|naee|eat|modes)|evalscope|run_naee_ablation|qwen3_eat_moe_ablation|run_qwen3_text_pipeline' | grep -Ev 'grep|watch_6method_progress' || true
    fi

    if [[ "$MAX_LOOPS" != "0" && "$loop_index" -ge "$MAX_LOOPS" ]]; then
        echo "[watch_6method_progress] reached max_loops=${MAX_LOOPS}, exit"
        exit 0
    fi

    sleep "$INTERVAL_SECONDS"
done
