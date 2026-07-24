#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
MAX_LOOPS=${MAX_LOOPS:-0}
GPU_ALLOWLIST=${GPU_ALLOWLIST:-0,1,2,3,4,5}
PYTHON_BIN=${PYTHON_BIN:-python}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/results/final_reports/auto_drive_logs}
MATRIX_JSON=${MATRIX_JSON:-$REPO_ROOT/results/final_reports/qwen3moe_method_matrix_partial/6method_matrix.json}

mkdir -p "$LOG_DIR"

is_all_complete() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

matrix_json = os.environ["MATRIX_JSON"]
required = {"method1", "method2", "method3", "NAEE", "EAT-MOE", "MoDES"}

if not os.path.exists(matrix_json):
    sys.exit(1)

with open(matrix_json, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

rows = payload.get("rows") or []
present = {str(row.get("method")) for row in rows}
if required - present:
    sys.exit(1)

for row in rows:
    method = row.get("method")
    if method not in required:
        continue
    if row.get("ppl") is None or row.get("avg_accuracy") is None:
        sys.exit(1)

sys.exit(0)
PY
}

find_free_gpu() {
    local allowlist_csv=$1
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F',' -v allowlist="$allowlist_csv" '
        BEGIN {
            n = split(allowlist, arr, ",")
            for (i = 1; i <= n; i++) {
                gsub(/ /, "", arr[i])
                allow[arr[i]] = 1
            }
        }
        {
            idx = $1; util = $2; mem = $3
            gsub(/ /, "", idx)
            gsub(/ /, "", util)
            gsub(/ /, "", mem)
            if (allow[idx] && util + 0 < 10 && mem + 0 < 2000) {
                print idx
                exit
            }
        }
    '
}

is_running() {
    local pattern=$1
    ps -ef | grep -E "$pattern" | grep -Ev 'grep|auto_drive_6method_plan|watch_6method_progress' >/dev/null 2>&1
}

launch_task() {
    local gpu=$1
    local name=$2
    local command=$3
    local log_file="$LOG_DIR/${name}.log"

    echo "[auto_drive] launch name=${name} gpu=${gpu} log=${log_file}"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" bash -lc "cd '$REPO_ROOT' && $command" >"$log_file" 2>&1 &
    echo "[auto_drive] launched name=${name} pid=$!"
}

try_launch_pending() {
    local free_gpu=$1
    if [[ -z "$free_gpu" ]]; then
        return
    fi

    if ! is_running 'run_method3_evalscope_worker\.sh|run_evalscope_eval.*--method method3'; then
        launch_task "$free_gpu" "method3_evalscope_full" "SKIP_COMPLETED=1 bash code/scripts/method3/run_method3_evalscope_worker.sh 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
        return
    fi

    if ! is_running 'run_method2_jump_evalscope_worker\.sh|run_evalscope_eval.*--method method2'; then
        launch_task "$free_gpu" "method2_evalscope_jump" "SKIP_COMPLETED=1 bash code/scripts/method2/run_method2_jump_evalscope_worker.sh"
        return
    fi

    if ! is_running 'run_naee_evalscope_worker\.sh|run_evalscope_eval.*--method naee'; then
        launch_task "$free_gpu" "naee_evalscope_full" "SKIP_COMPLETED=1 bash code/scripts/NAEE/run_naee_evalscope_worker.sh 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
        return
    fi

    if ! is_running 'run_eat_moe_full_grid\.sh|qwen3_eat_moe_ablation\.py'; then
        launch_task "$free_gpu" "eat_moe_full" "bash code/scripts/EAT-MOE/run_eat_moe_full_grid.sh"
        return
    fi

    if ! is_running 'run_modes_qwen3_pipeline\.sh|run_qwen3_text_pipeline\.py'; then
        launch_task "$free_gpu" "modes_full" "bash code/scripts/MoDES/run_modes_qwen3_pipeline.sh"
        return
    fi
}

loop_index=0
echo "[auto_drive] start interval=${INTERVAL_SECONDS}s max_loops=${MAX_LOOPS} allowlist=${GPU_ALLOWLIST}"

while true; do
    loop_index=$((loop_index + 1))
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[auto_drive] loop=${loop_index} ts=${ts} refresh"

    bash "$SCRIPT_DIR/refresh_6method_stage_outputs.sh"

    if MATRIX_JSON="$MATRIX_JSON" is_all_complete; then
        echo "[auto_drive] all methods complete, exit loop=${loop_index}"
        exit 0
    fi

    free_gpu=$(find_free_gpu "$GPU_ALLOWLIST")
    if [[ -n "$free_gpu" ]]; then
        echo "[auto_drive] free_gpu=${free_gpu}"
        try_launch_pending "$free_gpu"
    else
        echo "[auto_drive] no free gpu in allowlist=${GPU_ALLOWLIST}"
    fi

    if [[ "$MAX_LOOPS" != "0" && "$loop_index" -ge "$MAX_LOOPS" ]]; then
        echo "[auto_drive] reached max_loops=${MAX_LOOPS}, exit"
        exit 0
    fi

    sleep "$INTERVAL_SECONDS"
done
