#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
WORKSPACE_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${GPU:-7}
POLL_INTERVAL=${POLL_INTERVAL:-60}
TARGETS=${TARGETS:-"0.1 0.2 0.3 0.4 0.5 0.6"}
LOG_DIR=${LOG_DIR:-"$REPO_ROOT/results/logs"}
mkdir -p "$LOG_DIR"
STATE_DIR=${STATE_DIR:-"$REPO_ROOT/results/finalize_state"}
mkdir -p "$STATE_DIR"
STATUS_PATH=${STATUS_PATH:-"$STATE_DIR/status.json"}
MODES_CAL_LOG=${MODES_CAL_LOG:-"$REPO_ROOT/results/logs/modes_calibration_128.log"}
AUDIT_JSON=${AUDIT_JSON:-"$REPO_ROOT/results/audit/search_matrix_audit.json"}

export PYTHONPATH="$WORKSPACE_ROOT"

log() {
    if [[ -t 1 ]]; then
        printf '[finalize-matrix] %s\n' "$*" | tee -a "$LOG_DIR/finalize_matrix_pipeline.log"
    else
        printf '[finalize-matrix] %s\n' "$*" >>"$LOG_DIR/finalize_matrix_pipeline.log"
    fi
}

write_status() {
    local phase="$1"
    local detail="${2:-}"
    local incomplete="${3:-}"
    local progress="${4:-}"
    local eta_seconds="${5:-}"
    local calibration_pid="${6:-}"
    local calibration_running="${7:-}"
    "$PYTHON_BIN" - <<'PY' "$STATUS_PATH" "$phase" "$detail" "$incomplete" "$GPU" "$STATE_DIR" "$POLL_INTERVAL" "$$" "$progress" "$eta_seconds" "$calibration_pid" "$calibration_running" "$AUDIT_JSON"
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

path = Path(sys.argv[1])
phase = sys.argv[2]
detail = sys.argv[3]
incomplete_raw = sys.argv[4]
gpu = sys.argv[5]
state_dir = Path(sys.argv[6])
poll_interval = int(sys.argv[7])
pid = int(sys.argv[8])
progress = sys.argv[9]
eta_seconds_raw = sys.argv[10]
calibration_pid_raw = sys.argv[11]
calibration_running_raw = sys.argv[12]
audit_json = Path(sys.argv[13])
payload = {
    "phase": phase,
    "detail": detail,
    "gpu": gpu,
    "pid": pid,
    "poll_interval_seconds": poll_interval,
    "stages": {
        "modes_verification": (state_dir / "modes_verification.done").exists(),
        "wikitext_remediation": (state_dir / "wikitext_remediation.done").exists(),
        "final_plots": (state_dir / "final_plots.done").exists(),
    },
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
if incomplete_raw:
    payload["incomplete_cells"] = int(incomplete_raw)
elif audit_json.exists():
    try:
        audit_payload = json.loads(audit_json.read_text(encoding="utf-8"))
        payload["incomplete_cells"] = int(audit_payload["summary"]["incomplete_cells"])
    except Exception:
        pass
if progress:
    payload["progress"] = progress
if eta_seconds_raw:
    payload["eta_seconds"] = int(eta_seconds_raw)
if calibration_pid_raw or calibration_running_raw:
    payload["calibration_process"] = {
        "pid": int(calibration_pid_raw) if calibration_pid_raw else None,
        "running": calibration_running_raw == "1",
    }
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

stage_done() {
    local stage="$1"
    [[ -f "$STATE_DIR/${stage}.done" ]]
}

mark_stage_done() {
    local stage="$1"
    : >"$STATE_DIR/${stage}.done"
    write_status "stage_complete" "$stage"
    log "stage complete: $stage"
}

model_dirname_for() {
    case "$1" in
        qwen3) echo "Qwen3-30B-A3B-Instruct-2507" ;;
        qwen3.6|qwen3.5) echo "Qwen3.6-35B-A3B" ;;
        gemma4) echo "gemma-4-26B-A4B-it" ;;
        *) echo "$1" ;;
    esac
}

model_path_for() {
    case "$1" in
        qwen3) echo "${ACE_QWEN3_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}" ;;
        qwen3.6|qwen3.5) echo "${ACE_QWEN36_MODEL:-Qwen/Qwen3.5-35B-A3B}" ;;
        gemma4) echo "${ACE_GEMMA4_MODEL:-google/gemma-4-26b-a4b-it}" ;;
        *) echo "$1" ;;
    esac
}

modes_calibration_is_valid() {
    local path="$1"
    "$PYTHON_BIN" - <<'PY' "$path"
from pathlib import Path
import sys
from moe_prune.code.src.modes_calibration import calibration_file_is_valid

path = Path(sys.argv[1])
raise SystemExit(0 if calibration_file_is_valid(path) else 1)
PY
}

latest_modes_calibration_progress() {
    local line
    if [[ ! -f "$MODES_CAL_LOG" ]]; then
        return 0
    fi
    line=$(grep -E 'Processing batch [0-9]+/[0-9]+' "$MODES_CAL_LOG" | tail -n 1 || true)
    if [[ "$line" =~ Processing\ batch\ ([0-9]+)/([0-9]+) ]]; then
        printf '%s/%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    fi
}

latest_modes_calibration_eta_seconds() {
    if [[ ! -f "$MODES_CAL_LOG" ]]; then
        return 0
    fi
    "$PYTHON_BIN" - <<'PY' "$MODES_CAL_LOG"
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import re
import sys

pattern = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).+Processing batch (?P<idx>\d+)/(?P<total>\d+)')
rows = []
for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines():
    m = pattern.search(line)
    if not m:
        continue
    ts = datetime.strptime(m.group('ts'), '%Y-%m-%d %H:%M:%S.%f')
    idx = int(m.group('idx'))
    total = int(m.group('total'))
    rows.append((ts, idx, total))
if len(rows) < 2:
    raise SystemExit(0)
start_ts, start_idx, _ = rows[0]
end_ts, end_idx, total = rows[-1]
done = end_idx - start_idx
if done <= 0:
    raise SystemExit(0)
elapsed = (end_ts - start_ts).total_seconds()
if elapsed <= 0:
    raise SystemExit(0)
per_batch = elapsed / done
remaining = max(total - end_idx, 0)
print(int(round(per_batch * remaining)))
PY
}

modes_calibration_process_running() {
    local family="$1"
    [[ -n "$(modes_calibration_pid "$family")" ]]
}

modes_calibration_pid() {
    local family="$1"
    local model_path
    model_path=$(model_path_for "$family")
    ps -eo pid=,args= | grep -E "^[[:space:]]*[0-9]+[[:space:]]+python .*get_layer_importance_ddp.py --name_or_path ${model_path}" | grep -v grep | awk 'NR==1{print $1}'
}

launch_modes_calibration_if_needed() {
    local family="$1"
    local path="$2"
    if modes_calibration_is_valid "$path"; then
        return 0
    fi
    if modes_calibration_process_running "$family"; then
        return 0
    fi
    write_status "launching_missing_calibration" "$family"
    log "launching missing calibration for $family -> $path"
    printf '[finalize-matrix] relaunch calibration for %s on gpu=%s\n' "$family" "$GPU" >>"$MODES_CAL_LOG"
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$REPO_ROOT/ablation/MoDES/get_layer_importance_ddp.py" \
        --name_or_path "$(model_path_for "$family")" \
        --save_dir "$REPO_ROOT/results/MoDES/calibration" \
        --dataset wiki \
        --loss_type kl \
        --batch_size 1 \
        --num_samples 128 \
        --max_length 2048 \
        --temperature 1.0 >>"$MODES_CAL_LOG" 2>&1 &
}

wait_for_valid_modes_calibration() {
    local family="$1"
    local path="$2"
    while ! modes_calibration_is_valid "$path"; do
        local progress
        local eta_seconds
        local calibration_pid
        launch_modes_calibration_if_needed "$family" "$path"
        progress=$(latest_modes_calibration_progress)
        eta_seconds=$(latest_modes_calibration_eta_seconds)
        calibration_pid=$(modes_calibration_pid "$family")
        write_status "waiting_for_valid_calibration" "$path" "" "$progress" "$eta_seconds" "$calibration_pid" "1"
        log "waiting for valid calibration $path"
        sleep "$POLL_INTERVAL"
    done
}

run_once_if_missing() {
    local sentinel="$1"
    shift
    if valid_sentinel "$sentinel"; then
        write_status "reusing_sentinel" "$sentinel"
        log "reuse existing sentinel $sentinel"
        return 0
    fi
    write_status "running_command" "$*"
    log "running: $*"
    "$@"
}

valid_sentinel() {
    local sentinel="$1"
    [[ -s "$sentinel" ]] || return 1
    python - "$sentinel" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
suffix = path.suffix.lower()
try:
    text = path.read_text(encoding="utf-8").strip()
except Exception:
    raise SystemExit(1)
if not text:
    raise SystemExit(1)
if suffix == ".json":
    try:
        payload = json.loads(text)
    except Exception:
        raise SystemExit(1)
    raise SystemExit(0 if bool(payload) else 1)
if suffix == ".jsonl":
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except Exception:
            raise SystemExit(1)
        raise SystemExit(0)
    raise SystemExit(1)
raise SystemExit(0)
PY
}

archive_incomplete_dir() {
    local sentinel="$1"
    local work_dir="$2"
    if valid_sentinel "$sentinel" || [[ ! -d "$work_dir" ]]; then
        return 0
    fi
    local suffix archive_dir
    suffix=$(date -u +%Y%m%dT%H%M%SZ)
    archive_dir="${work_dir}.stale_${suffix}"
    local counter=1
    while [[ -e "$archive_dir" ]]; do
        archive_dir="${work_dir}.stale_${suffix}_${counter}"
        counter=$((counter + 1))
    done
    mv "$work_dir" "$archive_dir"
    log "archived incomplete work_dir $work_dir -> $archive_dir"
}

wait_for_no_matching_process() {
    local pattern="$1"
    while pgrep -f "$pattern" >/dev/null 2>&1; do
        write_status "waiting_for_process" "$pattern"
        log "waiting for process pattern to clear: $pattern"
        sleep "$POLL_INTERVAL"
    done
}

rerun_audit() {
    "$PYTHON_BIN" -m moe_prune.code.src.audit_search_matrix --results-root "$REPO_ROOT/results" --output-dir "$REPO_ROOT/results/audit" || true
}

audit_incomplete_cells() {
    "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path("results/audit/search_matrix_audit.json")
if not path.exists():
    print(-1)
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(int(payload["summary"]["incomplete_cells"]))
PY
}

run_qwen36_and_gemma4_modes_verifications() {
    if stage_done "modes_verification"; then
        log "reuse completed stage modes_verification"
        return 0
    fi
    local q36_cal="$REPO_ROOT/results/MoDES/calibration/wiki/Qwen3.6-35B-A3B/kl_0_128.pkl"
    local g4_cal="$REPO_ROOT/results/MoDES/calibration/wiki/gemma-4-26B-A4B-it/kl_0_128.pkl"
    local q36_verify_dir="$REPO_ROOT/results/MoDES/verification/Qwen3.6-35B-A3B/gsm8k_limit1_tau_0.0"
    local q36_verify_sentinel="$q36_verify_dir/predictions/Qwen3.6-35B-A3B/gsm8k_main.jsonl"
    local g4_verify_dir="$REPO_ROOT/results/MoDES/verification/gemma-4-26B-A4B-it/gsm8k_limit1_tau_0.0"
    local g4_verify_sentinel="$g4_verify_dir/predictions/gemma-4-26B-A4B-it/gsm8k_main.jsonl"

    wait_for_valid_modes_calibration "qwen3.6" "$q36_cal"
    archive_incomplete_dir "$q36_verify_sentinel" "$q36_verify_dir"
    run_once_if_missing \
        "$q36_verify_sentinel" \
        env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_evalscope_eval \
            --model-family qwen3.6 \
            --method modes \
            --tau 0.0 \
            --datasets gsm8k \
            --limit 1 \
            --eval-batch-size 1 \
            --generation-max-tokens 8192 \
            --layer-importance-path "$q36_cal" \
            --work-dir "$q36_verify_dir"

    wait_for_valid_modes_calibration "gemma4" "$g4_cal"
    archive_incomplete_dir "$g4_verify_sentinel" "$g4_verify_dir"
    run_once_if_missing \
        "$g4_verify_sentinel" \
        env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_evalscope_eval \
            --model-family gemma4 \
            --method modes \
            --tau 0.0 \
            --datasets gsm8k \
            --limit 1 \
            --eval-batch-size 1 \
            --generation-max-tokens 8192 \
            --layer-importance-path "$g4_cal" \
            --work-dir "$g4_verify_dir"
    mark_stage_done "modes_verification"
}

run_wikitext_remediation() {
    if stage_done "wikitext_remediation"; then
        log "reuse completed stage wikitext_remediation"
        return 0
    fi
    wait_for_no_matching_process 'results/MoDES/verification/.*/gsm8k_limit1_tau_0.0'

    run_once_if_missing \
        "$REPO_ROOT/results/method2/Qwen3.6-35B-A3B/wikitext_search/eval/tau_0.618750/ppl_by_tau/tau_0.6188/wikitext_ppl.json" \
        env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_ppl_search \
            --model-family qwen3.6 \
            --method method2 \
            --dataset wikitext \
            --target-pruning-ratios 0.2 0.4 0.5 0.6 \
            --tau-min 0.45 \
            --tau-max 0.675 \
            --max-search-steps 16 \
            --search-tolerance 0.005 \
            --skip-completed
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m moe_prune.code.src.rebuild_selected_targets \
        --search-dir "$REPO_ROOT/results/method2/Qwen3.6-35B-A3B/wikitext_search" \
        --kind ppl \
        --method method2

    run_once_if_missing \
        "$REPO_ROOT/results/method3/Qwen3-30B-A3B-Instruct-2507/wikitext_search/eval/tau_2.000000/ppl_by_tau/tau_2.0000/wikitext_ppl.json" \
        env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_ppl_search \
            --model-family qwen3 \
            --method method3 \
            --dataset wikitext \
            --target-pruning-ratios 0.5 0.6 \
            --tau-min 1.0 \
            --tau-max 3.0 \
            --max-search-steps 16 \
            --search-tolerance 0.005 \
            --skip-completed

    run_once_if_missing \
        "$REPO_ROOT/results/method3/Qwen3.6-35B-A3B/wikitext_search/eval/tau_2.000000/ppl_by_tau/tau_2.0000/wikitext_ppl.json" \
        env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_ppl_search \
            --model-family qwen3.6 \
            --method method3 \
            --dataset wikitext \
            --target-pruning-ratios 0.5 0.6 \
            --tau-min 1.0 \
            --tau-max 3.0 \
            --max-search-steps 16 \
            --search-tolerance 0.005 \
            --skip-completed

    run_once_if_missing \
        "$REPO_ROOT/results/method3/gemma-4-26B-A4B-it/wikitext_search/eval/tau_2.000000/ppl_by_tau/tau_2.0000/wikitext_ppl.json" \
        env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_ppl_search \
            --model-family gemma4 \
            --method method3 \
            --dataset wikitext \
            --target-pruning-ratios 0.6 \
            --tau-min 1.0 \
            --tau-max 3.0 \
            --max-search-steps 16 \
            --search-tolerance 0.005 \
            --skip-completed

    run_once_if_missing \
        "$REPO_ROOT/results/DiEP/gemma-4-26B-A4B-it/ppl_search/eval/tau_1.019531/wikitext_ppl.json" \
        env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m moe_prune.code.scripts.DiEP.run_diep_ppl_search_pruning_rate \
            --model-family gemma4 \
            --results-root "$REPO_ROOT/results/DiEP/gemma-4-26B-A4B-it" \
            --target-pruning-ratios 0.5 \
            --tau-min 1.0 \
            --tau-max 1.03125 \
            --max-search-steps 16 \
            --search-tolerance 0.005 \
            --skip-completed

    for family in qwen3 qwen3.6 gemma4; do
        local model_dir
        model_dir=$(model_dirname_for "$family")
        run_once_if_missing \
            "$REPO_ROOT/results/MoDES/${model_dir}/ppl_search/eval/tau_1.000000/ppl_by_tau/tau_1.0000/wikitext_ppl.json" \
            env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m moe_prune.code.scripts.MoDES.run_modes_ppl_search_pruning_rate \
                --model-family "$family" \
                --results-root "$REPO_ROOT/results/MoDES/${model_dir}" \
                --calibration-root "$REPO_ROOT/results/MoDES" \
                --target-pruning-ratios $TARGETS \
                --tau-min 0.0 \
                --tau-max 2.0 \
                --max-search-steps 16 \
                --search-tolerance 0.005 \
                --skip-completed \
                --skip-calibration
    done
    mark_stage_done "wikitext_remediation"
}

wait_for_full_matrix_and_plot() {
    if stage_done "final_plots"; then
        log "reuse completed stage final_plots"
        return 0
    fi
    while true; do
        rerun_audit
        local incomplete
        incomplete=$(audit_incomplete_cells)
        write_status "audit_poll" "results/audit/search_matrix_audit.json" "$incomplete"
        log "audit incomplete_cells=${incomplete}"
        if [[ "$incomplete" == "0" ]]; then
            write_status "generating_final_plots" "results/plots"
            log "matrix complete; generating final plots"
            "$PYTHON_BIN" -m moe_prune.code.src.plot_search_curves --results-root "$REPO_ROOT/results" --output-dir "$REPO_ROOT/results/plots" --layout dataset
            mark_stage_done "final_plots"
            return 0
        fi
        sleep "$POLL_INTERVAL"
    done
}

main() {
    cd "$REPO_ROOT"
    write_status "starting" "gpu=$GPU"
    log "starting finalization pipeline on gpu=$GPU"
    run_qwen36_and_gemma4_modes_verifications
    run_wikitext_remediation
    wait_for_full_matrix_and_plot
}

main "$@"
