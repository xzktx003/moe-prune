#!/usr/bin/env bash

SHARED_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "$SHARED_SCRIPT_DIR/../../.." && pwd)}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd -- "$REPO_ROOT/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}

# shellcheck source=/dev/null
source "$SHARED_SCRIPT_DIR/model_family.sh"

export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

matrix_model_family_for() {
    normalize_model_family "${1:-qwen3}"
}

matrix_model_path_for() {
    default_model_path_for_family "$(matrix_model_family_for "$1")"
}

matrix_model_tag_for() {
    basename "$(matrix_model_path_for "$1")"
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

diep_score_path_for() {
    local model="$1"
    local model_tag model_path hash
    model_tag=$(matrix_model_tag_for "$model")
    model_path=$(matrix_model_path_for "$model")
    hash=$(printf '%s' "${model_path}|128|2048|train" | sha1sum | cut -c1-12)
    printf '%s/results/DiEP/%s/calibration/score_cache/diep_score_%s_n128_L2048_train_%s.pkl\n' \
        "$REPO_ROOT" "$model_tag" "$model_tag" "$hash"
}

ensure_diep_score() {
    local model="$1" gpu="$2"
    local score_path
    score_path=$(diep_score_path_for "$model")
    if [[ -f "$score_path" ]]; then
        return 0
    fi
    echo "[diep-calib] generating score file: $score_path"
    mkdir -p "$(dirname "$score_path")"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$PYTHON_BIN" "$REPO_ROOT/ablation/DiEP/qwen3_diep_ablation.py" \
            --model-family "$(matrix_model_family_for "$model")" \
            --model-path "$(matrix_model_path_for "$model")" \
            --output-dir "$REPO_ROOT/results/DiEP/$(matrix_model_tag_for "$model")/calibration" \
            --score-path "$score_path" \
            --tau-grid 0.0 \
            --calibration-num-samples 128 \
            --calibration-max-length 2048 \
            --calibration-split train \
            --skip-eval
    )
}

modes_layer_importance_path_for() {
    local model="$1"
    printf '%s/results/MoDES/calibration/wiki/%s/kl_0_128.pkl\n' "$REPO_ROOT" "$(matrix_model_tag_for "$model")"
}

ensure_modes_calibration() {
    local model="$1" gpu="$2"
    local calibration_path
    calibration_path=$(modes_layer_importance_path_for "$model")
    if modes_calibration_is_valid "$calibration_path"; then
        return 0
    fi
    echo "[modes-calib] generating layer importance: $calibration_path"
    mkdir -p "$(dirname "$calibration_path")"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$PYTHON_BIN" "$REPO_ROOT/ablation/MoDES/get_layer_importance_ddp.py" \
            --name_or_path "$(matrix_model_path_for "$model")" \
            --save_dir "$REPO_ROOT/results/MoDES/calibration" \
            --dataset wiki \
            --loss_type kl \
            --batch_size 1 \
            --num_samples 128 \
            --max_length 2048 \
            --temperature 1.0
    )
    modes_calibration_is_valid "$calibration_path"
}
