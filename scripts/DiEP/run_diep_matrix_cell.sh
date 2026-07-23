#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../shared/matrix_common.sh"

if [[ $# -ne 2 ]]; then
    echo "usage: CUDA_VISIBLE_DEVICES=<gpu> $0 <model-family> <dataset>" >&2
    exit 1
fi

model=$1
dataset=$2
model_family=$(matrix_model_family_for "$model")

if [[ "$dataset" == "wikitext" ]]; then
    exec "$PYTHON_BIN" -m moe_prune.code.scripts.DiEP.run_diep_ppl_search_pruning_rate \
        --model-family "$model_family" \
        --results-root "$REPO_ROOT/results/DiEP/$(matrix_model_tag_for "$model")" \
        --target-pruning-ratios $TARGETS \
        --search-tolerance "$TOLERANCE" \
        --max-search-steps "$MAX_STEPS" \
        --skip-completed
fi

gpu=${CUDA_VISIBLE_DEVICES:-0}
score_path=$(diep_score_path_for "$model")
if [[ ! -f "$score_path" ]]; then
    ensure_diep_score "$model" "$gpu"
fi

extra=(--score-path "$score_path")
if [[ -n "${GENERATION_MAX_TOKENS:-}" ]]; then
    extra+=(--generation-max-tokens "$GENERATION_MAX_TOKENS")
fi

exec "$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_dataset_search \
    --model-family "$model_family" \
    --method "diep" \
    --dataset "$dataset" \
    --target-pruning-ratios $TARGETS \
    --search-mode binary \
    --max-search-steps "$MAX_STEPS" \
    --search-tolerance "$TOLERANCE" \
    --skip-completed \
    "${extra[@]}"
