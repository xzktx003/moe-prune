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
    exec "$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_ppl_search \
        --model-family "$model_family" \
        --method "aimer" \
        --dataset wikitext \
        --target-pruning-ratios $TARGETS \
        --search-mode quantile \
        --n-ctx 2048 \
        --n-batch 2048 \
        --skip-completed
fi

extra=()
if [[ -n "${GENERATION_MAX_TOKENS:-}" ]]; then
    extra+=(--generation-max-tokens "$GENERATION_MAX_TOKENS")
fi

exec "$PYTHON_BIN" -m moe_prune.code.scripts.shared.run_dataset_search \
    --model-family "$model_family" \
    --method "aimer" \
    --dataset "$dataset" \
    --target-pruning-ratios $TARGETS \
    --search-mode quantile \
    --skip-completed \
    "${extra[@]}"
