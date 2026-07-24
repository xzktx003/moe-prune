#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-${ACE_QWEN3_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}}
OUTPUT_DIR=${OUTPUT_DIR:-results/EAT-MOE/full_eval_nctx2048}
TAU_GRID=${TAU_GRID:-"0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40"}
EVAL_LIMIT=${EVAL_LIMIT:-}
WIKITEXT_ROW_LIMIT=${WIKITEXT_ROW_LIMIT:-}
N_CTX=${N_CTX:-2048}

cmd=(conda run -n xh2 python -u moe_prune/ablation/EAT-MOE/qwen3_eat_moe_ablation.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --tau-grid ${TAU_GRID} \
  --n-ctx "$N_CTX")

if [[ -n "$EVAL_LIMIT" ]]; then
  cmd+=(--eval-limit "$EVAL_LIMIT")
fi
if [[ -n "$WIKITEXT_ROW_LIMIT" ]]; then
  cmd+=(--wikitext-row-limit "$WIKITEXT_ROW_LIMIT")
fi

"${cmd[@]}"
