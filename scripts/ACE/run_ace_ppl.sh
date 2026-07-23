#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY="${MODEL_FAMILY:-qwen3}"
MODEL_PATH="${MODEL_PATH:-${ACE_QWEN3_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ACE/ppl}"

python -m moe_prune.code.scripts.shared.ppl_eval \
  --model-family "${MODEL_FAMILY}" \
  --model-path "${MODEL_PATH}" \
  --method ace \
  --output-dir "${OUTPUT_DIR}" \
  --n-ctx 2048 \
  --n-batch 2048 \
  "$@"
