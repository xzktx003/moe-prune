#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY="${MODEL_FAMILY:-qwen3}"
MODEL_PATH="${MODEL_PATH:-${ACE_QWEN3_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"

python -m moe_prune.code.scripts.shared.run_evalscope_eval \
  --model-family "${MODEL_FAMILY}" \
  --model-path "${MODEL_PATH}" \
  --method ace \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
