#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 {calibrate|ppl|evalscope} [arguments...]" >&2
  exit 2
fi
shift

MODEL_FAMILY="${MODEL_FAMILY:-qwen3}"
MODEL_PATH="${MODEL_PATH:-${ACE_QWEN3_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}}"

case "${MODE}" in
  calibrate)
    python -m moe_prune.code.scripts.shared.quantile_calibration \
      --model-family "${MODEL_FAMILY}" \
      --model-path "${MODEL_PATH}" \
      --n-ctx 2048 \
      --n-batch 2048 \
      --split train \
      --min-text-length 0 \
      --calibration-sequences 128 \
      "$@"
    ;;
  ppl)
    OUTPUT_DIR="${OUTPUT_DIR:-results/ACE/ppl}"
    python -m moe_prune.code.scripts.shared.ppl_eval \
      --model-family "${MODEL_FAMILY}" \
      --model-path "${MODEL_PATH}" \
      --output-dir "${OUTPUT_DIR}" \
      --n-ctx 2048 \
      --n-batch 2048 \
      "$@"
    ;;
  evalscope)
    OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
    python -m moe_prune.code.scripts.shared.run_evalscope_eval \
      --model-family "${MODEL_FAMILY}" \
      --model-path "${MODEL_PATH}" \
      --output-root "${OUTPUT_ROOT}" \
      "$@"
    ;;
  *)
    echo "Unknown mode: ${MODE}. Expected calibrate, ppl, or evalscope." >&2
    exit 2
    ;;
esac