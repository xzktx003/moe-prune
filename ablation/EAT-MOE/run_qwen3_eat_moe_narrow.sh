#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-${ACE_QWEN3_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}}
OUTPUT_DIR=${OUTPUT_DIR:-results/EAT-MOE/preflight_narrow}
TAU_GRID=${TAU_GRID:-"0.05 0.10 0.15"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
EVAL_LIMIT=${EVAL_LIMIT:-16}
WIKITEXT_ROW_LIMIT=${WIKITEXT_ROW_LIMIT:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL:-30}
N_CTX=${N_CTX:-2048}

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1} \
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
conda run -n xh2 python -u moe_prune/ablation/EAT-MOE/qwen3_eat_moe_ablation.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --tau-grid ${TAU_GRID} \
  --n-ctx "$N_CTX" \
  --eval-limit "$EVAL_LIMIT" \
  --wikitext-row-limit "$WIKITEXT_ROW_LIMIT" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  > "$OUTPUT_DIR/run.log" 2>&1 &

RUN_PID=$!
echo "$RUN_PID" | tee "$OUTPUT_DIR/run.pid"

nohup bash -lc "cd $(pwd) && while [ -d /proc/$RUN_PID ]; do conda run -n xh2 python moe_prune/ablation/EAT-MOE/write_run_heartbeat.py --pid $RUN_PID --output-dir $OUTPUT_DIR --once; sleep $HEARTBEAT_INTERVAL; done" \
  > "$OUTPUT_DIR/heartbeat.log" 2>&1 &

echo "Started narrow EAT run pid=$RUN_PID output_dir=$OUTPUT_DIR" >&2
wait "$RUN_PID"
