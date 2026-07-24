#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../shared/model_family.sh"

parse_model_family_cli_args "$@"
set -- "${MODEL_CLI_ARGS[@]}"
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL_PATH")}
BASELINE_ROOT=${BASELINE_ROOT:-$REPO_ROOT/results}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/method3/${MODEL_TAG}}
TAUS=${TAUS:-"0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"}
LAMBDA_PENALTY=${LAMBDA_PENALTY:-0.5}
GAMMA_KEEP=${GAMMA_KEEP:-0.5}
SIMILARITY_MODE=${SIMILARITY_MODE:-fast}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
CAL_LIMIT=${CAL_LIMIT:-16}
CAL_MAX_LENGTH=${CAL_MAX_LENGTH:-384}
CAL_TOKEN_LIMIT=${CAL_TOKEN_LIMIT:-128}
PPL_N_CTX=${PPL_N_CTX:-512}
PPL_N_BATCH=${PPL_N_BATCH:-512}
GPU_SET=${GPU_SET:-6,7}
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

mkdir -p "${OUTPUT_ROOT}"

echo "[method3] running WikiText PPL sweep on GPUs ${GPU_SET}"
CUDA_VISIBLE_DEVICES="${GPU_SET}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
python -m moe_prune.code.scripts.method3.ppl_eval_method3 \
  --model-family "${MODEL_FAMILY}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_ROOT}/ppl" \
  --taus ${TAUS} \
  --n-ctx "${PPL_N_CTX}" \
  --n-batch "${PPL_N_BATCH}" \
  --lambda-penalty "${LAMBDA_PENALTY}" \
  --gamma-keep "${GAMMA_KEEP}" \
  --similarity-mode "${SIMILARITY_MODE}" \
  > "${OUTPUT_ROOT}/ppl.log" 2>&1

echo "[method3] running zero-shot sweep on GPUs ${GPU_SET}"
CUDA_VISIBLE_DEVICES="${GPU_SET}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
python -m moe_prune.code.scripts.method3.run_experiment_method3 \
  --model-family "${MODEL_FAMILY}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_ROOT}/zeroshot" \
  --tau-grid ${TAUS} \
  --calibration-limit-per-dataset "${CAL_LIMIT}" \
  --calibration-max-length "${CAL_MAX_LENGTH}" \
  --calibration-token-limit "${CAL_TOKEN_LIMIT}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --lambda-penalty "${LAMBDA_PENALTY}" \
  --gamma-keep "${GAMMA_KEEP}" \
  --similarity-mode "${SIMILARITY_MODE}" \
  > "${OUTPUT_ROOT}/zeroshot.log" 2>&1

echo "[method3] collecting aligned Method1/2/3 comparison snapshot"
python -m moe_prune.code.src.collect_method_results \
  --ppl method1="${BASELINE_ROOT}/runs/outputs-ppl-method1-full-grid/wikitext_ppl.json" \
  --ppl method2="${BASELINE_ROOT}/method2_full/outputs-ppl-method2-stable/wikitext_ppl.json" \
  --ppl method3="${OUTPUT_ROOT}/ppl/wikitext_ppl.json" \
  --zeroshot method1="${BASELINE_ROOT}/runs/outputs-full-method1-grid/results_table.json" \
  --zeroshot method2="${BASELINE_ROOT}/method2_full/outputs-full-method2-grid-live/results_table.json" \
  --zeroshot method3="${OUTPUT_ROOT}/zeroshot/results_table.json" \
  --tau-grid ${TAUS} \
  --output-dir "${OUTPUT_ROOT}/comparison"

echo "[method3] done"
