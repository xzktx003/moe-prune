#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../shared/model_family.sh"

parse_model_family_cli_args "$@"
set -- "${MODEL_CLI_ARGS[@]}"
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL_PATH")}
OUTPUT_ROOT=${OUTPUT_ROOT:-moe_prune/results/score_only_ppl_ablation/${MODEL_TAG}}
THRESHOLDS=${THRESHOLDS:-"0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40"}
GPU=${GPU:-7}
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

mkdir -p "${OUTPUT_ROOT}"

run_single() {
  local gpu="$1"
  shift
  local thresholds=("$@")
  local run_dir="${OUTPUT_ROOT}/gpu${gpu}"

  mkdir -p "${run_dir}"
  echo "[score-only] GPU ${gpu} thresholds: ${thresholds[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
    python -m moe_prune.code.scripts.shared.ppl_eval \
    --model-family "${MODEL_FAMILY}" \
    --model-path "${MODEL_PATH}" \
    --method score_only \
    --output-dir "${run_dir}" \
    --taus "${thresholds[@]}" \
    > "${run_dir}/run.log" 2>&1
}

read -r -a threshold_values <<< "${THRESHOLDS}"
run_single "${GPU}" "${threshold_values[@]}"

OUTPUT_ROOT="${OUTPUT_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT"])
rows = []
artifact_rows = []
for split_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
    json_path = split_dir / "wikitext_ppl.json"
    if not json_path.exists():
        continue
    payload = json.loads(json_path.read_text())
    rows.extend(payload)
    artifact_rows.append(
        {
            "split": split_dir.name,
            "json": str(json_path),
            "markdown": str(split_dir / "wikitext_ppl.md"),
            "log": str(split_dir / "run.log"),
        }
    )

rows.sort(key=lambda row: row["tau"])
summary_json = output_root / "summary_table.json"
summary_md = output_root / "summary_table.md"
artifact_json = output_root / "artifact_paths.json"

summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
artifact_json.write_text(
    json.dumps(artifact_rows, indent=2),
    encoding="utf-8",
)

lines = [
    "| tau | ppl | active_expert_pruning_ratio | artifact_json | artifact_log |",
    "| --- | --- | --- | --- | --- |",
]
artifact_lookup = {}
for item in artifact_rows:
    split_payload = json.loads(Path(item["json"]).read_text())
    for row in split_payload:
        artifact_lookup[float(row["tau"])] = item

for row in rows:
    artifact = artifact_lookup[float(row["tau"])]
    lines.append(
        "| {tau:.2f} | {ppl:.4f} | {ratio:.4f} | {json_path} | {log_path} |".format(
            tau=row["tau"],
            ppl=row["ppl"],
            ratio=row["active_expert_pruning_ratio"],
            json_path=artifact["json"],
            log_path=artifact["log"],
        )
    )

summary_md.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {summary_json}")
print(f"wrote {summary_md}")
print(f"wrote {artifact_json}")
PY
