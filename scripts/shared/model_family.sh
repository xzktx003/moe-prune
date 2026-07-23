#!/usr/bin/env bash

normalize_model_family() {
    local raw=${1:-qwen3}
    raw=$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')
    case "$raw" in
        qwen3|qwen3moe|qwen3-moe)
            printf 'qwen3\n'
            ;;
        qwen3.6|qwen3_6|qwen3.5|qwen3_5|qwen3.5-moe|qwen3_5_moe)
            printf 'qwen3.6\n'
            ;;
        gemma4|gemma-4|gemma-4-it|gemma-4-26b-a4b-it)
            printf 'gemma4\n'
            ;;
        *)
            echo "unsupported model family: $1" >&2
            return 1
            ;;
    esac
}

detect_model_family_from_path() {
    local model_path=${1:-}
    local model_name
    model_name=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')
    case "$model_name" in
        *gemma-4*|*gemma4*)
            printf 'gemma4\n'
            return 0
            ;;
        *qwen3.6*|*qwen3.5*|*qwen3_5*)
            printf 'qwen3.6\n'
            return 0
            ;;
        *qwen3*)
            printf 'qwen3\n'
            return 0
            ;;
    esac

    local config_path="$model_path/config.json"
    if [[ ! -f "$config_path" ]]; then
        return 1
    fi
    local detected
    detected=$(
        python - "$config_path" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

model_type = str(payload.get("model_type", "")).lower()
text_model_type = str(payload.get("text_config", {}).get("model_type", "")).lower()
if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} or text_model_type == "qwen3_5_moe_text":
    print("qwen3.6")
elif model_type in {"gemma4", "gemma4_text"} or text_model_type == "gemma4_text":
    print("gemma4")
elif model_type in {"qwen3_moe", "qwen3_moe_text"} or text_model_type == "qwen3_moe_text":
    print("qwen3")
else:
    raise SystemExit(1)
PY
    ) || return 1
    printf '%s\n' "$detected"
}

default_model_path_for_family() {
    local family
    family=$(normalize_model_family "${1:-qwen3}")
    case "$family" in
        qwen3)
            printf '%s\n' "${ACE_QWEN3_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
            ;;
        qwen3.6)
            printf '%s\n' "${ACE_QWEN36_MODEL:-Qwen/Qwen3.5-35B-A3B}"
            ;;
        gemma4)
            printf '%s\n' "${ACE_GEMMA4_MODEL:-google/gemma-4-26b-a4b-it}"
            ;;
    esac
}

parse_model_family_cli_args() {
    MODEL_CLI_ARGS=()

    local family_value=${MODEL_FAMILY:-}
    local path_value=${MODEL_PATH:-}
    local explicit_family=0

    while (($#)); do
        case "$1" in
            --model-family)
                if (($# < 2)); then
                    echo "missing value for --model-family" >&2
                    return 1
                fi
                family_value=$2
                explicit_family=1
                shift 2
                ;;
            --model-family=*)
                family_value=${1#*=}
                explicit_family=1
                shift
                ;;
            --model-path)
                if (($# < 2)); then
                    echo "missing value for --model-path" >&2
                    return 1
                fi
                path_value=$2
                shift 2
                ;;
            --model-path=*)
                path_value=${1#*=}
                shift
                ;;
            --)
                shift
                MODEL_CLI_ARGS+=("$@")
                break
                ;;
            *)
                MODEL_CLI_ARGS+=("$@")
                break
                ;;
        esac
    done

    if [[ -z "$family_value" && -n "$path_value" ]]; then
        family_value=$(detect_model_family_from_path "$path_value" || true)
    fi
    if [[ -z "$family_value" ]]; then
        family_value=qwen3
    fi
    MODEL_FAMILY=$(normalize_model_family "$family_value") || return 1
    if [[ -n "$path_value" ]]; then
        MODEL_PATH=$path_value
    else
        MODEL_PATH=$(default_model_path_for_family "$MODEL_FAMILY") || return 1
    fi
    MODEL_FAMILY_EXPLICIT=$explicit_family
}
