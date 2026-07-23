from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

MODEL_FAMILY_ALIASES = {
    "qwen3": "qwen3",
    "qwen3moe": "qwen3",
    "qwen3-moe": "qwen3",
    "qwen3.6": "qwen3.6",
    "qwen3_6": "qwen3.6",
    "qwen3.5": "qwen3.6",
    "qwen3_5": "qwen3.6",
    "qwen3.5-moe": "qwen3.6",
    "qwen3_5_moe": "qwen3.6",
    "gemma4": "gemma4",
    "gemma-4": "gemma4",
    "gemma-4-it": "gemma4",
    "gemma-4-26b-a4b-it": "gemma4",
}

DEFAULT_MODEL_PATHS = {
    "qwen3": os.environ.get("ACE_QWEN3_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    "qwen3.6": os.environ.get("ACE_QWEN36_MODEL", "Qwen/Qwen3.5-35B-A3B"),
    "gemma4": os.environ.get("ACE_GEMMA4_MODEL", "google/gemma-4-26b-a4b-it"),
}


def normalize_model_family(model_family: str | None, default: str = "qwen3") -> str:
    raw = default if model_family is None else str(model_family).strip()
    if not raw:
        raw = default
    alias = MODEL_FAMILY_ALIASES.get(raw.lower())
    if alias is None:
        raise ValueError(
            f"Unsupported model family {model_family!r}. "
            f"Expected one of {sorted(DEFAULT_MODEL_PATHS)} or a supported alias."
        )
    return alias


def default_model_path_for_family(model_family: str) -> str:
    family = normalize_model_family(model_family)
    return DEFAULT_MODEL_PATHS[family]


def _load_raw_config_json(model_path: str | None) -> dict[str, Any] | None:
    if not model_path:
        return None
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def detect_model_family_from_path(model_path: str | None) -> str | None:
    if not model_path:
        return None
    model_name = Path(model_path).name.lower()
    if "gemma-4" in model_name or "gemma4" in model_name:
        return "gemma4"
    if "qwen3.6" in model_name or "qwen3.5" in model_name or "qwen3_5" in model_name:
        return "qwen3.6"
    if "qwen3" in model_name:
        return "qwen3"

    payload = _load_raw_config_json(model_path)
    if payload is None:
        return None

    model_type = str(payload.get("model_type", "")).lower()
    text_model_type = str(payload.get("text_config", {}).get("model_type", "")).lower()
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} or text_model_type == "qwen3_5_moe_text":
        return "qwen3.6"
    if model_type in {"gemma4", "gemma4_text"} or text_model_type == "gemma4_text":
        return "gemma4"
    if model_type in {"qwen3_moe", "qwen3_moe_text"} or text_model_type == "qwen3_moe_text":
        return "qwen3"
    return None


def resolve_model_family(
    *,
    model_path: str | None = None,
    model_family: str | None = None,
    default: str = "qwen3",
) -> str:
    if model_family is not None:
        return normalize_model_family(model_family, default=default)
    detected = detect_model_family_from_path(model_path)
    if detected is not None:
        return detected
    return normalize_model_family(default, default=default)


def resolve_model_path(
    *,
    model_path: str | None = None,
    model_family: str | None = None,
    default: str = "qwen3",
) -> str:
    if model_path:
        return str(model_path)
    family = resolve_model_family(model_path=model_path, model_family=model_family, default=default)
    return default_model_path_for_family(family)


def add_model_selection_args(
    parser: argparse.ArgumentParser,
    *,
    default_family: str = "qwen3",
    required: bool = False,
) -> None:
    default_family = normalize_model_family(default_family)
    parser.add_argument(
        "--model-family",
        type=str,
        default=None,
        help=(
            "Model family selector. Canonical values: "
            f"{', '.join(sorted(DEFAULT_MODEL_PATHS))}. "
            f"Also accepts aliases like qwen3.5 -> qwen3.6. "
            f"If omitted, infer from --model-path before falling back to {default_family}."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=required,
        default=None,
        help=(
            "Local model path or Hugging Face model ID. If omitted, a public model ID "
            "is resolved from --model-family and can be overridden with ACE_*_MODEL variables."
        ),
    )


def finalize_model_selection(
    args: argparse.Namespace,
    *,
    model_path_attr: str = "model_path",
    model_family_attr: str = "model_family",
    default_family: str = "qwen3",
) -> argparse.Namespace:
    model_path = getattr(args, model_path_attr, None)
    model_family = getattr(args, model_family_attr, None)
    resolved_family = resolve_model_family(
        model_path=model_path,
        model_family=model_family,
        default=default_family,
    )
    resolved_path = resolve_model_path(
        model_path=model_path,
        model_family=resolved_family,
        default=default_family,
    )
    setattr(args, model_family_attr, resolved_family)
    setattr(args, model_path_attr, resolved_path)
    return args
