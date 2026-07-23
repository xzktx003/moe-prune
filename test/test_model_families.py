from __future__ import annotations

import argparse

from moe_prune.code.src.model_families import (
    detect_model_family_from_path,
    finalize_model_selection,
    normalize_model_family,
    resolve_model_path,
)


def test_normalize_model_family_accepts_aliases() -> None:
    assert normalize_model_family("qwen3.5") == "qwen3.6"
    assert normalize_model_family("gemma-4") == "gemma4"


def test_detect_model_family_from_path_uses_path_name() -> None:
    assert detect_model_family_from_path("/models/Qwen3.6-35B-A3B") == "qwen3.6"
    assert detect_model_family_from_path("/models/gemma-4-26B-A4B-it") == "gemma4"


def test_resolve_model_path_uses_family_default() -> None:
    assert resolve_model_path(model_family="qwen3.6") == "Qwen/Qwen3.5-35B-A3B"
    assert resolve_model_path(model_family="gemma4") == "google/gemma-4-26b-a4b-it"


def test_finalize_model_selection_backfills_default_path() -> None:
    args = argparse.Namespace(model_family="gemma4", model_path=None)

    resolved = finalize_model_selection(args)

    assert resolved.model_family == "gemma4"
    assert resolved.model_path == "google/gemma-4-26b-a4b-it"


def test_finalize_model_selection_infers_family_from_explicit_path() -> None:
    args = argparse.Namespace(model_family=None, model_path="/models/Qwen3.6-35B-A3B")

    resolved = finalize_model_selection(args)

    assert resolved.model_family == "qwen3.6"
    assert resolved.model_path.endswith("Qwen3.6-35B-A3B")
