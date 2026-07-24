from __future__ import annotations

import os
from inspect import signature
from contextlib import contextmanager, nullcontext
from typing import Dict, Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_families import resolve_model_family
from .runtime_pruner import (
    RuntimeStats,
    build_uniform_tau_by_layer,
    patch_moe_blocks_for_ace,
)


def clear_hf_proxy_env() -> None:
    for name in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ.pop(name, None)


def apply_torch_autocast_compat_patch() -> None:
    """
    transformers>=5 may call torch.is_autocast_enabled(device_type=...),
    while older torch builds only accept zero arguments.
    """
    try:
        param_count = len(signature(torch.is_autocast_enabled).parameters)
    except (TypeError, ValueError):
        param_count = 0
    if param_count >= 1:
        return

    original = torch.is_autocast_enabled

    def _compat_is_autocast_enabled(device_type=None):
        return original()

    torch.is_autocast_enabled = _compat_is_autocast_enabled


def load_qwen3_moe(model_path: str, device_map=None, model_family: str | None = None):
    clear_hf_proxy_env()
    apply_torch_autocast_compat_patch()
    family = resolve_model_family(model_path=model_path, model_family=model_family)
    if device_map is None:
        device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    common_kwargs = {
        "torch_dtype": "auto",
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if family == "qwen3":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **common_kwargs,
        )
    elif family == "qwen3.6":
        model_cls = getattr(transformers, "Qwen3_5MoeForConditionalGeneration", None)
        if model_cls is None:
            raise ImportError(
                "model-family=qwen3.6 requires a transformers build that provides "
                "Qwen3_5MoeForConditionalGeneration."
            )
        model = model_cls.from_pretrained(model_path, **common_kwargs)
    elif family == "gemma4":
        model_cls = getattr(transformers, "Gemma4ForConditionalGeneration", None)
        if model_cls is None:
            raise ImportError(
                "model-family=gemma4 requires a transformers build that provides "
                "Gemma4ForConditionalGeneration."
            )
        model = model_cls.from_pretrained(model_path, **common_kwargs)
        # Keep final_logit_softcapping at the model's native value (30.0).
        # Gemma4 was trained with softcapping, which is part of its architecture.
        # PPL values will be higher than models without softcapping (e.g., Qwen3).
        # This is expected behavior, not a bug.
    else:
        raise ValueError(f"Unsupported model family: {family}")
    model.eval()
    return model, tokenizer


@contextmanager
def maybe_bf16_autocast():
    if not torch.cuda.is_available():
        with nullcontext():
            yield
        return
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yield


@contextmanager
def patched_model_for_ace(
    model,
    gsp_amp_table: Dict[int, torch.Tensor],
    rcr_amp_table: Dict[int, torch.Tensor],
    tau: float,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
    min_keep: int = 1,
):
    """Apply ACE using max(1.0 * normalized GSP, 0.1 * normalized RCR)."""
    common_layers = gsp_amp_table.keys() & rcr_amp_table.keys()
    tau_by_layer = build_uniform_tau_by_layer(common_layers, tau)
    with patch_moe_blocks_for_ace(
        model=model,
        gsp_amp_table=gsp_amp_table,
        rcr_amp_table=rcr_amp_table,
        tau_by_layer=tau_by_layer,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
        min_keep=min_keep,
    ):
        yield model
