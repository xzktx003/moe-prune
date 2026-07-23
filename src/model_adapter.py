from __future__ import annotations

import os
from inspect import signature
from contextlib import contextmanager, nullcontext
from typing import Dict, Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .aimer_selector import build_aimer_keep_table_for_model, patch_qwen3_moe_blocks_aimer
from .expert_similarity import build_model_similarity_table
from .expert_sparsity_selector import patch_qwen3_moe_blocks_expert_sparsity
from .model_structure import get_decoder_layers
from .model_families import resolve_model_family
from .runtime_pruner import (
    RuntimeStats,
    build_uniform_tau_by_layer,
    patch_qwen3_moe_blocks,
    patch_qwen3_moe_blocks_dual_view,
)
from .score_only_selector import patch_qwen3_moe_blocks_score_only
from .sere_selector import patch_qwen3_moe_blocks_sere
from .top_p_selector import patch_qwen3_moe_blocks_top_p
from .xshare_selector import patch_qwen3_moe_blocks_xshare


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
def patched_model_for_gsp(
    model,
    amp_table: Dict[int, torch.Tensor],
    tau: float,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    tau_by_layer = build_uniform_tau_by_layer(amp_table.keys(), tau)
    with patch_qwen3_moe_blocks(
        model=model,
        amp_table=amp_table,
        tau_by_layer=tau_by_layer,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
    ):
        yield model


@contextmanager
def patched_model_for_rcr(
    model,
    proto_amp_table: Dict[int, torch.Tensor],
    tau: float,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    tau_by_layer = build_uniform_tau_by_layer(proto_amp_table.keys(), tau)
    with patch_qwen3_moe_blocks(
        model=model,
        amp_table=proto_amp_table,
        tau_by_layer=tau_by_layer,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
    ):
        yield model


@contextmanager
def patched_model_for_ace(
    model,
    slanc_amp_table: Dict[int, torch.Tensor],
    proto_amp_table: Dict[int, torch.Tensor],
    tau: float,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
    min_keep: int = 1,
):
    """Apply ACE using max(1.0 * normalized GSP, 0.1 * normalized RCR)."""
    common_layers = slanc_amp_table.keys() & proto_amp_table.keys()
    tau_by_layer = build_uniform_tau_by_layer(common_layers, tau)
    with patch_qwen3_moe_blocks_dual_view(
        model=model,
        slanc_amp_table=slanc_amp_table,
        proto_amp_table=proto_amp_table,
        tau_by_layer=tau_by_layer,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
        min_keep=min_keep,
    ):
        yield model


@contextmanager
def patched_model_for_score_only(
    model,
    gate_threshold: float,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    gate_threshold_by_layer = build_uniform_tau_by_layer(
        range(len(get_decoder_layers(model))),
        gate_threshold,
    )
    with patch_qwen3_moe_blocks_score_only(
        model=model,
        gate_threshold_by_layer=gate_threshold_by_layer,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
    ):
        yield model


@contextmanager
def patched_model_for_expert_sparsity(
    model,
    beta: float,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    beta_by_layer = build_uniform_tau_by_layer(
        range(len(get_decoder_layers(model))),
        beta,
    )
    with patch_qwen3_moe_blocks_expert_sparsity(
        model=model,
        beta_by_layer=beta_by_layer,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
    ):
        yield model


@contextmanager
def patched_model_for_aimer(
    model,
    keep_table: Dict[int, torch.Tensor],
    tau: float,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    tau_by_layer = build_uniform_tau_by_layer(keep_table.keys(), tau)
    with patch_qwen3_moe_blocks_aimer(
        model=model,
        keep_table=keep_table,
        tau_by_layer=tau_by_layer,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
    ):
        yield model


@contextmanager
def patched_model_for_top_p(
    model,
    tau: float,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    tau_by_layer = build_uniform_tau_by_layer(
        range(len(get_decoder_layers(model))),
        tau,
    )
    with patch_qwen3_moe_blocks_top_p(
        model=model,
        tau_by_layer=tau_by_layer,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
    ):
        yield model


@contextmanager
def patched_model_for_sere(
    model,
    tau: float,
    similarity_mode: str = "fast",
    sim_table: Optional[Dict[int, torch.Tensor]] = None,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    sim_table = sim_table or build_model_similarity_table(model, mode=similarity_mode)
    tau_by_layer = build_uniform_tau_by_layer(
        range(len(get_decoder_layers(model))),
        tau,
    )
    with patch_qwen3_moe_blocks_sere(
        model=model,
        sim_table=sim_table,
        tau_by_layer=tau_by_layer,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
    ):
        yield model


@contextmanager
def patched_model_for_xshare(
    model,
    tau: float,
    runtime_stats: Optional[RuntimeStats] = None,
    moe_backend: str = "triton",
):
    tau_by_layer = build_uniform_tau_by_layer(
        range(len(get_decoder_layers(model))),
        tau,
    )
    with patch_qwen3_moe_blocks_xshare(
        model=model,
        tau_by_layer=tau_by_layer,
        runtime_stats=runtime_stats,
        moe_backend=moe_backend,
    ):
        yield model


def build_aimer_keep_table(model) -> Dict[int, torch.Tensor]:
    return build_aimer_keep_table_for_model(model)
