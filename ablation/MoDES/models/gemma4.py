import pickle
from typing import Optional
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from MoDES.models.utils import apply_scaler_scale
except ModuleNotFoundError:
    try:
        from models.utils import apply_scaler_scale
    except ModuleNotFoundError:
        from .utils import apply_scaler_scale

THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from moe_prune.code.src.runtime_pruner import compute_moe_weighted_hidden_states

SKIP_EXP_COUNT = 0
TOTAL_EXP_COUNT = 0


def experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    moe_backend = getattr(self, "_moe_backend", "triton")
    final_hidden_states, _, _ = compute_moe_weighted_hidden_states(
        hidden_states,
        self,
        top_k_index,
        top_k_weights,
        keep_mask=top_k_weights.gt(0),
        moe_backend=moe_backend,
    )
    return final_hidden_states


def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    per_layer_input: torch.Tensor = None,
    shared_kv_states=None,
    position_embeddings: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    **kwargs,
) -> torch.Tensor:
    global SKIP_EXP_COUNT, TOTAL_EXP_COUNT

    moe_layer_skip = kwargs.pop("moe_layer_skip", None)
    skip_modality = kwargs.pop("skip_modality", None)
    enable_tau_skip = bool(kwargs.pop("enable_tau_skip", False))
    tau = kwargs.pop("tau", None) or {"text": 0.0, "visual": 0.0}
    kwargs.pop("enable_load_layer_importance", None)
    kwargs.pop("layer_importance_path", None)

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        shared_kv_states=shared_kv_states,
        position_ids=position_ids,
        past_key_values=past_key_values,
        **kwargs,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)

    if self.enable_moe_block:
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        _, top_k_weights, top_k_index = self.router(hidden_states_flat)
        top_k_weights = top_k_weights.clone()

        text_mask = getattr(self, "_modes_moe_text_mask", None)
        padding_mask = getattr(self, "_modes_moe_padding_mask", None)
        if text_mask is not None:
            active_mask = text_mask.squeeze(-1).to(hidden_states_flat.device)
        else:
            active_mask = torch.ones(hidden_states_flat.shape[0], dtype=torch.bool, device=hidden_states_flat.device)
        if padding_mask is not None:
            active_mask = active_mask & (~padding_mask.squeeze(-1).to(hidden_states_flat.device))

        inactive_mask = ~active_mask
        if inactive_mask.any():
            top_k_weights[inactive_mask] = 0.0

        top_k = top_k_weights.shape[-1]
        TOTAL_EXP_COUNT += int(active_mask.sum().item()) * top_k

        if moe_layer_skip is not None and self.layer_idx == moe_layer_skip and skip_modality == "text":
            if active_mask.any():
                top_k_weights[active_mask] = 0.0
        elif enable_tau_skip and active_mask.any():
            new_top_k_weights = top_k_weights.clone()
            text_layer_importance = getattr(self, "_modes_text_layer_importance", 1.0)
            new_top_k_weights = apply_scaler_scale(
                text_layer_importance,
                new_top_k_weights,
                active_mask[:, None],
            )
            to_zero = active_mask[:, None] & (new_top_k_weights < float(tau["text"]))
            top_k_weights.masked_fill_(to_zero, 0.0)

        SKIP_EXP_COUNT += int(active_mask.sum().item()) * top_k - int(top_k_weights[active_mask].gt(0).sum().item())

        hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
        hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        hidden_states = hidden_states_1 + hidden_states_2

    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    if self.hidden_size_per_layer_input:
        residual = hidden_states
        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states *= self.layer_scalar
    return hidden_states


def text_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    per_layer_inputs: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    moe_layer_skip: Optional[int] = None,
    skip_modality: Optional[str] = None,
    enable_tau_skip: Optional[bool] = False,
    tau: Optional[dict] = None,
    enable_load_layer_importance: Optional[bool] = False,
    layer_importance_path: Optional[str] = None,
    **kwargs,
):
    if not enable_tau_skip:
        enable_tau_skip = bool(getattr(self, "_modes_force_enable_tau_skip", False))
    if tau is None:
        tau = getattr(self, "_modes_force_tau", None)
    if not enable_load_layer_importance:
        enable_load_layer_importance = bool(
            getattr(self, "_modes_force_enable_load_layer_importance", False)
        )
    if layer_importance_path is None:
        layer_importance_path = getattr(self, "_modes_force_layer_importance_path", None)

    text_mask = None
    padding_mask = None
    if enable_tau_skip or moe_layer_skip is not None:
        if input_ids is not None:
            text_mask = torch.ones_like(input_ids, dtype=torch.bool)
            if hasattr(self, "special_token_id_tensor"):
                text_mask = ~torch.isin(input_ids, self.special_token_id_tensor.to(input_ids.device))
            text_mask = text_mask.view(-1)
            decode_stage = input_ids.shape[-1] == 1
        else:
            if inputs_embeds is None:
                raise ValueError("Both input_ids and inputs_embeds cannot be None")
            token_num = inputs_embeds.shape[0] * inputs_embeds.shape[1]
            text_mask = torch.ones(token_num, dtype=torch.bool, device=inputs_embeds.device)
            decode_stage = inputs_embeds.shape[1] == 1
        if (
            attention_mask is not None
            and not isinstance(attention_mask, dict)
            and attention_mask.ndim == 2
            and not decode_stage
            and attention_mask.numel() == text_mask.numel()
        ):
            padding_mask = (~attention_mask.to(torch.bool)).view(-1)

    if enable_load_layer_importance and layer_importance_path is not None and not hasattr(self, "_modes_layer_importance"):
        with open(layer_importance_path, "rb") as f:
            self._modes_layer_importance = pickle.load(f)

    for idx, layer in enumerate(self.layers):
        if getattr(layer, "enable_moe_block", False):
            layer._modes_moe_text_mask = text_mask[:, None] if text_mask is not None else None
            layer._modes_moe_padding_mask = padding_mask[:, None] if padding_mask is not None else None
            layer._modes_text_layer_importance = 1.0
            if hasattr(self, "_modes_layer_importance"):
                layer_payload = self._modes_layer_importance.get(idx, {})
                layer._modes_text_layer_importance = float(layer_payload.get("text", 1.0))

    return self._modes_original_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        use_cache=use_cache,
        moe_layer_skip=moe_layer_skip,
        skip_modality=skip_modality,
        enable_tau_skip=enable_tau_skip,
        tau=tau,
        enable_load_layer_importance=enable_load_layer_importance,
        layer_importance_path=layer_importance_path,
        **kwargs,
    )


def model_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    input_features: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    input_features_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    image_position_ids: torch.LongTensor | None = None,
    video_position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    mm_token_type_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    moe_layer_skip: Optional[int] = None,
    skip_modality: Optional[str] = None,
    enable_tau_skip: Optional[bool] = False,
    tau: Optional[dict] = None,
    enable_load_layer_importance: Optional[bool] = False,
    layer_importance_path: Optional[str] = None,
    **kwargs,
):
    if not enable_tau_skip:
        enable_tau_skip = bool(
            getattr(
                self,
                "_modes_force_enable_tau_skip",
                getattr(self.model.language_model, "_modes_force_enable_tau_skip", False),
            )
        )
    if tau is None:
        tau = getattr(self, "_modes_force_tau", getattr(self.model.language_model, "_modes_force_tau", None))
    if not enable_load_layer_importance:
        enable_load_layer_importance = bool(
            getattr(
                self,
                "_modes_force_enable_load_layer_importance",
                getattr(self.model.language_model, "_modes_force_enable_load_layer_importance", False),
            )
        )
    if layer_importance_path is None:
        layer_importance_path = getattr(
            self,
            "_modes_force_layer_importance_path",
            getattr(self.model.language_model, "_modes_force_layer_importance_path", None),
        )

    return self._modes_original_forward(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        logits_to_keep=logits_to_keep,
        moe_layer_skip=moe_layer_skip,
        skip_modality=skip_modality,
        enable_tau_skip=enable_tau_skip,
        tau=tau,
        enable_load_layer_importance=enable_load_layer_importance,
        layer_importance_path=layer_importance_path,
        **kwargs,
    )


def patch_loaded_model(model, tokenizer) -> None:
    if getattr(model, "_modes_gemma4_patched", False):
        return

    model.eval()
    model._modes_original_forward = model.forward
    model.forward = model_forward.__get__(model)
    model.model.language_model._modes_original_forward = model.model.language_model.forward
    model.model.language_model.forward = text_model_forward.__get__(model.model.language_model)
    model.model.language_model.special_token_id_tensor = torch.tensor(tokenizer.all_special_ids)

    for layer in model.model.language_model.layers:
        if getattr(layer, "enable_moe_block", False):
            layer.forward = decoder_layer_forward.__get__(layer)
            layer.experts.forward = experts_forward.__get__(layer.experts)

    model._modes_gemma4_patched = True


def load_model(
    model_path: str,
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    layer_gate_dict: dict = None,
):
    del layer_gate_dict
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", None)
    if model_type != "gemma4":
        raise ValueError(f"Unsupported model_type={model_type} for {model_path}.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    patch_loaded_model(model, tokenizer)
    return model, tokenizer
