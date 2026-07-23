from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MoeLayerBinding:
    layer_idx: int
    layer: Any
    patch_target: Any
    router: Any
    experts: Any
    top_k: int
    norm_topk_prob: bool
    kind: str


def get_text_model_root(model) -> Any:
    if hasattr(model, "language_model"):
        return model.language_model
    inner = getattr(model, "model", None)
    if inner is not None:
        if hasattr(inner, "language_model"):
            return inner.language_model
        return inner
    return model


def get_decoder_layers(model) -> Iterable[Any]:
    root = get_text_model_root(model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise AttributeError(f"Unable to find decoder layers on {type(model).__name__}")
    return layers


def get_layer_gamma_weight(layer) -> Any:
    for attr in (
        "pre_feedforward_layernorm_2",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "input_layernorm",
    ):
        norm = getattr(layer, attr, None)
        weight = getattr(norm, "weight", None)
        if weight is not None:
            return weight
    raise AttributeError(f"Unable to find layer norm weight for {type(layer).__name__}")


def iter_moe_layer_bindings(model) -> Iterable[MoeLayerBinding]:
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    for layer_idx, layer in enumerate(get_decoder_layers(model)):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        gate = getattr(mlp, "gate", None)
        if experts is not None:
            yield MoeLayerBinding(
                layer_idx=layer_idx,
                layer=layer,
                patch_target=mlp,
                router=gate,
                experts=experts,
                top_k=int(
                    getattr(
                        mlp,
                        "top_k",
                        getattr(
                            text_config,
                            "num_experts_per_tok",
                            getattr(model_config, "num_experts_per_tok", 2),
                        ),
                    )
                ),
                norm_topk_prob=bool(getattr(mlp, "norm_topk_prob", True)),
                kind="mlp",
            )
            continue

        experts = getattr(layer, "experts", None)
        router = getattr(layer, "router", None)
        if getattr(layer, "enable_moe_block", False) and experts is not None and router is not None:
            layer_config = getattr(layer, "config", None) or text_config or model_config
            yield MoeLayerBinding(
                layer_idx=layer_idx,
                layer=layer,
                patch_target=experts,
                router=router,
                experts=experts,
                top_k=int(
                    getattr(
                        layer_config,
                        "top_k_experts",
                        getattr(text_config, "top_k_experts", getattr(model_config, "top_k_experts", 2)),
                    )
                ),
                norm_topk_prob=True,
                kind="fused_experts",
            )
