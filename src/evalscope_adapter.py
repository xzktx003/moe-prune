"""EvalScope adapter for end-to-end ACE evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo


def _pop(values: Dict[str, Any], key: str, default: Any = None) -> Any:
    return values.pop(key, default)


@register_model_api(name="ace_moe")
def _factory():
    from evalscope.models.modelscope import ModelScopeAPI

    from .amp_proxy import build_amp_table_for_model, build_router_proto_amp_table_for_model
    from .model_adapter import patched_model_for_ace
    from .runtime_pruner import RuntimeStats

    class ACEModelAPI(ModelScopeAPI):
        """Hugging Face MoE model evaluated with ACE runtime pruning."""

        def __init__(self, model_name: str, **kwargs: Any) -> None:
            self._prune_tau = float(_pop(kwargs, "prune_tau"))
            self._router_weight_centering = bool(
                _pop(kwargs, "prune_router_weight_centering", False)
            )
            stats_path = _pop(kwargs, "prune_stats_path", None)
            self._prune_stats_path: Optional[Path] = Path(stats_path) if stats_path else None
            self._runtime_stats = RuntimeStats()

            super().__init__(model_name=model_name, **kwargs)

            self._gsp_amp_table = build_amp_table_for_model(self.model)
            self._rcr_amp_table = build_router_proto_amp_table_for_model(
                self.model,
                center_router_weights=self._router_weight_centering,
            )
            self._flush_runtime_stats()

        def _flush_runtime_stats(self) -> None:
            if self._prune_stats_path is None:
                return
            payload = {
                "method": "ace",
                "tau": self._prune_tau,
                "router_weight_centering": self._router_weight_centering,
                "avg_dynamic_pruning_ratio": float(self._runtime_stats.mean_pruning_ratio()),
                "layers": {
                    str(layer_idx): {
                        "pruning_ratio": float(layer_stats.pruning_ratio),
                        "avg_active_experts": float(layer_stats.avg_active_experts),
                        "total_tokens": int(layer_stats.total_tokens),
                        "total_slots": int(layer_stats.total_slots),
                        "kept_slots": int(layer_stats.kept_slots),
                    }
                    for layer_idx, layer_stats in sorted(self._runtime_stats.layers.items())
                },
            }
            self._prune_stats_path.parent.mkdir(parents=True, exist_ok=True)
            self._prune_stats_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        def generate(  # type: ignore[override]
            self,
            input: List[ChatMessage],
            tools: List[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
        ) -> ModelOutput:
            with patched_model_for_ace(
                self.model,
                gsp_amp_table=self._gsp_amp_table,
                rcr_amp_table=self._rcr_amp_table,
                tau=self._prune_tau,
                runtime_stats=self._runtime_stats,
            ):
                output = super().generate(input, tools, tool_choice, config)
            self._flush_runtime_stats()
            return output

    return ACEModelAPI
