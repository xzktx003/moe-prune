"""EvalScope ModelAPI adapter for Qwen3-MoE runtime pruning.

Registers a custom model API ``qwen3_moe_pruned`` that:

1. Loads Qwen3-MoE once via the standard HuggingFace path.
2. Builds an AMP (expert importance) table from the model weights.
3. At ``generate`` time, wraps ``super().generate`` inside the appropriate
   runtime-pruning context manager, driven by ``model_args`` keys:

     - ``prune_method``: ACE, its GSP/RCR component ablations, or a baseline
         reported in the paper (Score, NAEE, DiEP, MoDES, AIMER,
         ExpertSparsity, Top-P, SERE, or XShare).
   - ``prune_tau``: float (tau-threshold methods)
     - ``prune_similarity_mode``: ``fast`` / ``full`` (SERE only)

All other kwargs fall through to ``AutoModelForCausalLM.from_pretrained``.

Usage:
    import evalscope
    from moe_prune.code.src import evalscope_adapter  # noqa: F401  (side effect)
    from evalscope import TaskConfig, run_task
    run_task(TaskConfig(
        model='Qwen/Qwen3-30B-A3B-Instruct-2507',
        eval_type='qwen3_moe_pruned',
        model_args={'prune_method': 'ace', 'prune_tau': 0.2, 'precision': 'bfloat16'},
        datasets=['arc', 'math_qa', 'openbookqa'],
        dataset_args={'arc': {'subset_list': ['ARC-Challenge']}},
    ))
"""

from __future__ import annotations

import copy
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo


def _pop(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = d.get(key, default)
    if key in d:
        del d[key]
    return value


@register_model_api(name='qwen3_moe_pruned')
def _factory():
    # Lazy import so the registry side-effect works without torch eagerly.
    from evalscope.models.modelscope import ModelScopeAPI

    from .aimer_selector import build_aimer_keep_table_for_model
    from .amp_proxy import build_amp_table_for_model, build_router_proto_amp_table_for_model
    from .expert_similarity import build_model_similarity_table
    from .model_adapter import (
        patched_model_for_ace,
        patched_model_for_gsp,
        patched_model_for_rcr,
        patched_model_for_aimer,
        patched_model_for_expert_sparsity,
        patched_model_for_sere,
        patched_model_for_top_p,
        patched_model_for_xshare,
    )
    from .quantile_collector import QUANTILE_RUNTIME_METHODS, patched_model_for_quantile_collection
    from .runtime_pruner import RuntimeStats
    from ..scripts.NAEE.run_naee_ablation import (
        build_amp_table_for_model as build_naee_amp_table_for_model,
        patched_model_for_naee,
    )
    from .modes_calibration import calibration_file_is_valid, calibration_path

    _REPO_ROOT = Path(__file__).resolve().parents[1]
    _MODES_RESULTS_ROOT = _REPO_ROOT / 'results' / 'MoDES'

    def _load_diep_payload(score_path: str):
        import pickle
        with open(score_path, 'rb') as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict) or 'per_layer_beta' not in payload:
            raise ValueError(f'Invalid DiEP score file: {score_path}')
        return payload

    def _resolve_modes_layer_importance_path(model_path: str, explicit_path: Optional[str]) -> Path:
        if explicit_path:
            resolved = Path(explicit_path)
        else:
            resolved = calibration_path(
                results_root=_MODES_RESULTS_ROOT,
                model_path=model_path,
                loss_type='kl',
                num_samples=128,
            )
        if not calibration_file_is_valid(resolved):
            raise ValueError(
                'prune_method=modes requires a valid MoDES calibration pickle at '
                f'{resolved}'
            )
        return resolved

    def _patch_modes_text_model(model, tokenizer) -> None:
        import sys
        import torch

        modes_root = _REPO_ROOT / 'ablation' / 'MoDES'
        if str(modes_root) not in sys.path:
            sys.path.insert(0, str(modes_root))
        if getattr(model, '_modes_evalscope_patched', False):
            return

        model_type = getattr(model.config, 'model_type', '')
        if model_type in {'qwen3_moe', 'qwen3_5_moe', 'qwen3_5_moe_text'}:
            from models.qwen3 import (  # type: ignore
                _is_sparse_moe_layer,
                decoder_layer_forward,
                experts_forward,
                mlp_forward,
                text_forward,
                text_moe_forward,
            )

            model.eval()
            text_config = model.config
            model.forward = text_forward.__get__(model)
            model.model._modes_original_forward = model.model.forward
            model.model.forward = text_moe_forward.__get__(model.model)
            model.model.special_token_id_tensor = torch.tensor(tokenizer.all_special_ids)

            for idx, layer in enumerate(model.model.layers):
                layer.forward = decoder_layer_forward.__get__(layer)
                if _is_sparse_moe_layer(text_config, idx, layer):
                    layer.mlp.forward = mlp_forward.__get__(layer.mlp)
                    layer.mlp.experts.forward = experts_forward.__get__(layer.mlp.experts)
                layer.mlp.layer_idx = idx
        elif model_type == 'gemma4':
            from models.gemma4 import patch_loaded_model  # type: ignore

            patch_loaded_model(model, tokenizer)
        else:
            raise ValueError(f'MoDES does not support model_type={model_type!r}')

        model._modes_evalscope_patched = True

    def _reset_modes_counters(model_type: str) -> None:
        import sys

        modes_root = _REPO_ROOT / 'ablation' / 'MoDES'
        if str(modes_root) not in sys.path:
            sys.path.insert(0, str(modes_root))
        if model_type == 'gemma4':
            from models import gemma4 as modes_model  # type: ignore
        else:
            from models import qwen3 as modes_model  # type: ignore

        modes_model.SKIP_EXP_COUNT = 0
        modes_model.TOTAL_EXP_COUNT = 0

    def _read_modes_pruning_ratio(model_type: str) -> Optional[float]:
        import sys

        modes_root = _REPO_ROOT / 'ablation' / 'MoDES'
        if str(modes_root) not in sys.path:
            sys.path.insert(0, str(modes_root))
        if model_type == 'gemma4':
            from models import gemma4 as modes_model  # type: ignore
        else:
            from models import qwen3 as modes_model  # type: ignore

        total = float(modes_model.TOTAL_EXP_COUNT)
        if total <= 0:
            return None
        return float(modes_model.SKIP_EXP_COUNT) / total

    def _modes_override_targets(model) -> List[object]:
        targets: List[object] = [model]
        model_type = getattr(model.config, 'model_type', '')
        if model_type == 'gemma4':
            language_model = getattr(getattr(model, 'model', None), 'language_model', None)
            if language_model is not None:
                targets.append(language_model)
        else:
            text_model = getattr(model, 'model', None)
            if text_model is not None:
                targets.append(text_model)
        return targets

    class Qwen3PrunedModelAPI(ModelScopeAPI):
        """Qwen3-MoE checkpoint with runtime expert-pruning context."""

        _SUPPORTED_METHODS = {
            'none',
            'ace',
            'gsp',
            'rcr',
            'naee',
            'score_only',
            'diep',
            'modes',
            'aimer',
            'expert_sparsity',
            'top_p',
            'sere',
            'xshare',
        }

        def __init__(self, model_name: str, **kwargs: Any) -> None:
            # Pop prune-specific kwargs before delegating to base (which
            # passes **kwargs to from_pretrained and would error on unknowns).
            self._prune_method: str = str(_pop(kwargs, 'prune_method', 'none')).lower()
            self._prune_tau: Optional[float] = _pop(kwargs, 'prune_tau', None)
            self._prune_beta: Optional[float] = _pop(kwargs, 'prune_beta', None)
            self._prune_lambda: float = float(_pop(kwargs, 'prune_lambda', 0.5))
            self._prune_gamma: float = float(_pop(kwargs, 'prune_gamma', 0.5))
            self._prune_sim_mode: str = str(_pop(kwargs, 'prune_similarity_mode', 'fast'))
            self._router_weight_centering: bool = bool(
                _pop(kwargs, 'prune_router_weight_centering', False)
            )
            score_path = _pop(kwargs, 'prune_score_path', None)
            self._prune_score_path: Optional[str] = str(score_path) if score_path else None
            modes_layer_path = _pop(kwargs, 'prune_layer_importance_path', None)
            self._prune_layer_importance_path: Optional[str] = (
                str(modes_layer_path) if modes_layer_path else None
            )
            stats_path = _pop(kwargs, 'prune_stats_path', None)
            self._prune_stats_path: Optional[Path] = Path(stats_path) if stats_path else None
            quantile_collect_method = _pop(kwargs, 'prune_quantile_collect_method', None)
            self._quantile_collect_method: Optional[str] = (
                str(quantile_collect_method).lower() if quantile_collect_method else None
            )
            quantile_collect_dir = _pop(kwargs, 'prune_quantile_collect_dir', None)
            self._quantile_collect_dir: Optional[Path] = (
                Path(quantile_collect_dir) if quantile_collect_dir else None
            )
            self._quantile_collect_chunk_index = 0
            self._runtime_stats = RuntimeStats()
            self._modes_pruning_ratio: Optional[float] = None
            self._model_path_for_pruning = str(kwargs.get('model_path', model_name))

            if self._prune_method not in self._SUPPORTED_METHODS:
                raise ValueError(
                    f'Unsupported prune_method={self._prune_method!r}. '
                    f'Must be one of {sorted(self._SUPPORTED_METHODS)}. '
                    'for qwen3_moe_pruned.'
                )
            if (
                self._quantile_collect_method is not None
                and self._quantile_collect_method not in QUANTILE_RUNTIME_METHODS
            ):
                raise ValueError(
                    f'Unsupported prune_quantile_collect_method={self._quantile_collect_method!r}. '
                    f'Must be one of {sorted(QUANTILE_RUNTIME_METHODS)}.'
                )
            if self._prune_method in {
                'ace',
                'gsp',
                'rcr',
                'diep',
                'modes',
                'aimer',
                'top_p',
                'sere',
                'xshare',
            } and self._prune_tau is None:
                raise ValueError(f'prune_method={self._prune_method} requires prune_tau.')
            if self._prune_method in {'naee', 'score_only', 'expert_sparsity'} and self._prune_beta is None:
                raise ValueError(f'prune_method={self._prune_method} requires prune_beta.')
            if self._prune_method == 'diep' and self._prune_score_path is None:
                raise ValueError('prune_method=diep requires prune_score_path.')

            super().__init__(model_name=model_name, **kwargs)

            # AMP table is derived from model weights; compute once after load.
            naee_active = self._prune_method in {'naee', 'score_only'} and float(self._prune_beta) > 0.0
            if self._prune_method != 'none':
                if self._prune_method in {'naee', 'score_only'}:
                    self._amp_table = build_naee_amp_table_for_model(self.model) if naee_active else None
                    self._proto_amp_table = None
                    self._aimer_keep_table = None
                    self._diep_payload = None
                    self._modes_layer_importance_path = None
                elif self._prune_method in {'expert_sparsity', 'top_p', 'xshare'}:
                    self._amp_table = None
                    self._proto_amp_table = None
                    self._aimer_keep_table = None
                    self._diep_payload = None
                    self._modes_layer_importance_path = None
                    self._sim_table = None
                elif self._prune_method == 'sere':
                    self._amp_table = None
                    self._proto_amp_table = None
                    self._aimer_keep_table = None
                    self._diep_payload = None
                    self._modes_layer_importance_path = None
                    self._sim_table = build_model_similarity_table(self.model, mode=self._prune_sim_mode)
                elif self._prune_method in {'rcr', 'ace'}:
                    self._amp_table = (
                        build_amp_table_for_model(self.model)
                        if self._prune_method == 'ace'
                        else None
                    )
                    self._proto_amp_table = build_router_proto_amp_table_for_model(
                        self.model,
                        center_router_weights=self._router_weight_centering,
                    )
                    self._aimer_keep_table = None
                    self._diep_payload = None
                    self._modes_layer_importance_path = None
                    self._sim_table = None
                elif self._prune_method == 'aimer':
                    self._amp_table = None
                    self._proto_amp_table = None
                    self._aimer_keep_table = build_aimer_keep_table_for_model(self.model)
                    self._diep_payload = None
                    self._modes_layer_importance_path = None
                    self._sim_table = None
                elif self._prune_method == 'diep':
                    self._amp_table = None
                    self._proto_amp_table = None
                    self._aimer_keep_table = None
                    self._diep_payload = _load_diep_payload(self._prune_score_path)
                    self._modes_layer_importance_path = None
                elif self._prune_method == 'modes':
                    self._amp_table = None
                    self._proto_amp_table = None
                    self._aimer_keep_table = None
                    self._diep_payload = None
                    self._sim_table = None
                    self._modes_layer_importance_path = _resolve_modes_layer_importance_path(
                        self._model_path_for_pruning,
                        self._prune_layer_importance_path,
                    )
                    _patch_modes_text_model(self.model, self.tokenizer)
                else:
                    self._amp_table = build_amp_table_for_model(self.model)
                    self._proto_amp_table = None
                    self._aimer_keep_table = None
                    self._diep_payload = None
                    self._modes_layer_importance_path = None
                    self._sim_table = None
            else:
                self._amp_table = None
                self._proto_amp_table = None
                self._aimer_keep_table = None
                self._diep_payload = None
                self._modes_layer_importance_path = None
                self._sim_table = None

            if self._quantile_collect_method == 'gsp':
                self._quantile_amp_table = build_amp_table_for_model(self.model)
                self._quantile_proto_amp_table = None
            elif self._quantile_collect_method == 'ace':
                self._quantile_amp_table = build_amp_table_for_model(self.model)
                self._quantile_proto_amp_table = build_router_proto_amp_table_for_model(
                    self.model,
                    center_router_weights=self._router_weight_centering,
                )
            elif self._quantile_collect_method in {'naee'}:
                self._quantile_amp_table = build_naee_amp_table_for_model(self.model)
                self._quantile_proto_amp_table = None
            elif self._quantile_collect_method == 'rcr':
                self._quantile_amp_table = None
                self._quantile_proto_amp_table = build_router_proto_amp_table_for_model(
                    self.model,
                    center_router_weights=self._router_weight_centering,
                )
            elif self._quantile_collect_method == 'aimer':
                self._quantile_amp_table = build_aimer_keep_table_for_model(self.model)
                self._quantile_proto_amp_table = None
            elif self._quantile_collect_method in {'sere'}:
                self._quantile_amp_table = None
                self._quantile_proto_amp_table = None
                if getattr(self, '_sim_table', None) is None:
                    self._sim_table = build_model_similarity_table(self.model, mode=self._prune_sim_mode)
            else:
                self._quantile_amp_table = None
                self._quantile_proto_amp_table = None

            self._flush_runtime_stats()

        def _flush_runtime_stats(self) -> None:
            if self._prune_stats_path is None:
                return
            avg_dynamic_pruning_ratio = (
                self._modes_pruning_ratio
                if self._prune_method == 'modes'
                else float(self._runtime_stats.mean_pruning_ratio())
            )
            payload = {
                'prune_method': self._prune_method,
                'tau': self._prune_tau,
                'beta': self._prune_beta,
                'router_weight_centering': self._router_weight_centering,
                'avg_dynamic_pruning_ratio': avg_dynamic_pruning_ratio,
                'layers': {
                    str(layer_idx): {
                        'pruning_ratio': float(layer_stats.pruning_ratio),
                        'avg_active_experts': float(layer_stats.avg_active_experts),
                        'total_tokens': int(layer_stats.total_tokens),
                        'total_slots': int(layer_stats.total_slots),
                        'kept_slots': int(layer_stats.kept_slots),
                    }
                    for layer_idx, layer_stats in sorted(self._runtime_stats.layers.items())
                },
            }
            self._prune_stats_path.parent.mkdir(parents=True, exist_ok=True)
            self._prune_stats_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding='utf-8',
            )

        def _install_modes_generate_overrides(self, stack: ExitStack) -> None:
            if self._prune_method != 'modes':
                return

            sentinel = object()
            overrides = {
                '_modes_force_enable_tau_skip': True,
                '_modes_force_tau': {'text': float(self._prune_tau), 'visual': 0.0},
                '_modes_force_enable_load_layer_importance': True,
                '_modes_force_layer_importance_path': str(self._modes_layer_importance_path),
            }

            for target in _modes_override_targets(self.model):
                previous = {
                    name: getattr(target, name, sentinel)
                    for name in overrides
                }
                for name, value in overrides.items():
                    setattr(target, name, value)

                def _restore(target=target, previous=previous) -> None:
                    for name, old_value in previous.items():
                        if old_value is sentinel:
                            delattr(target, name)
                        else:
                            setattr(target, name, old_value)

                stack.callback(_restore)

        def _flush_quantile_collection_chunk(
            self,
            candidate_scores_by_layer: Dict[int, List[object]],
            total_slots_by_layer: Dict[int, int],
        ) -> None:
            if self._quantile_collect_dir is None:
                return
            import torch

            candidate_parts = [
                torch.cat(parts, dim=0)
                for parts in candidate_scores_by_layer.values()
                if parts
            ]
            if not candidate_parts:
                return
            payload = {
                'candidate_scores': torch.cat(candidate_parts, dim=0),
                'total_slots': int(sum(total_slots_by_layer.values())),
            }
            self._quantile_collect_dir.mkdir(parents=True, exist_ok=True)
            path = self._quantile_collect_dir / f'chunk_{self._quantile_collect_chunk_index:06d}.pt'
            torch.save(payload, path)
            self._quantile_collect_chunk_index += 1

        def _enter_prune_context(self, stack: ExitStack) -> None:
            method = self._prune_method
            if method == 'none':
                return
            if method == 'gsp':
                stack.enter_context(
                    patched_model_for_gsp(
                        self.model,
                        amp_table=self._amp_table,
                        tau=float(self._prune_tau),
                        runtime_stats=self._runtime_stats,
                    )
                )
            elif method == 'ace':
                stack.enter_context(
                    patched_model_for_ace(
                        self.model,
                        slanc_amp_table=self._amp_table,
                        proto_amp_table=self._proto_amp_table,
                        tau=float(self._prune_tau),
                        runtime_stats=self._runtime_stats,
                    )
                )
            elif method == 'rcr':
                stack.enter_context(
                    patched_model_for_rcr(
                        self.model,
                        proto_amp_table=self._proto_amp_table,
                        tau=float(self._prune_tau),
                        runtime_stats=self._runtime_stats,
                    )
                )
            elif method == 'aimer':
                stack.enter_context(
                    patched_model_for_aimer(
                        self.model,
                        keep_table=self._aimer_keep_table,
                        tau=float(self._prune_tau),
                        runtime_stats=self._runtime_stats,
                    )
                )
            elif method == 'expert_sparsity':
                if float(self._prune_beta) > 0.0:
                    stack.enter_context(
                        patched_model_for_expert_sparsity(
                            self.model,
                            beta=float(self._prune_beta),
                            runtime_stats=self._runtime_stats,
                        )
                    )
            elif method == 'top_p':
                stack.enter_context(
                    patched_model_for_top_p(
                        self.model,
                        tau=float(self._prune_tau),
                        runtime_stats=self._runtime_stats,
                    )
                )
            elif method == 'sere':
                stack.enter_context(
                    patched_model_for_sere(
                        self.model,
                        tau=float(self._prune_tau),
                        similarity_mode=self._prune_sim_mode,
                        sim_table=self._sim_table,
                        runtime_stats=self._runtime_stats,
                    )
                )
            elif method == 'xshare':
                stack.enter_context(
                    patched_model_for_xshare(
                        self.model,
                        tau=float(self._prune_tau),
                        runtime_stats=self._runtime_stats,
                    )
                )
            elif method in {'naee', 'score_only'}:
                if float(self._prune_beta) > 0.0:
                    stack.enter_context(
                        patched_model_for_naee(
                            self.model,
                            amp_table=self._amp_table,
                            beta=float(self._prune_beta),
                            score_mode='gate' if method == 'score_only' else 'amp',
                            runtime_stats=self._runtime_stats,
                        )
                    )
            elif method == 'diep':
                import sys
                from pathlib import Path as _Path
                _repo = _Path(__file__).resolve().parents[2]
                _diep_dir = _repo / 'ablation' / 'DiEP'
                if str(_diep_dir) not in sys.path:
                    sys.path.insert(0, str(_diep_dir))
                from qwen3_diep_ablation import (  # type: ignore
                    patch_qwen3_moe_blocks_diep,
                )
                stack.enter_context(
                    patch_qwen3_moe_blocks_diep(
                        self.model,
                        per_layer_beta=self._diep_payload['per_layer_beta'],
                        tau=float(self._prune_tau),
                        runtime_stats=self._runtime_stats,
                    )
                )

        def generate(  # type: ignore[override]
            self,
            input: List[ChatMessage],
            tools: List[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
        ) -> ModelOutput:
            config_for_generate = config
            if self._prune_method == 'modes':
                _reset_modes_counters(getattr(self.model.config, 'model_type', ''))
                config_for_generate = copy.deepcopy(config)
                extra_body = dict(config_for_generate.extra_body or {})
                extra_body.update(
                    {
                        'enable_tau_skip': True,
                        'tau': {'text': float(self._prune_tau), 'visual': 0.0},
                        'enable_load_layer_importance': True,
                        'layer_importance_path': str(self._modes_layer_importance_path),
                    }
                )
                config_for_generate.extra_body = extra_body
            with ExitStack() as stack:
                self._install_modes_generate_overrides(stack)
                quantile_scores_by_layer: Dict[int, List[object]] = {}
                quantile_total_slots_by_layer: Dict[int, int] = {}
                if self._quantile_collect_method is not None:
                    stack.enter_context(
                        patched_model_for_quantile_collection(
                            self.model,
                            method=self._quantile_collect_method,
                            candidate_scores_by_layer=quantile_scores_by_layer,
                            total_slots_by_layer=quantile_total_slots_by_layer,
                            amp_tables=self._quantile_amp_table,
                            proto_amp_tables=self._quantile_proto_amp_table,
                            sim_tables=self._sim_table,
                        )
                    )
                self._enter_prune_context(stack)
                output = super().generate(input, tools, tool_choice, config_for_generate)
            if self._quantile_collect_method is not None:
                self._flush_quantile_collection_chunk(
                    quantile_scores_by_layer,
                    quantile_total_slots_by_layer,
                )
            if self._prune_method == 'modes':
                self._modes_pruning_ratio = _read_modes_pruning_ratio(getattr(self.model.config, 'model_type', ''))
            self._flush_runtime_stats()
            return output

    return Qwen3PrunedModelAPI
