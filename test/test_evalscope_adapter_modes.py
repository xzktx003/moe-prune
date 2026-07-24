from pathlib import Path


def test_modes_evalscope_adapter_supports_qwen35_text_model_type() -> None:
    text = (Path(__file__).resolve().parents[1] / "src" / "evalscope_adapter.py").read_text(encoding="utf-8")

    assert 'qwen3_5_moe_text' in text
    assert "if model_type in {'qwen3_moe', 'qwen3_5_moe', 'qwen3_5_moe_text'}:" in text


def test_naee_evalscope_adapter_short_circuits_zero_beta_setup() -> None:
    text = (Path(__file__).resolve().parents[1] / "src" / "evalscope_adapter.py").read_text(encoding="utf-8")

    assert "naee_active = self._prune_method in {'naee', 'score_only'} and float(self._prune_beta) > 0.0" in text
    assert "self._amp_table = build_naee_amp_table_for_model(self.model)" in text
    assert "elif method in {'naee', 'score_only'}:" in text
    assert "if float(self._prune_beta) > 0.0:" in text


def test_method4_evalscope_adapter_builds_proto_tables_and_patch_context() -> None:
    text = (Path(__file__).resolve().parents[1] / "src" / "evalscope_adapter.py").read_text(encoding="utf-8")

    assert "'method4'" in text
    assert "self._proto_amp_table = build_router_proto_amp_table_for_model(self.model)" in text
    assert "elif self._quantile_collect_method in {'method4'}:" in text
    assert "self._quantile_proto_amp_table = build_router_proto_amp_table_for_model(self.model)" in text
    assert "elif method == 'method4':" in text
    assert "patched_model_for_method4(" in text


def test_modes_evalscope_adapter_installs_generate_overrides() -> None:
    text = (Path(__file__).resolve().parents[1] / "src" / "evalscope_adapter.py").read_text(encoding="utf-8")

    assert "def _install_modes_generate_overrides(self, stack: ExitStack) -> None:" in text
    assert "'_modes_force_enable_tau_skip': True" in text
    assert "'_modes_force_tau': {'text': float(self._prune_tau), 'visual': 0.0}" in text
    assert "self._install_modes_generate_overrides(stack)" in text


def test_modes_model_patches_can_fallback_to_generate_overrides() -> None:
    modes_root = Path(__file__).resolve().parents[1] / "ablation" / "MoDES" / "models"
    qwen3_text = (modes_root / "qwen3.py").read_text(encoding="utf-8")
    gemma4_text = (modes_root / "gemma4.py").read_text(encoding="utf-8")

    assert 'getattr(self, "_modes_force_enable_tau_skip"' in qwen3_text
    assert 'getattr(self.model, "_modes_force_enable_tau_skip"' in qwen3_text
    assert 'getattr(self, "_modes_force_layer_importance_path"' in qwen3_text
    assert 'getattr(self, "_modes_force_enable_tau_skip"' in gemma4_text
    assert 'getattr(self.model.language_model, "_modes_force_enable_tau_skip"' in gemma4_text


def test_modes_qwen3_decoder_forwards_tau_without_top_k_attr_dependency() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "ablation" / "MoDES" / "models" / "qwen3.py"
    ).read_text(encoding="utf-8")

    assert 'if hasattr(self.mlp, "gate"):' in text
    assert 'if hasattr(self.mlp, "top_k"):' not in text
