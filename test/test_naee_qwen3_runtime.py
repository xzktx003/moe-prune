from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from moe_prune.code.scripts.NAEE.run_naee_ablation import (
        DEFAULT_NAEE_OUTPUT_DIR,
        DEFAULT_PPL_SEQ_LEN,
        build_final_summary,
        beta_row_exists,
        compute_naee_score,
        format_tau_dir,
        build_relative_beta_keep_mask,
        load_existing_artifact,
        load_existing_ppl_rows,
        merge_rows_for_beta_grid,
        missing_betas,
        parse_args,
        patch_qwen3_moe_blocks_naee,
        patched_model_for_naee,
        write_repro_commands,
    )
except ModuleNotFoundError:
    MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "NAEE" / "run_naee_ablation.py"
    SPEC = importlib.util.spec_from_file_location("run_naee_ablation_local", MODULE_PATH)
    assert SPEC is not None and SPEC.loader is not None
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)

    DEFAULT_NAEE_OUTPUT_DIR = MODULE.DEFAULT_NAEE_OUTPUT_DIR
    DEFAULT_PPL_SEQ_LEN = MODULE.DEFAULT_PPL_SEQ_LEN
    build_final_summary = MODULE.build_final_summary
    beta_row_exists = MODULE.beta_row_exists
    compute_naee_score = MODULE.compute_naee_score
    format_tau_dir = MODULE.format_tau_dir
    build_relative_beta_keep_mask = MODULE.build_relative_beta_keep_mask
    load_existing_artifact = MODULE.load_existing_artifact
    load_existing_ppl_rows = MODULE.load_existing_ppl_rows
    merge_rows_for_beta_grid = MODULE.merge_rows_for_beta_grid
    missing_betas = MODULE.missing_betas
    parse_args = MODULE.parse_args
    patch_qwen3_moe_blocks_naee = MODULE.patch_qwen3_moe_blocks_naee
    patched_model_for_naee = MODULE.patched_model_for_naee
    write_repro_commands = MODULE.write_repro_commands


def test_relative_beta_keep_mask_keeps_best_slot():
    score = torch.tensor([[0.2, 0.1, 0.05]])
    keep = build_relative_beta_keep_mask(score, beta=0.9)
    assert keep.tolist() == [[True, False, False]]


def test_relative_beta_keep_mask_thresholds_against_max_proxy():
    score = torch.tensor([[0.8, 0.41, 0.39, 0.2]])
    keep = build_relative_beta_keep_mask(score, beta=0.5)
    assert keep.tolist() == [[True, True, False, False]]


def test_relative_beta_keep_mask_handles_non_monotonic_proxy_scores():
    score = torch.tensor([[0.3, 0.7, 0.69, 0.1]])
    keep = build_relative_beta_keep_mask(score, beta=0.95)
    assert keep.tolist() == [[False, True, True, False]]


def test_compute_naee_score_uses_gate_only():
    gate = torch.tensor([[0.7, 0.2, 0.1]])
    score = compute_naee_score(gate)
    assert torch.equal(score, gate)


def test_build_final_summary_uses_null_for_missing_sections():
    rows = build_final_summary(
        beta_grid=[0.0, 0.15],
        datasets=["mathqa", "openbookqa", "arc_challenge"],
        evaluation_rows=[{"beta": 0.0, "mathqa": 0.5, "openbookqa": 0.6, "arc_challenge": 0.7, "avg_dynamic_pruning_ratio": 0.1}],
        ppl_rows=[],
        calibration_summary={},
    )

    assert rows[0]["mathqa"] == 0.5
    assert rows[0]["wikitext_ppl"] is None
    assert rows[1]["mathqa"] is None
    assert rows[1]["avg_dynamic_pruning_ratio"] is None
    assert rows[1]["calibration_rel_mse"] is None


def test_write_repro_commands_targets_invocable_module(tmp_path):
    write_repro_commands(tmp_path, ["--beta-grid", "0.0", "0.15", "--output-dir", "foo"])
    script = (tmp_path / "repro_commands.sh").read_text()

    assert "python -m moe_prune.code.scripts.NAEE.run_naee_ablation" in script
    assert "moe_prune.ablation.NAEE.qwen3_naee_ablation" not in script


def test_parse_args_defaults_to_canonical_output_and_seq2048(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_naee_ablation"])
    args = parse_args()

    assert args.output_dir == DEFAULT_NAEE_OUTPUT_DIR
    assert args.ppl_n_ctx == DEFAULT_PPL_SEQ_LEN
    assert args.ppl_n_batch == DEFAULT_PPL_SEQ_LEN


def test_load_existing_artifact_prefers_final_json_and_preserves_list_shape(tmp_path):
    final_path = tmp_path / "results_table.json"
    final_path.write_text('[{"beta": 0.1, "mathqa": 0.5}]', encoding="utf-8")

    loaded = load_existing_artifact(final_path, [])

    assert loaded == [{"beta": 0.1, "mathqa": 0.5}]


def test_load_existing_artifact_falls_back_to_partial_json_when_final_missing(tmp_path):
    final_path = tmp_path / "wikitext_ppl.json"
    partial_path = tmp_path / "wikitext_ppl.partial.json"
    partial_path.write_text('[{"beta": 0.2, "ppl": 7.3}]', encoding="utf-8")

    loaded = load_existing_artifact(final_path, [])

    assert loaded == [{"beta": 0.2, "ppl": 7.3}]


def test_load_existing_artifact_ignores_mismatched_json_shape(tmp_path):
    final_path = tmp_path / "proxy_summary.json"
    final_path.write_text("[]", encoding="utf-8")

    loaded = load_existing_artifact(final_path, {"layer_count": 0.0})

    assert loaded == {"layer_count": 0.0}


def test_naee_resume_helpers_detect_missing_beta_rows():
    rows = [{"beta": 0.25, "ppl": 10.0}]

    assert beta_row_exists(rows, 0.25)
    assert missing_betas([0.25, 0.35], rows) == [0.35]


def test_naee_merge_rows_preserves_requested_beta_order():
    merged = merge_rows_for_beta_grid(
        beta_grid=[0.25, 0.35],
        existing_rows=[{"beta": 0.35, "ppl": 11.0}],
        new_rows=[{"beta": 0.25, "ppl": 10.0}],
    )

    assert merged == [{"beta": 0.25, "ppl": 10.0}, {"beta": 0.35, "ppl": 11.0}]


def test_load_existing_ppl_rows_reads_per_tau_artifact(tmp_path):
    per_tau_dir = tmp_path / "ppl_by_tau" / format_tau_dir(0.25)
    per_tau_dir.mkdir(parents=True)
    (per_tau_dir / "wikitext_ppl.json").write_text('{"beta": 0.25, "ppl": 10.0}', encoding="utf-8")

    rows = load_existing_ppl_rows(tmp_path, [0.25, 0.35])

    assert rows == [{"beta": 0.25, "ppl": 10.0}]


def test_patched_model_for_naee_short_circuits_zero_beta(monkeypatch) -> None:
    calls = []

    @contextmanager
    def _fake_patch(**kwargs):
        calls.append(kwargs)
        yield kwargs["model"]

    monkeypatch.setattr(MODULE if "MODULE" in globals() else sys.modules[patched_model_for_naee.__module__], "patch_qwen3_moe_blocks_naee", _fake_patch)

    model = object()
    with patched_model_for_naee(model=model, amp_table={0: torch.ones(1)}, beta=0.0):
        pass

    assert calls == []


class _FakeRouter(torch.nn.Module):
    def __init__(self, num_experts: int, hidden: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(num_experts, hidden, generator=g))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class _FakeMLP(torch.nn.Module):
    def __init__(self, layer_idx: int, hidden: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.norm_topk_prob = True
        self.gate = _FakeRouter(num_experts, hidden, seed=100 + layer_idx)
        self.experts = SimpleNamespace(
            gate_up_proj=torch.randn(num_experts, 2 * hidden, hidden) * 0.02,
            down_proj=torch.randn(num_experts, hidden, hidden) * 0.02,
            act_fn=torch.nn.SiLU(),
        )

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(flat)
        probs = torch.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        module = sys.modules[patch_qwen3_moe_blocks_naee.__module__]
        out = module.compute_expert_outputs_local(flat, self.experts, selected_experts)
        merged = (routing_weights.unsqueeze(-1) * out).sum(dim=1)
        return merged.view(batch_size, sequence_length, hidden_dim)


class _FakeLayer(torch.nn.Module):
    def __init__(self, layer_idx: int, hidden: int, num_experts: int, top_k: int):
        super().__init__()
        self.mlp = _FakeMLP(layer_idx, hidden, num_experts, top_k)


class _FakeModel(torch.nn.Module):
    def __init__(self, num_layers: int = 2, hidden: int = 16, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.config = SimpleNamespace(num_experts_per_tok=top_k, num_experts=num_experts, vocab_size=64)
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList(
                [_FakeLayer(i, hidden, num_experts, top_k) for i in range(num_layers)]
            )
        )
        self.hidden = hidden

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        del use_cache, kwargs
        if input_ids is None:
            raise ValueError("input_ids required")
        hidden = torch.randn(
            input_ids.shape[0],
            input_ids.shape[1],
            self.hidden,
            generator=torch.Generator().manual_seed(int(input_ids.sum().item()) % 1000),
        )
        for layer in self.model.layers:
            hidden = layer.mlp(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def test_patch_qwen3_moe_blocks_naee_mlp_returns_tensor() -> None:
    model = _FakeModel()
    runtime_stats = sys.modules[patch_qwen3_moe_blocks_naee.__module__].RuntimeStats()
    amp_table = {0: torch.ones(4), 1: torch.ones(4)}
    beta_by_layer = {0: 0.1, 1: 0.1}

    with patch_qwen3_moe_blocks_naee(
        model=model,
        amp_table=amp_table,
        beta_by_layer=beta_by_layer,
        runtime_stats=runtime_stats,
    ):
        outputs = model(input_ids=torch.arange(1, 9, dtype=torch.long).unsqueeze(0), use_cache=False)

    assert isinstance(outputs.last_hidden_state, torch.Tensor)
    assert runtime_stats.layers
