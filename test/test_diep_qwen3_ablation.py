"""Smoke test for the DiEP Qwen3-MoE adaptation.

We monkeypatch just enough to exercise:
  * calibration loop -> per-layer beta storage on disk
  * DiEP dynamic skip mask semantics for edge tau values (0 -> keep all, large -> collapse)

Runs on CPU with a tiny synthetic "model" so no GPU is required.
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "code"
    / "ablation"
    / "DiEP"
    / "qwen3_diep_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("qwen3_diep_ablation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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
        self.layer_idx = layer_idx
        self.hidden = hidden
        self.top_k = top_k
        self.norm_topk_prob = True
        self.gate = _FakeRouter(num_experts, hidden, seed=100 + layer_idx)
        # Fused-like experts layout. F.linear(x, w) computes x @ w.T so
        # gate_up_proj per expert is [2*hidden, hidden]; down_proj is [hidden, hidden].
        self.experts = SimpleNamespace(
            gate_up_proj=torch.randn(num_experts, 2 * hidden, hidden) * 0.02,
            down_proj=torch.randn(num_experts, hidden, hidden) * 0.02,
            act_fn=torch.nn.SiLU(),
        )

    def forward(self, hidden_states):
        # Route top_k to simulate a normal Qwen3MoE forward.
        b, s, h = hidden_states.shape
        flat = hidden_states.view(-1, h)
        router_logits = self.gate(flat)
        probs = torch.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        out = MODULE.compute_expert_outputs(flat, self.experts, selected_experts)
        merged = (routing_weights.unsqueeze(-1) * out).sum(dim=1)
        return merged.view(b, s, h)


class _FakeLayer(torch.nn.Module):
    def __init__(self, layer_idx: int, hidden: int, num_experts: int, top_k: int):
        super().__init__()
        self.mlp = _FakeMLP(layer_idx, hidden, num_experts, top_k)


class _FakeModel(torch.nn.Module):
    def __init__(self, num_layers: int = 2, hidden: int = 16, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.config = SimpleNamespace(
            num_experts_per_tok=top_k,
            num_experts=num_experts,
            vocab_size=64,
        )
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList(
                [_FakeLayer(i, hidden, num_experts, top_k) for i in range(num_layers)]
            ),
        )
        self.hidden = hidden

    def forward(self, input_ids=None, labels=None, use_cache=False, **kwargs):
        del use_cache, kwargs
        # Pretend an embedding: map token ids to deterministic hidden states.
        if input_ids is None:
            raise ValueError("input_ids required")
        # use a small fake embedding
        hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], self.hidden, generator=torch.Generator().manual_seed(int(input_ids.sum().item()) % 1000))
        for layer in self.model.layers:
            hidden = layer.mlp(hidden)
        loss = hidden.pow(2).mean()
        return SimpleNamespace(loss=loss)


def _patch_module_helpers(monkeypatch):
    monkeypatch.setattr(MODULE, "_resolve_device", lambda model: torch.device("cpu"))

    def _fake_calibration_text(**kwargs):
        return ("hello " * 64, 4)

    def _fake_eval_text(**kwargs):
        return ("testing " * 128, 8)

    monkeypatch.setattr(MODULE, "load_wikitext_calibration_text", _fake_calibration_text)
    monkeypatch.setattr(MODULE, "load_wikitext_eval_text", _fake_eval_text)

    # replace autocast with a no-op context manager
    from contextlib import contextmanager

    @contextmanager
    def _noop():
        yield

    monkeypatch.setattr(MODULE, "maybe_bf16_autocast", _noop)


def _fake_tokenizer():
    class _Tok:
        model_max_length = 2**31 - 1

        def __call__(self, text, truncation, return_tensors):
            del text, truncation, return_tensors
            ids = torch.arange(1, 33, dtype=torch.long).unsqueeze(0)
            return SimpleNamespace(input_ids=ids)

    return _Tok()


def test_build_diep_keep_mask_tau_zero_keeps_all():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(5, 6), dim=-1)
    routing_weights, selected = torch.topk(probs, 3, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    keep_mask = MODULE.build_diep_keep_mask(
        routing_weights=routing_weights,
        full_probs=probs,
        beta_layer=0.5,
        tau=0.0,
    )
    assert keep_mask.all()


def test_build_diep_keep_mask_large_tau_collapses_to_top1():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(5, 6), dim=-1)
    routing_weights, _ = torch.topk(probs, 3, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    keep_mask = MODULE.build_diep_keep_mask(
        routing_weights=routing_weights,
        full_probs=probs,
        beta_layer=10.0,
        tau=100.0,
    )
    # every row keeps exactly one (the top-1) expert
    assert keep_mask.sum(dim=-1).tolist() == [1] * routing_weights.shape[0]
    top1_positions = routing_weights.argmax(dim=-1)
    for row, pos in enumerate(top1_positions.tolist()):
        assert bool(keep_mask[row, pos])


def test_calibration_and_score_persistence(tmp_path, monkeypatch):
    _patch_module_helpers(monkeypatch)
    model = _FakeModel(num_layers=2, hidden=16, num_experts=4, top_k=2)
    tokenizer = _fake_tokenizer()

    score_path = tmp_path / "score.pkl"
    payload = MODULE.load_or_compute_score(
        model=model,
        tokenizer=tokenizer,
        score_path=score_path,
        num_samples=2,
        max_length=8,
        calibration_split="train",
        force_recalibrate=True,
    )

    assert score_path.exists()
    assert set(payload["per_layer_beta"].keys()) == {0, 1}
    for beta in payload["per_layer_beta"].values():
        assert 0.0 <= beta <= 1.0
    with open(score_path, "rb") as handle:
        loaded = pickle.load(handle)
    assert loaded["per_layer_beta"] == payload["per_layer_beta"]


def test_evaluate_with_patch_changes_pruning_ratio(tmp_path, monkeypatch):
    _patch_module_helpers(monkeypatch)
    model = _FakeModel(num_layers=2, hidden=16, num_experts=4, top_k=2)
    tokenizer = _fake_tokenizer()

    score_payload = MODULE.calibrate_layer_scores(
        model=model, tokenizer=tokenizer, num_samples=2, max_length=8
    )
    per_layer_beta = score_payload["per_layer_beta"]

    # tau=0 should prune nothing
    stats_zero = MODULE.RuntimeStats()
    with MODULE.patch_qwen3_moe_blocks_diep(
        model=model, per_layer_beta=per_layer_beta, tau=0.0, runtime_stats=stats_zero
    ):
        MODULE.evaluate_wikitext_ppl(
            model,
            tokenizer,
            split="test",
            text_column="text",
            min_text_length=1,
            n_ctx=8,
        )
    # very aggressive tau should prune some experts
    stats_high = MODULE.RuntimeStats()
    with MODULE.patch_qwen3_moe_blocks_diep(
        model=model, per_layer_beta=per_layer_beta, tau=100.0, runtime_stats=stats_high
    ):
        MODULE.evaluate_wikitext_ppl(
            model,
            tokenizer,
            split="test",
            text_column="text",
            min_text_length=1,
            n_ctx=8,
        )

    assert stats_zero.mean_pruning_ratio() == 0.0
    assert stats_high.mean_pruning_ratio() > stats_zero.mean_pruning_ratio()
