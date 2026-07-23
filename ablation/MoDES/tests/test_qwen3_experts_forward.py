import unittest
import importlib
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "MoDES" not in sys.modules:
    modes_pkg = types.ModuleType("MoDES")
    modes_pkg.__path__ = [str(ROOT)]
    sys.modules["MoDES"] = modes_pkg
    sys.modules["MoDES.models"] = importlib.import_module("models")
    sys.modules["MoDES.models.utils"] = importlib.import_module("models.utils")

from models.qwen3 import experts_forward, mlp_forward


class DummyExperts:
    def __init__(self):
        self.hidden_size = 1
        self.num_experts = 3
        self.gate_up_proj = torch.tensor(
            [
                [[1.0], [1.0]],
                [[2.0], [1.0]],
                [[3.0], [1.0]],
            ],
            dtype=torch.float32,
        )
        self.down_proj = torch.tensor(
            [
                [[1.0]],
                [[1.0]],
                [[1.0]],
            ],
            dtype=torch.float32,
        )
        self.act_fn = lambda x: x


class DummyExpertsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 1
        self.num_experts = 3
        self.gate_up_proj = torch.tensor(
            [
                [[1.0], [1.0]],
                [[2.0], [1.0]],
                [[3.0], [1.0]],
            ],
            dtype=torch.float32,
        )
        self.down_proj = torch.tensor(
            [
                [[1.0]],
                [[1.0]],
                [[1.0]],
            ],
            dtype=torch.float32,
        )
        self.act_fn = lambda x: x

    forward = experts_forward


class DummyCountingExpertsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 3

    def forward(self, hidden_states, routing_weights, router_indices):
        return torch.zeros_like(hidden_states)


class DummyGate:
    def __init__(self, router_logits: torch.Tensor):
        self.router_logits = router_logits
        self.moe_text_mask = torch.tensor([[True]])
        self.moe_media_mask = torch.tensor([[False]])
        self.text_layer_importance = 1.0
        self.visual_layer_importance = 1.0

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.router_logits.to(hidden_states.device)


class DummyTupleGate(DummyGate):
    def __init__(
        self,
        router_logits: torch.Tensor,
        routing_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ):
        super().__init__(router_logits)
        self.routing_weights = routing_weights
        self.router_indices = router_indices

    def __call__(self, hidden_states: torch.Tensor):
        return (
            self.router_logits.to(hidden_states.device),
            self.routing_weights.to(hidden_states.device),
            self.router_indices.to(hidden_states.device),
        )


class DummyMLP:
    def __init__(self, router_logits: torch.Tensor):
        self.hidden_size = 1
        self.top_k = 2
        self.experts = DummyExpertsModule()
        self.gate = DummyGate(router_logits)
        self.moe_text_mask = torch.tensor([[True]])
        self.moe_media_mask = torch.tensor([[False]])
        self.moe_padding_mask = None

    forward = mlp_forward


class DummyTupleMLP(DummyMLP):
    def __init__(
        self,
        router_logits: torch.Tensor,
        routing_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ):
        self.hidden_size = 1
        self.top_k = 2
        self.experts = DummyCountingExpertsModule()
        self.gate = DummyTupleGate(router_logits, routing_weights, router_indices)
        self.moe_text_mask = torch.tensor([[True]])
        self.moe_media_mask = torch.tensor([[False]])
        self.moe_padding_mask = None


def reference_experts_forward(self, hidden_states, routing_weights, router_indices):
    next_states = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(
        router_indices, num_classes=self.num_experts
    ).permute(2, 1, 0)
    expert_mask = expert_mask * routing_weights.gt(0).permute(1, 0).unsqueeze(0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = expert_idx_tensor[0].item()
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        out = torch.nn.functional.linear(up * self.act_fn(gate), self.down_proj[expert_idx])
        next_states.index_add_(
            0, token_idx, out * routing_weights[token_idx, top_k_pos, None]
        )
    return next_states


class Qwen3ExpertsForwardTest(unittest.TestCase):
    def test_matches_reference_when_expert_id_exceeds_topk_axis(self):
        dummy = DummyExperts()
        hidden_states = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        routing_weights = torch.tensor(
            [[0.25, 0.75], [0.60, 0.40]], dtype=torch.float32
        )
        router_indices = torch.tensor([[2, 0], [1, 2]], dtype=torch.long)

        expected = reference_experts_forward(
            dummy, hidden_states, routing_weights, router_indices
        )
        actual = experts_forward(dummy, hidden_states, routing_weights, router_indices)

        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(actual, torch.tensor([[1.5], [9.6]])))

    def test_ignores_zero_weight_experts(self):
        dummy = DummyExperts()
        dummy.gate_up_proj[2].fill_(float("nan"))
        dummy.down_proj[2].fill_(float("nan"))

        hidden_states = torch.tensor([[1.0]], dtype=torch.float32)
        routing_weights = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        router_indices = torch.tensor([[0, 2]], dtype=torch.long)

        actual = experts_forward(dummy, hidden_states, routing_weights, router_indices)

        self.assertTrue(torch.allclose(actual, torch.tensor([[1.0]])))
        self.assertFalse(torch.isnan(actual).any())

    def test_tau_skip_changes_dispatched_weights_and_counts(self):
        import models.qwen3 as qwen3_model

        dummy = DummyMLP(torch.tensor([[4.0, 3.0, 0.0]], dtype=torch.float32))
        hidden_states = torch.tensor([[[1.0]]], dtype=torch.float32)

        qwen3_model.SKIP_EXP_COUNT = 0
        qwen3_model.TOTAL_EXP_COUNT = 0
        baseline = mlp_forward(dummy, hidden_states, enable_tau_skip=False)

        qwen3_model.SKIP_EXP_COUNT = 0
        qwen3_model.TOTAL_EXP_COUNT = 0
        pruned = mlp_forward(
            dummy,
            hidden_states,
            enable_tau_skip=True,
            tau={"text": 0.5, "visual": 0.0},
        )

        topk_values = torch.topk(
            torch.softmax(dummy.gate.router_logits, dim=-1), 2, dim=-1
        ).values
        expected_top1 = topk_values.div(topk_values.sum(dim=-1, keepdim=True))[0, 0]

        self.assertFalse(torch.allclose(baseline, pruned))
        self.assertTrue(
            torch.allclose(
                pruned,
                torch.tensor([[[float(expected_top1)]]], dtype=torch.float32),
            )
        )
        self.assertEqual(qwen3_model.TOTAL_EXP_COUNT, 2)
        self.assertEqual(qwen3_model.SKIP_EXP_COUNT, 1)

    def test_tau_zero_with_dense_gate_output_never_reports_negative_skips(self):
        import models.qwen3 as qwen3_model

        dummy = DummyTupleMLP(
            router_logits=torch.tensor([[4.0, 3.0, 1.0]], dtype=torch.float32),
            routing_weights=torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float32),
            router_indices=torch.tensor([[0, 1, 2]], dtype=torch.long),
        )
        hidden_states = torch.tensor([[[1.0]]], dtype=torch.float32)

        qwen3_model.SKIP_EXP_COUNT = 0
        qwen3_model.TOTAL_EXP_COUNT = 0
        mlp_forward(
            dummy,
            hidden_states,
            enable_tau_skip=True,
            tau={"text": 0.0, "visual": 0.0},
        )

        self.assertEqual(qwen3_model.TOTAL_EXP_COUNT, 3)
        self.assertEqual(qwen3_model.SKIP_EXP_COUNT, 0)


if __name__ == "__main__":
    unittest.main()
