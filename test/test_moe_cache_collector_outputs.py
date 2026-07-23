from __future__ import annotations

import torch

from moe_prune.code.src.moe_cache_collector import compute_expert_outputs_local


class _BiasExpert(torch.nn.Module):
    def __init__(self, bias: float) -> None:
        super().__init__()
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


def test_compute_expert_outputs_local_supports_module_list_experts() -> None:
    experts = torch.nn.ModuleList([_BiasExpert(1.0), _BiasExpert(2.0)])
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    selected = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    outputs = compute_expert_outputs_local(hidden, experts, selected)

    assert outputs.shape == (2, 2, 2)
    assert torch.allclose(outputs[0, 0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(outputs[0, 1], torch.tensor([3.0, 4.0]))
    assert torch.allclose(outputs[1, 0], torch.tensor([5.0, 6.0]))
    assert torch.allclose(outputs[1, 1], torch.tensor([4.0, 5.0]))
