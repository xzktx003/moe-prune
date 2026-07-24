from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from moe_prune.code.src.triton_group_gemm import triton_moe_available


pytestmark = pytest.mark.skipif(not triton_moe_available(), reason="Ablation Triton tests require CUDA + Triton")
MODES_ROOT = REPO_ROOT / "code" / "ablation" / "MoDES"
if str(MODES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODES_ROOT))
if "MoDES" not in sys.modules:
    modes_pkg = types.ModuleType("MoDES")
    modes_pkg.__path__ = [str(MODES_ROOT)]
    sys.modules["MoDES"] = modes_pkg
    sys.modules["MoDES.models"] = importlib.import_module("models")
    sys.modules["MoDES.models.utils"] = importlib.import_module("models.utils")


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


diep_module = _load_module("_copilot_diep_runtime", "code/ablation/DiEP/qwen3_diep_ablation.py")
eat_module = _load_module("_copilot_eat_runtime", "code/ablation/EAT-MOE/qwen3_eat_moe_ablation.py")
modes_qwen3 = importlib.import_module("models.qwen3")
modes_gemma4 = importlib.import_module("models.gemma4")


def _make_fused_experts(
    *,
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    return SimpleNamespace(
        num_experts=num_experts,
        gate_up_proj=torch.randn(
            num_experts,
            2 * intermediate_dim,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        * 0.02,
        down_proj=torch.randn(
            num_experts,
            hidden_dim,
            intermediate_dim,
            device=device,
            dtype=dtype,
        )
        * 0.02,
        act_fn=torch.nn.SiLU(),
    )


def _make_router(hidden_dim: int, num_experts: int, dtype: torch.dtype, device: torch.device) -> torch.nn.Linear:
    router = torch.nn.Linear(hidden_dim, num_experts, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        router.weight.copy_(torch.randn_like(router.weight) * 0.02)
    return router


class _SharedExpert(torch.nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False, device=device, dtype=dtype)
        self.up_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False, device=device, dtype=dtype)
        self.down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.act_fn = torch.nn.SiLU()
        with torch.no_grad():
            for layer in (self.gate_proj, self.up_proj, self.down_proj):
                layer.weight.copy_(torch.randn_like(layer.weight) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_diep_triton_matches_torch() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    hidden_states = torch.randn(2, 6, 16, device=device, dtype=dtype)
    router = _make_router(16, 8, dtype, device)
    experts = _make_fused_experts(
        num_experts=8,
        hidden_dim=16,
        intermediate_dim=12,
        dtype=dtype,
        device=device,
    )

    torch_output = diep_module.moe_forward_with_diep_skip(
        hidden_states=hidden_states,
        router=router,
        experts=experts,
        beta_layer=0.65,
        tau=0.75,
        top_k=4,
        norm_topk_prob=True,
        moe_backend="torch",
    )
    triton_output = diep_module.moe_forward_with_diep_skip(
        hidden_states=hidden_states,
        router=router,
        experts=experts,
        beta_layer=0.65,
        tau=0.75,
        top_k=4,
        norm_topk_prob=True,
        moe_backend="triton",
    )

    assert triton_output.dtype == torch_output.dtype
    assert torch.allclose(triton_output, torch_output, atol=5e-2, rtol=5e-2)


def test_diep_qwen36_shared_expert_triton_matches_torch() -> None:
    torch.manual_seed(10)
    device = torch.device("cuda")
    dtype = torch.float16
    hidden_states = torch.randn(2, 4, 16, device=device, dtype=dtype)
    router = _make_router(16, 8, dtype, device)
    experts = _make_fused_experts(
        num_experts=8,
        hidden_dim=16,
        intermediate_dim=12,
        dtype=dtype,
        device=device,
    )
    shared_expert = _SharedExpert(16, 10, dtype, device)
    shared_expert_gate = torch.nn.Linear(16, 1, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        shared_expert_gate.weight.copy_(torch.randn_like(shared_expert_gate.weight) * 0.02)

    kwargs = dict(
        hidden_states=hidden_states,
        router=router,
        experts=experts,
        beta_layer=0.55,
        tau=0.7,
        top_k=4,
        norm_topk_prob=True,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
    )
    torch_output = diep_module.moe_forward_with_diep_skip(moe_backend="torch", **kwargs)
    triton_output = diep_module.moe_forward_with_diep_skip(moe_backend="triton", **kwargs)

    assert torch.allclose(triton_output, torch_output, atol=5e-2, rtol=5e-2)


def test_diep_gemma4_triton_matches_torch() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.float16
    hidden_states = torch.randn(9, 16, device=device, dtype=dtype)
    experts = _make_fused_experts(
        num_experts=6,
        hidden_dim=16,
        intermediate_dim=12,
        dtype=dtype,
        device=device,
    )
    logits = torch.randn(9, 6, device=device, dtype=torch.float32)
    full_probs = torch.softmax(logits, dim=-1)
    routing_weights, selected_experts = torch.topk(full_probs, 3, dim=-1)
    routing_weights = routing_weights.to(dtype)

    kwargs = dict(
        hidden_states=hidden_states,
        experts=experts,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        full_probs=full_probs,
        beta_layer=0.5,
        tau=0.6,
    )
    torch_output = diep_module.moe_experts_forward_with_diep_skip(moe_backend="torch", **kwargs)
    triton_output = diep_module.moe_experts_forward_with_diep_skip(moe_backend="triton", **kwargs)

    assert torch.allclose(triton_output, torch_output, atol=5e-2, rtol=5e-2)


def test_eat_patch_triton_matches_torch() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.float16
    hidden_states = torch.randn(2, 5, 16, device=device, dtype=dtype)
    experts = _make_fused_experts(
        num_experts=8,
        hidden_dim=16,
        intermediate_dim=12,
        dtype=dtype,
        device=device,
    )
    gate = _make_router(16, 8, dtype, device)
    mlp = SimpleNamespace(experts=experts, gate=gate)
    mlp.forward = lambda hidden_states: hidden_states
    layer = SimpleNamespace(mlp=mlp)
    model = SimpleNamespace(
        config=SimpleNamespace(num_experts_per_tok=4, norm_topk_prob=True),
        model=SimpleNamespace(layers=[layer]),
    )

    def run(backend: str):
        controller = eat_module.EATController(
            num_layers=1,
            config=eat_module.EATConfig(
                initial_threshold=0.15,
                min_threshold=0.01,
                max_threshold=0.95,
                max_active_experts=4,
            ),
        )
        with eat_module.patch_qwen3_moe_blocks_eat(model, controller, moe_backend=backend):
            output = mlp.forward(hidden_states)
        return output, controller

    torch_output, torch_controller = run("torch")
    triton_output, triton_controller = run("triton")

    assert torch.allclose(triton_output, torch_output, atol=5e-2, rtol=5e-2)
    assert torch_controller.runtime_stats.mean_pruning_ratio() == pytest.approx(
        triton_controller.runtime_stats.mean_pruning_ratio(),
        abs=1e-6,
    )


def test_eat_qwen36_shared_expert_triton_matches_torch() -> None:
    torch.manual_seed(12)
    device = torch.device("cuda")
    dtype = torch.float16
    hidden_states = torch.randn(2, 4, 16, device=device, dtype=dtype)
    experts = _make_fused_experts(
        num_experts=8,
        hidden_dim=16,
        intermediate_dim=12,
        dtype=dtype,
        device=device,
    )
    gate = _make_router(16, 8, dtype, device)
    shared_expert = _SharedExpert(16, 10, dtype, device)
    shared_expert_gate = torch.nn.Linear(16, 1, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        shared_expert_gate.weight.copy_(torch.randn_like(shared_expert_gate.weight) * 0.02)

    def run(backend: str):
        controller = eat_module.EATController(
            num_layers=1,
            config=eat_module.EATConfig(
                initial_threshold=0.15,
                min_threshold=0.01,
                max_threshold=0.95,
                max_active_experts=4,
            ),
        )
        return eat_module.moe_forward_with_eat_selection(
            hidden_states,
            router=gate,
            experts=experts,
            controller=controller,
            layer_idx=0,
            top_k=4,
            norm_topk_prob=True,
            moe_backend=backend,
            shared_expert=shared_expert,
            shared_expert_gate=shared_expert_gate,
        )

    torch_output = run("torch")
    triton_output = run("triton")
    assert torch.allclose(triton_output, torch_output, atol=5e-2, rtol=5e-2)


def test_eat_gemma4_triton_matches_torch() -> None:
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.float16
    hidden_states = torch.randn(7, 16, device=device, dtype=dtype)
    experts = _make_fused_experts(
        num_experts=5,
        hidden_dim=16,
        intermediate_dim=12,
        dtype=dtype,
        device=device,
    )
    top_k_index = torch.randint(0, 5, (7, 3), device=device)
    top_k_weights = torch.rand(7, 3, device=device, dtype=dtype)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    def run(backend: str):
        controller = eat_module.EATController(
            num_layers=1,
            config=eat_module.EATConfig(
                initial_threshold=0.1,
                min_threshold=0.01,
                max_threshold=0.95,
                max_active_experts=3,
            ),
        )
        return eat_module.moe_experts_forward_with_eat_selection(
            hidden_states,
            experts=experts,
            selected_experts=top_k_index,
            routing_weights=top_k_weights,
            controller=controller,
            layer_idx=0,
            moe_backend=backend,
        )

    torch_output = run("torch")
    triton_output = run("triton")
    assert torch.allclose(triton_output, torch_output, atol=5e-2, rtol=5e-2)


class _ModesGate:
    def __init__(self, router_logits: torch.Tensor):
        self.router_logits = router_logits
        self.moe_text_mask = torch.tensor([[True]], device=router_logits.device)
        self.moe_media_mask = torch.tensor([[False]], device=router_logits.device)
        self.text_layer_importance = 1.0
        self.visual_layer_importance = 1.0

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.router_logits.to(hidden_states.device)


class _ModesMLP:
    def __init__(self, router_logits: torch.Tensor, experts) -> None:
        self.hidden_size = experts.gate_up_proj.shape[-1]
        self.top_k = 4
        self.experts = experts
        self.gate = _ModesGate(router_logits)
        self.moe_text_mask = torch.tensor([[True]], device=router_logits.device)
        self.moe_media_mask = torch.tensor([[False]], device=router_logits.device)
        self.moe_padding_mask = None
        self._moe_backend = getattr(experts, "_moe_backend", "triton")


def test_modes_qwen3_triton_matches_torch() -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.float16
    router_logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.5, 0.25]], device=device, dtype=dtype)
    hidden_states = torch.randn(1, 1, 16, device=device, dtype=dtype)

    experts = _make_fused_experts(
        num_experts=6,
        hidden_dim=16,
        intermediate_dim=10,
        dtype=dtype,
        device=device,
    )

    def run(backend: str):
        experts._moe_backend = backend
        mlp = _ModesMLP(router_logits, experts)
        mlp._moe_backend = backend
        return modes_qwen3.mlp_forward(
            mlp,
            hidden_states,
            enable_tau_skip=True,
            tau={"text": 0.15, "visual": 0.0},
        )

    torch_output = run("torch")
    triton_output = run("triton")
    assert torch.allclose(triton_output, torch_output, atol=5e-2, rtol=5e-2)


def test_modes_gemma4_triton_matches_torch() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.float16
    experts = _make_fused_experts(
        num_experts=5,
        hidden_dim=16,
        intermediate_dim=12,
        dtype=dtype,
        device=device,
    )
    hidden_states = torch.randn(7, 16, device=device, dtype=dtype)
    top_k_index = torch.randint(0, 5, (7, 3), device=device)
    top_k_weights = torch.rand(7, 3, device=device, dtype=dtype)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    experts._moe_backend = "torch"
    torch_output = modes_gemma4.experts_forward(experts, hidden_states, top_k_index, top_k_weights)
    experts._moe_backend = "triton"
    triton_output = modes_gemma4.experts_forward(experts, hidden_states, top_k_index, top_k_weights)

    assert torch.allclose(triton_output, torch_output, atol=5e-2, rtol=5e-2)
