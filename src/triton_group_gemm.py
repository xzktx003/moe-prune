from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only when Triton is absent.
    class _TritonStub:
        @staticmethod
        def jit(fn):
            return fn

    class _TLStub:
        constexpr = object

    triton = None
    tl = _TLStub()
    _triton_jit = _TritonStub.jit
else:
    _triton_jit = triton.jit


@dataclass(frozen=True)
class TritonMoeStats:
    active_slots: int
    hit_experts: int


def triton_moe_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _require_triton() -> None:
    if triton is None:
        raise ImportError("Triton score_only MoE backend requires the triton package.")
    if not torch.cuda.is_available():
        raise RuntimeError("Triton score_only MoE backend requires CUDA.")


@_triton_jit
def _grouped_gemm_token_to_grouped_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    gather_indices_ptr,
    m_sizes_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tidx = tl.program_id(0).to(tl.int32)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    processed_tiles = tl.zeros((), dtype=tl.int32)
    m_end = tl.zeros((), dtype=tl.int32)

    for expert_idx in tl.range(0, NUM_EXPERTS):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size
        num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        tile_end = processed_tiles + num_m_tiles * num_n_tiles

        while tidx >= processed_tiles and tidx < tile_end:
            tile_idx = tidx - processed_tiles
            group_id = tile_idx // (GROUP_SIZE_M * num_n_tiles)
            first_m = group_id * GROUP_SIZE_M
            group_size_m = tl.minimum(num_m_tiles - first_m, GROUP_SIZE_M)
            within_group = tile_idx % (GROUP_SIZE_M * num_n_tiles)
            tile_m_idx = first_m + (within_group % group_size_m)
            tile_n_idx = within_group // group_size_m

            rows = tile_m_idx * BLOCK_SIZE_M + offs_m
            cols = tile_n_idx * BLOCK_SIZE_N + offs_n
            row_mask = rows < m_size
            col_mask = cols < N
            grouped_rows = m_start + rows
            flat_slots = tl.load(gather_indices_ptr + grouped_rows, mask=row_mask, other=0).to(tl.int32)
            token_idx = flat_slots // TOPK

            total = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k_start in tl.range(0, K, BLOCK_SIZE_K):
                k_offsets = k_start + offs_k
                k_mask = k_offsets < K
                x_offsets = token_idx[:, None].to(tl.int64) * K + k_offsets[None, :]
                w_offsets = (
                    expert_idx.to(tl.int64) * (N * K)
                    + cols[:, None].to(tl.int64) * K
                    + k_offsets[None, :]
                )
                x_vals = tl.load(x_ptr + x_offsets, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                w_vals = tl.load(w_ptr + w_offsets, mask=col_mask[:, None] & k_mask[None, :], other=0.0)
                total += tl.dot(x_vals, tl.trans(w_vals))

            tl.store(
                y_ptr + grouped_rows[:, None].to(tl.int64) * N + cols[None, :],
                total,
                mask=row_mask[:, None] & col_mask[None, :],
            )
            tidx += NUM_SMS

        processed_tiles = tile_end


@_triton_jit
def _grouped_gemm_grouped_to_token_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    gather_indices_ptr,
    m_sizes_ptr,
    weights_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tidx = tl.program_id(0).to(tl.int32)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    processed_tiles = tl.zeros((), dtype=tl.int32)
    m_end = tl.zeros((), dtype=tl.int32)

    for expert_idx in tl.range(0, NUM_EXPERTS):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size
        num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        tile_end = processed_tiles + num_m_tiles * num_n_tiles

        while tidx >= processed_tiles and tidx < tile_end:
            tile_idx = tidx - processed_tiles
            group_id = tile_idx // (GROUP_SIZE_M * num_n_tiles)
            first_m = group_id * GROUP_SIZE_M
            group_size_m = tl.minimum(num_m_tiles - first_m, GROUP_SIZE_M)
            within_group = tile_idx % (GROUP_SIZE_M * num_n_tiles)
            tile_m_idx = first_m + (within_group % group_size_m)
            tile_n_idx = within_group // group_size_m

            rows = tile_m_idx * BLOCK_SIZE_M + offs_m
            cols = tile_n_idx * BLOCK_SIZE_N + offs_n
            row_mask = rows < m_size
            col_mask = cols < N
            grouped_rows = m_start + rows
            flat_slots = tl.load(gather_indices_ptr + grouped_rows, mask=row_mask, other=0).to(tl.int32)
            token_idx = flat_slots // TOPK

            total = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k_start in tl.range(0, K, BLOCK_SIZE_K):
                k_offsets = k_start + offs_k
                k_mask = k_offsets < K
                x_offsets = grouped_rows[:, None].to(tl.int64) * K + k_offsets[None, :]
                w_offsets = (
                    expert_idx.to(tl.int64) * (N * K)
                    + cols[:, None].to(tl.int64) * K
                    + k_offsets[None, :]
                )
                x_vals = tl.load(x_ptr + x_offsets, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                w_vals = tl.load(w_ptr + w_offsets, mask=col_mask[:, None] & k_mask[None, :], other=0.0)
                total += tl.dot(x_vals, tl.trans(w_vals))

            weights = tl.load(weights_ptr + flat_slots, mask=row_mask, other=0.0).to(tl.float32)
            total *= weights[:, None]
            tl.atomic_add(
                y_ptr + token_idx[:, None].to(tl.int64) * N + cols[None, :],
                total,
                sem="relaxed",
                mask=row_mask[:, None] & col_mask[None, :],
            )
            tidx += NUM_SMS

        processed_tiles = tile_end


@_triton_jit
def _grouped_gemm_grouped_to_grouped_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    m_sizes_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tidx = tl.program_id(0).to(tl.int32)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    processed_tiles = tl.zeros((), dtype=tl.int32)
    m_end = tl.zeros((), dtype=tl.int32)

    for expert_idx in tl.range(0, NUM_EXPERTS):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size
        num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        tile_end = processed_tiles + num_m_tiles * num_n_tiles

        while tidx >= processed_tiles and tidx < tile_end:
            tile_idx = tidx - processed_tiles
            group_id = tile_idx // (GROUP_SIZE_M * num_n_tiles)
            first_m = group_id * GROUP_SIZE_M
            group_size_m = tl.minimum(num_m_tiles - first_m, GROUP_SIZE_M)
            within_group = tile_idx % (GROUP_SIZE_M * num_n_tiles)
            tile_m_idx = first_m + (within_group % group_size_m)
            tile_n_idx = within_group // group_size_m

            rows = tile_m_idx * BLOCK_SIZE_M + offs_m
            cols = tile_n_idx * BLOCK_SIZE_N + offs_n
            row_mask = rows < m_size
            col_mask = cols < N
            grouped_rows = m_start + rows

            total = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k_start in tl.range(0, K, BLOCK_SIZE_K):
                k_offsets = k_start + offs_k
                k_mask = k_offsets < K
                x_offsets = grouped_rows[:, None].to(tl.int64) * K + k_offsets[None, :]
                w_offsets = (
                    expert_idx.to(tl.int64) * (N * K)
                    + cols[:, None].to(tl.int64) * K
                    + k_offsets[None, :]
                )
                x_vals = tl.load(x_ptr + x_offsets, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                w_vals = tl.load(w_ptr + w_offsets, mask=col_mask[:, None] & k_mask[None, :], other=0.0)
                total += tl.dot(x_vals, tl.trans(w_vals))

            tl.store(
                y_ptr + grouped_rows[:, None].to(tl.int64) * N + cols[None, :],
                total,
                mask=row_mask[:, None] & col_mask[None, :],
            )
            tidx += NUM_SMS

        processed_tiles = tile_end


def _build_grouped_metadata(
    selected_experts: torch.Tensor,
    keep_mask: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, top_k = selected_experts.shape
    active_positions = torch.nonzero(keep_mask, as_tuple=False)
    active_experts = selected_experts[keep_mask].to(torch.int64)
    flat_slots = (active_positions[:, 0] * top_k + active_positions[:, 1]).to(torch.int32)
    sorted_experts, sort_idx = torch.sort(active_experts)
    gather_indices = flat_slots[sort_idx].contiguous()
    m_sizes = torch.bincount(sorted_experts, minlength=num_experts).to(torch.int32)
    return m_sizes.contiguous(), gather_indices, sorted_experts


def _resolve_triton_compute_tensors(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    """Align tensors to a single low-precision dtype accepted by Triton dot()."""
    output_dtype = hidden_states.dtype
    if gate_up_proj.dtype != down_proj.dtype:
        raise ValueError(
            "Triton fused experts require gate_up_proj and down_proj to share a dtype, got "
            f"{gate_up_proj.dtype} vs {down_proj.dtype}."
        )

    compute_dtype = gate_up_proj.dtype
    if compute_dtype not in {torch.float16, torch.bfloat16}:
        if hidden_states.dtype in {torch.float16, torch.bfloat16}:
            compute_dtype = hidden_states.dtype
        else:
            raise ValueError(
                "Triton fused experts require fp16/bf16 compute tensors, got "
                f"hidden_states={hidden_states.dtype}, weights={gate_up_proj.dtype}."
            )

    hidden_states_compute = hidden_states
    if hidden_states_compute.dtype != compute_dtype:
        hidden_states_compute = hidden_states_compute.to(compute_dtype)

    gate_up_proj_compute = gate_up_proj
    if gate_up_proj_compute.dtype != compute_dtype:
        gate_up_proj_compute = gate_up_proj_compute.to(compute_dtype)

    down_proj_compute = down_proj
    if down_proj_compute.dtype != compute_dtype:
        down_proj_compute = down_proj_compute.to(compute_dtype)

    return hidden_states_compute, gate_up_proj_compute, down_proj_compute, output_dtype


def compute_fused_expert_outputs_triton(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    group_m: int = 8,
    block_k: int = 64,
) -> tuple[torch.Tensor, TritonMoeStats]:
    """Compute per-slot expert outputs with Triton grouped GEMM."""
    _require_triton()
    if not hidden_states.is_cuda:
        raise ValueError("Triton score_only MoE backend requires CUDA tensors.")
    if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
        raise ValueError("Triton score_only MoE backend currently supports only fused Qwen3 experts.")
    if keep_mask.shape != selected_experts.shape:
        raise ValueError("keep_mask must match selected_experts shape.")

    active_mask = keep_mask.to(dtype=torch.bool)
    active_slots = int(active_mask.sum().item())
    tokens, top_k = selected_experts.shape
    hidden_dim = hidden_states.shape[-1]
    if active_slots == 0:
        return hidden_states.new_zeros((tokens, top_k, hidden_dim)), TritonMoeStats(0, 0)

    gate_up_proj = experts.gate_up_proj
    down_proj = experts.down_proj
    if not gate_up_proj.is_contiguous():
        gate_up_proj = gate_up_proj.contiguous()
    if not down_proj.is_contiguous():
        down_proj = down_proj.contiguous()

    num_experts, gate_up_dim, gate_hidden_dim = gate_up_proj.shape
    if gate_hidden_dim != hidden_dim:
        raise ValueError(f"gate_up_proj hidden dim {gate_hidden_dim} does not match hidden {hidden_dim}.")
    if gate_up_dim % 2 != 0:
        raise ValueError(f"gate_up_proj output dim must be even, got {gate_up_dim}.")
    intermediate_dim = gate_up_dim // 2
    if tuple(down_proj.shape) != (num_experts, hidden_dim, intermediate_dim):
        raise ValueError(
            "down_proj must have shape "
            f"({num_experts}, {hidden_dim}, {intermediate_dim}), got {tuple(down_proj.shape)}."
        )

    hidden_states_compute, gate_up_proj_compute, down_proj_compute, output_dtype = _resolve_triton_compute_tensors(
        hidden_states,
        gate_up_proj,
        down_proj,
    )

    m_sizes, gather_indices, _ = _build_grouped_metadata(
        selected_experts=selected_experts,
        keep_mask=active_mask,
        num_experts=num_experts,
    )
    hit_experts = int((m_sizes > 0).sum().item())

    gate_up_grouped = torch.empty(
        (active_slots, gate_up_dim),
        device=hidden_states.device,
        dtype=hidden_states_compute.dtype,
    )
    num_sms = torch.cuda.get_device_properties(hidden_states.device).multi_processor_count
    grid = (int(num_sms),)

    _grouped_gemm_token_to_grouped_kernel[grid](
        hidden_states_compute,
        gate_up_proj_compute,
        gate_up_grouped,
        gather_indices,
        m_sizes,
        N=gate_up_dim,
        K=hidden_dim,
        TOPK=top_k,
        NUM_EXPERTS=num_experts,
        NUM_SMS=int(num_sms),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        GROUP_SIZE_M=group_m,
        BLOCK_SIZE_K=block_k,
    )

    gate_branch, up_branch = gate_up_grouped.chunk(2, dim=-1)
    intermediate_grouped = F.silu(gate_branch) * up_branch

    grouped_outputs = torch.empty(
        (active_slots, hidden_dim),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    _grouped_gemm_grouped_to_grouped_kernel[grid](
        intermediate_grouped,
        down_proj_compute,
        grouped_outputs,
        m_sizes,
        N=hidden_dim,
        K=intermediate_dim,
        NUM_EXPERTS=num_experts,
        NUM_SMS=int(num_sms),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        GROUP_SIZE_M=group_m,
        BLOCK_SIZE_K=block_k,
    )

    expert_outputs = hidden_states.new_zeros((tokens, top_k, hidden_dim))
    token_idx = torch.div(gather_indices.to(torch.int64), top_k, rounding_mode="floor")
    slot_idx = torch.remainder(gather_indices.to(torch.int64), top_k)
    expert_outputs[token_idx, slot_idx] = grouped_outputs.to(output_dtype)
    return expert_outputs, TritonMoeStats(
        active_slots=active_slots,
        hit_experts=hit_experts,
    )


def compute_fused_experts_triton(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    keep_mask: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    group_m: int = 8,
    block_k: int = 64,
) -> tuple[torch.Tensor, TritonMoeStats]:
    """Compute Qwen3 fused expert outputs with Triton grouped GEMM.

    Returns the already gate-weighted and slot-summed final hidden states with
    the same dtype and shape as ``hidden_states``. The caller decides how to
    build ``keep_mask`` and ``routing_weights``; score_only is only the first
    integration point.
    """
    _require_triton()
    if not hidden_states.is_cuda:
        raise ValueError("Triton score_only MoE backend requires CUDA tensors.")
    if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
        raise ValueError("Triton score_only MoE backend currently supports only fused Qwen3 experts.")
    if keep_mask.shape != selected_experts.shape or routing_weights.shape != selected_experts.shape:
        raise ValueError("keep_mask and routing_weights must match selected_experts shape.")

    expert_outputs, stats = compute_fused_expert_outputs_triton(
        hidden_states,
        experts,
        selected_experts,
        keep_mask,
        block_m=block_m,
        block_n=block_n,
        group_m=group_m,
        block_k=block_k,
    )
    final_hidden = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
    return final_hidden, stats
