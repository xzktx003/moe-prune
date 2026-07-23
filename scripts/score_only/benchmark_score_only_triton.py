from __future__ import annotations

import argparse
import json
import statistics
from types import SimpleNamespace

import torch

from moe_prune.code.src.runtime_pruner import compute_expert_outputs, renorm_gate_after_pruning
from moe_prune.code.src.triton_group_gemm import compute_fused_experts_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark score_only torch vs Triton MoE expert path.")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--avg-active-experts", type=float, default=3.2)
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=0.02,
        help="Random expert weight stddev; Qwen-style checkpoints are much smaller than N(0, 1).",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def cuda_time_ms(fn, *, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p90_ms": statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times),
        "raw_ms": times,
    }


def build_keep_mask(tokens: int, top_k: int, avg_active_experts: float, device) -> torch.Tensor:
    if top_k <= 1:
        keep_prob = 1.0
    else:
        keep_prob = max(0.0, min(1.0, (float(avg_active_experts) - 1.0) / float(top_k - 1)))
    keep_mask = torch.rand(tokens, top_k, device=device) < keep_prob
    top1 = torch.randint(0, top_k, (tokens, 1), device=device)
    keep_mask.scatter_(1, top1, True)
    return keep_mask


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton score_only benchmark.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    experts = SimpleNamespace(
        gate_up_proj=torch.randn(
            args.num_experts,
            args.intermediate_size * 2,
            args.hidden_size,
            device=device,
            dtype=dtype,
        )
        * args.weight_scale,
        down_proj=torch.randn(
            args.num_experts,
            args.hidden_size,
            args.intermediate_size,
            device=device,
            dtype=dtype,
        )
        * args.weight_scale,
        act_fn=torch.nn.SiLU(),
    )
    hidden_states = torch.randn(args.tokens, args.hidden_size, device=device, dtype=dtype)
    selected_experts = torch.randint(
        0,
        args.num_experts,
        (args.tokens, args.top_k),
        device=device,
        dtype=torch.long,
    )
    keep_mask = build_keep_mask(args.tokens, args.top_k, args.avg_active_experts, device)
    routing_weights = torch.rand(args.tokens, args.top_k, device=device, dtype=dtype)
    gate_kept = renorm_gate_after_pruning(routing_weights, keep_mask)

    def torch_path() -> torch.Tensor:
        expert_outputs = compute_expert_outputs(
            hidden_states,
            experts,
            selected_experts,
            keep_mask=keep_mask,
        )
        return (gate_kept.unsqueeze(-1) * expert_outputs).sum(dim=1)

    def triton_path() -> torch.Tensor:
        output, _ = compute_fused_experts_triton(
            hidden_states,
            experts,
            selected_experts,
            keep_mask,
            gate_kept,
        )
        return output

    torch_output = torch_path()
    triton_output = triton_path()
    torch.cuda.synchronize()
    max_abs = float((torch_output - triton_output).abs().max().item())
    mean_abs = float((torch_output - triton_output).abs().float().mean().item())
    max_ref = float(torch_output.abs().max().item())
    mean_ref = float(torch_output.abs().float().mean().item())
    allclose = bool(torch.allclose(torch_output, triton_output, atol=5e-2, rtol=5e-2))

    torch_time = cuda_time_ms(torch_path, warmup=args.warmup, iters=args.iters)
    triton_time = cuda_time_ms(triton_path, warmup=args.warmup, iters=args.iters)
    speedup = torch_time["mean_ms"] / triton_time["mean_ms"]

    active_slots = int(keep_mask.sum().item())
    result = {
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "dtype": args.dtype,
        "active_slots": active_slots,
        "avg_active_experts_actual": active_slots / args.tokens,
        "torch_mean_ms": torch_time["mean_ms"],
        "triton_mean_ms": triton_time["mean_ms"],
        "speedup": speedup,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_ref_abs": max_ref,
        "mean_ref_abs": mean_ref,
        "max_abs_diff_over_max_ref": max_abs / max(max_ref, 1e-12),
        "mean_abs_diff_over_mean_ref": mean_abs / max(mean_ref, 1e-12),
        "allclose_atol_rtol_5e_2": allclose,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
