# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure when recovering Qwen3.5 Linear Attention state beats recomputing it.

The benchmark measures a single Gated DeltaNet layer for a batch of concurrent
requests, then also prints the linear extrapolation across all GDN layers in
the configured model.  Transfer includes the conv and recurrent state; compute
is the GDN recurrent-state update and read for the suffix.  It intentionally
does not claim to measure the Q/K/V projections or the rest of a transformer
block.

On hosts without RDMA, ``h2d`` is a real pinned-CPU-to-GPU lower-tier recovery
measurement.  It is a useful local baseline for Mooncake, but is not presented
as a Mooncake RDMA result.  Use ``--simulate-rdma-gbps`` only to obtain an
explicitly modelled wire-time reference for a target Mooncake fabric.

Example:
    python benchmarks/reproductions/mooncake_linear_recovery.py \\
      --config /data/models/Qwen3.5-27B/config.json \\
      --concurrency 1,4,16,64 --suffix-tokens 1,8,32
"""

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from vllm.v1.kv_offload.mooncake_linear_recovery import (
    GatedDeltaNetStateLayout,
    choose_recovery_path,
    qwen_gated_delta_net_state_layout,
)


DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "bf16": (torch.bfloat16, 2),
    "fp16": (torch.float16, 2),
    "fp32": (torch.float32, 4),
}


def parse_int_list(value: str) -> list[int]:
    values = [int(part) for part in value.split(",") if part]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a non-empty list of positive integers"
        )
    return values


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("cannot calculate percentile of no samples")
    index = (len(ordered) - 1) * quantile
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def measure_cuda_ms(
    operation: Callable[[], None], *, warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def make_gdn_state_operation(
    layout: GatedDeltaNetStateLayout,
    concurrency: int,
    suffix_tokens: int,
    dtype: torch.dtype,
) -> Callable[[], None]:
    """Create the stateful GDN update/read kernel used as recompute baseline."""
    heads = layout.num_value_heads
    state = torch.randn(
        (concurrency * heads, layout.value_head_dim, layout.key_head_dim),
        device="cuda",
        dtype=dtype,
    )
    key = torch.randn(
        (concurrency * heads, layout.key_head_dim), device="cuda", dtype=dtype
    )
    value = torch.randn(
        (concurrency * heads, layout.value_head_dim), device="cuda", dtype=dtype
    )
    query = torch.randn_like(key)
    decay = torch.full(
        (concurrency * heads, 1, 1), 0.99, device="cuda", dtype=dtype
    )

    @torch.inference_mode()
    def recompute() -> None:
        # This is the recurrent core of a Gated DeltaNet layer: state decay,
        # key/value incorporation, and query readout. Projection work is not
        # included, so this is a lower bound for full Linear Attention compute.
        for _ in range(suffix_tokens):
            state.mul_(decay)
            state.add_(torch.bmm(value.unsqueeze(2), key.unsqueeze(1)))
            torch.bmm(state, query.unsqueeze(2))

    return recompute


def make_transfer_operation(
    state_bytes_per_layer: int,
    concurrency: int,
    *,
    aggregate: bool,
) -> Callable[[], None]:
    """Create an actual pinned-host-to-device state recovery operation."""
    host = torch.empty(
        (concurrency, state_bytes_per_layer), dtype=torch.uint8, pin_memory=True
    )
    device = torch.empty_like(host, device="cuda")
    host.zero_()

    @torch.inference_mode()
    def transfer() -> None:
        if aggregate:
            device.copy_(host, non_blocking=True)
        else:
            # One descriptor per concurrently recovered request approximates a
            # Mooncake batch more closely than a synthetic monolithic memcpy.
            for request_idx in range(concurrency):
                device[request_idx].copy_(host[request_idx], non_blocking=True)

    return transfer


def format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**2):.2f} MiB"


def print_header(layout: GatedDeltaNetStateLayout, dtype_name: str) -> None:
    try:
        from mooncake.engine import TransferEngine  # noqa: F401

        mooncake_status = "available"
    except ImportError:
        mooncake_status = "not installed"
    print("Mooncake Linear Attention recovery crossover benchmark")
    print(f"Mooncake Transfer Engine: {mooncake_status}")
    print(
        "GDN state/layer: "
        f"{format_mib(layout.state_bytes_per_layer)} "
        f"({layout.temporal_state_elements:,} recurrent + "
        f"{layout.conv_state_elements:,} conv elements, {dtype_name})"
    )
    print(
        f"Linear layers: {layout.num_linear_layers}; state/request: "
        f"{format_mib(layout.state_bytes_per_request)}"
    )
    print(
        "transfer is measured H2D recovery; recompute is GDN recurrent "
        "update/read only (a lower bound for the complete Linear Attention layer)."
    )
    print()
    print(
        " conc  suffix  batch-state  transfer p50/p90   recompute p50/p90  "
        "winner       full-model p50 (transfer/recompute)"
    )


def run_case(
    layout: GatedDeltaNetStateLayout,
    *,
    dtype: torch.dtype,
    concurrency: int,
    suffix_tokens: int,
    warmup: int,
    iterations: int,
    aggregate_transfer: bool,
    simulate_rdma_gbps: float | None,
) -> None:
    transfer = make_transfer_operation(
        layout.state_bytes_per_layer, concurrency, aggregate=aggregate_transfer
    )
    recompute = make_gdn_state_operation(layout, concurrency, suffix_tokens, dtype)
    transfer_samples = measure_cuda_ms(transfer, warmup=warmup, iterations=iterations)
    recompute_samples = measure_cuda_ms(recompute, warmup=warmup, iterations=iterations)
    transfer_p50 = statistics.median(transfer_samples)
    recompute_p50 = statistics.median(recompute_samples)
    decision = choose_recovery_path(transfer_p50, recompute_p50)
    layers = layout.num_linear_layers
    print(
        f"{concurrency:5d}  {suffix_tokens:6d}  "
        f"{format_mib(layout.state_bytes_per_layer * concurrency):>11}  "
        f"{transfer_p50:7.3f}/{percentile(transfer_samples, 0.9):7.3f} ms  "
        f"{recompute_p50:7.3f}/{percentile(recompute_samples, 0.9):7.3f} ms  "
        f"{decision.path:9s} "
        f"{transfer_p50 * layers:8.2f}/{recompute_p50 * layers:8.2f} ms"
    )
    if simulate_rdma_gbps is not None:
        wire_ms = (
            layout.state_bytes_per_layer
            * concurrency
            / (simulate_rdma_gbps * 1_000_000_000)
            * 1_000
        )
        print(
            f"{'':5s}  {'':6s}  modelled Mooncake wire time at "
            f"{simulate_rdma_gbps:g} GB/s: {wire_ms:.3f} ms/layer "
            "(protocol and contention excluded)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/data/models/Qwen3.5-27B/config.json"),
        help="Qwen3.5 config.json; weights are not needed",
    )
    parser.add_argument("--concurrency", type=parse_int_list, default=[1, 4, 16, 64])
    parser.add_argument("--suffix-tokens", type=parse_int_list, default=[1, 8, 32])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--aggregate-transfer",
        action="store_true",
        help="issue one contiguous copy instead of one descriptor per request",
    )
    parser.add_argument(
        "--simulate-rdma-gbps",
        type=float,
        help="print a modelled RDMA wire time; this is not a measured result",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if args.simulate_rdma_gbps is not None and args.simulate_rdma_gbps <= 0:
        raise ValueError("simulate-rdma-gbps must be positive")

    _, dtype_bytes = DTYPES[args.dtype]
    layout = qwen_gated_delta_net_state_layout(
        json.loads(args.config.read_text()),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype_bytes=dtype_bytes,
    )
    dtype, _ = DTYPES[args.dtype]
    print_header(layout, args.dtype)
    for concurrency in args.concurrency:
        for suffix_tokens in args.suffix_tokens:
            run_case(
                layout,
                dtype=dtype,
                concurrency=concurrency,
                suffix_tokens=suffix_tokens,
                warmup=args.warmup,
                iterations=args.iterations,
                aggregate_transfer=args.aggregate_transfer,
                simulate_rdma_gbps=args.simulate_rdma_gbps,
            )


if __name__ == "__main__":
    main()
