# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Qwen3.5 Full-Attention and Linear-state cache recovery costs.

The benchmark uses the aligned 784-token prefix page used by Qwen3.5-27B in
vLLM. It measures real Mooncake CPU-to-GPU recovery separately for:

* Full Attention K/V blocks across every Full-Attention layer; and
* Gated DeltaNet conv plus recurrent state across every Linear layer.

It compares those transfers with actual vLLM GDN replay kernels:
Full Attention runs Q/K/V projection plus causal attention, while Linear
Attention replays its Gated DeltaNet recurrence. Neither includes MLP or
residual/norm work, so the reported compute times are favorable to recompute.
The Mooncake mode uses the running standalone CPU segment at
``127.0.0.1:50053`` and the real ``batch_get_into_multi_buffers`` API. It is a
group-level framework measurement, not a synthetic bandwidth estimate.
"""

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.v1.kv_offload.mooncake_linear_recovery import (
    FullAttentionKVLayout,
    GatedDeltaNetStateLayout,
    qwen_full_attention_kv_layout,
    qwen_gated_delta_net_state_layout,
)


def parse_int_list(value: str) -> list[int]:
    values = [int(part) for part in value.split(",") if part]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a non-empty list of positive integers"
        )
    return values


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**2):.2f} MiB"


def seed_everything(seed: int) -> None:
    """Fix benchmark inputs and CUDA RNGs; timing still needs repetitions."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def measure_h2d_batch_ms(
    bytes_per_request: int,
    concurrency: int,
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Measure all concurrent request descriptors completing on one stream."""
    host = torch.empty(
        (concurrency, bytes_per_request), dtype=torch.uint8, pin_memory=True
    )
    device = torch.empty_like(host, device="cuda")
    host.zero_()
    for _ in range(warmup):
        for request_idx in range(concurrency):
            device[request_idx].copy_(host[request_idx], non_blocking=True)
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for request_idx in range(concurrency):
            device[request_idx].copy_(host[request_idx], non_blocking=True)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def measure_bidirectional_pinned_batch_ms(
    bytes_per_request: int,
    concurrency: int,
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[float], list[float], list[float]]:
    """Measure GPU->CPU offload, CPU->GPU restore, and their round trip.

    The host buffers are pinned, matching vLLM's native CPU offloading path.
    Each request is queued on the same CUDA stream in the order
    ``GPU->CPU`` then ``CPU->GPU``.  This captures DMA serialization for a
    batch, while the native vLLM end-to-end test remains responsible for
    prefix matching and scheduler/connector queueing.
    """
    host = torch.empty(
        (concurrency, bytes_per_request), dtype=torch.uint8, pin_memory=True
    )
    device = torch.empty_like(host, device="cuda")
    device.zero_()
    host.zero_()
    for _ in range(warmup):
        for request_idx in range(concurrency):
            device[request_idx].copy_(host[request_idx], non_blocking=True)
            host[request_idx].copy_(device[request_idx], non_blocking=True)
    torch.cuda.synchronize()

    d2h_samples: list[float] = []
    h2d_samples: list[float] = []
    roundtrip_samples: list[float] = []
    for _ in range(iterations):
        d2h_start = torch.cuda.Event(enable_timing=True)
        d2h_end = torch.cuda.Event(enable_timing=True)
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        roundtrip_start = torch.cuda.Event(enable_timing=True)
        roundtrip_end = torch.cuda.Event(enable_timing=True)

        roundtrip_start.record()
        d2h_start.record()
        for request_idx in range(concurrency):
            host[request_idx].copy_(device[request_idx], non_blocking=True)
        d2h_end.record()
        h2d_start.record()
        for request_idx in range(concurrency):
            device[request_idx].copy_(host[request_idx], non_blocking=True)
        h2d_end.record()
        roundtrip_end.record()
        roundtrip_end.synchronize()

        d2h_samples.append(d2h_start.elapsed_time(d2h_end))
        h2d_samples.append(h2d_start.elapsed_time(h2d_end))
        roundtrip_samples.append(roundtrip_start.elapsed_time(roundtrip_end))
    return d2h_samples, h2d_samples, roundtrip_samples


def measure_mooncake_get_batch_ms(
    bytes_per_request: int,
    concurrency: int,
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Measure real Mooncake CPU-segment to GPU-buffer recovery.

    Objects are put into the existing standalone CPU segment first. Each
    timed operation then calls Mooncake's native batched get API into CUDA
    buffers, so the samples include the actual TCP/RPC and transfer path.
    """
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    store = MooncakeDistributedStore()
    result = store.setup(
        {
            "local_hostname": "127.0.0.1",
            "metadata_server": "P2PHANDSHAKE",
            "global_segment_size": 0,
            "local_buffer_size": 256 * 1024 * 1024,
            "protocol": "tcp",
            "rdma_devices": "",
            "master_server_addr": "127.0.0.1:50051",
        }
    )
    if result != 0:
        raise RuntimeError(f"Mooncake store setup failed with status {result}")

    host = [
        torch.empty(bytes_per_request, dtype=torch.uint8, pin_memory=True)
        for _ in range(concurrency)
    ]
    device = [torch.empty(bytes_per_request, dtype=torch.uint8, device="cuda")
              for _ in range(concurrency)]
    keys = [
        f"qwen35-direct-linear-recovery-{bytes_per_request}-{concurrency}-{i}"
        for i in range(concurrency)
    ]
    replicate = ReplicateConfig()
    replicate.preferred_segment = "127.0.0.1:50053"
    try:
        put_result = store.batch_put_from_multi_buffers(
            keys,
            [[item.data_ptr()] for item in host],
            [[bytes_per_request] for _ in host],
            replicate,
        )
        if any(status < 0 for status in put_result):
            raise RuntimeError(f"Mooncake CPU-cache put failed: {put_result}")
        if store.batch_is_exist(keys) != [1] * concurrency:
            raise RuntimeError("Mooncake CPU-cache keys were not visible after put")

        addresses = [[item.data_ptr()] for item in device]
        sizes = [[bytes_per_request] for _ in device]

        def get_batch() -> None:
            statuses = store.batch_get_into_multi_buffers(keys, addresses, sizes)
            if any(status < 0 for status in statuses):
                raise RuntimeError(f"Mooncake CPU-cache get failed: {statuses}")

        for _ in range(warmup):
            get_batch()
        torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            get_batch()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return samples
    finally:
        # Do not consume the shared 32GB CPU segment across cases.
        store.batch_remove(keys, force=True)


def measure_mooncake_concurrent_requests_ms(
    bytes_per_request: int,
    concurrency: int,
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Measure matching + transfer for concurrent independent requests."""
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    store = MooncakeDistributedStore()
    result = store.setup(
        {
            "local_hostname": "127.0.0.1",
            "metadata_server": "P2PHANDSHAKE",
            "global_segment_size": 0,
            "local_buffer_size": 256 * 1024 * 1024,
            "protocol": "tcp",
            "rdma_devices": "",
            "master_server_addr": "127.0.0.1:50051",
        }
    )
    if result != 0:
        raise RuntimeError(f"Mooncake store setup failed with status {result}")

    host = [
        torch.empty(bytes_per_request, dtype=torch.uint8, pin_memory=True)
        for _ in range(concurrency)
    ]
    device = [torch.empty(bytes_per_request, dtype=torch.uint8, device="cuda")
              for _ in range(concurrency)]
    keys = [
        f"qwen35-direct-linear-request-{bytes_per_request}-{concurrency}-{i}"
        for i in range(concurrency)
    ]
    replicate = ReplicateConfig()
    replicate.preferred_segment = "127.0.0.1:50053"
    try:
        put_result = store.batch_put_from_multi_buffers(
            keys,
            [[item.data_ptr()] for item in host],
            [[bytes_per_request] for _ in host],
            replicate,
        )
        if any(status < 0 for status in put_result):
            raise RuntimeError(f"Mooncake CPU-cache put failed: {put_result}")

        def one_request(index: int) -> float:
            started = time.perf_counter()
            if store.batch_is_exist([keys[index]]) != [1]:
                raise RuntimeError("Mooncake matching failed")
            status = store.batch_get_into_multi_buffers(
                [keys[index]], [[device[index].data_ptr()]], [[bytes_per_request]]
            )
            if status[0] < 0:
                raise RuntimeError(f"Mooncake get failed: {status}")
            torch.cuda.synchronize()
            return (time.perf_counter() - started) * 1000

        def one_round() -> float:
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(one_request, range(concurrency)))
            return (time.perf_counter() - started) * 1000

        for _ in range(warmup):
            one_round()
        return [one_round() for _ in range(iterations)]
    finally:
        store.batch_remove(keys, force=True)


def make_full_attention_recompute_operation(
    *,
    concurrency: int,
    prefix_tokens: int,
    hidden_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Callable[[], None]:
    """One Full-Attention layer's Q/K/V and causal-attention lower bound."""
    dtype = torch.bfloat16
    hidden = torch.randn(
        (concurrency, prefix_tokens, hidden_size), device="cuda", dtype=dtype
    )
    q_weight = torch.randn(
        (num_attention_heads * head_dim, hidden_size), device="cuda", dtype=dtype
    )
    kv_weight = torch.randn(
        (2 * num_kv_heads * head_dim, hidden_size), device="cuda", dtype=dtype
    )
    kv_width = num_kv_heads * head_dim

    @torch.inference_mode()
    def recompute() -> None:
        query = F.linear(hidden, q_weight).view(
            concurrency, prefix_tokens, num_attention_heads, head_dim
        )
        key_value = F.linear(hidden, kv_weight)
        key, value = key_value.split(kv_width, dim=-1)
        key = key.view(
            concurrency, prefix_tokens, num_kv_heads, head_dim
        ).transpose(1, 2)
        value = value.view(
            concurrency, prefix_tokens, num_kv_heads, head_dim
        ).transpose(1, 2)
        query = query.transpose(1, 2)
        # Repeat KV heads to match Q heads. This is the Qwen GQA layout.
        repeat = num_attention_heads // num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        F.scaled_dot_product_attention(query, key, value, is_causal=True)

    return recompute


def make_linear_recompute_operation(
    layout: GatedDeltaNetStateLayout,
    *,
    concurrency: int,
    prefix_tokens: int,
    num_layers: int | None = None,
) -> Callable[[], None]:
    """Actual chunked-prefill recurrent-state replay for GDN layers."""
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule

    heads = layout.num_value_heads
    state = torch.randn(
        (concurrency, heads, layout.value_head_dim, layout.key_head_dim),
        device="cuda",
        dtype=torch.float32,
    )
    query = torch.randn(
        (concurrency, prefix_tokens, heads, layout.key_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn_like(query)
    value = torch.randn(
        (concurrency, prefix_tokens, heads, layout.value_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = torch.full(
        (concurrency, prefix_tokens, heads),
        -0.01,
        device="cuda",
        dtype=torch.bfloat16,
    )
    beta = torch.full(
        (concurrency, prefix_tokens, heads),
        0.5,
        device="cuda",
        dtype=torch.bfloat16,
    )
    layers = num_layers or layout.num_linear_layers

    @torch.inference_mode()
    def recompute() -> None:
        for _ in range(layers):
            chunk_gated_delta_rule(
                query,
                key,
                value,
                gate,
                beta,
                initial_state=state,
                output_final_state=True,
            )

    return recompute


def print_case(
    name: str,
    bytes_per_request: int,
    transfer_samples: Sequence[float],
    recompute_samples: Sequence[float],
) -> None:
    transfer = statistics.median(transfer_samples)
    recompute = statistics.median(recompute_samples)
    winner = "transfer" if transfer <= recompute else "recompute"
    ratio = max(transfer, recompute) / min(transfer, recompute)
    print(
        f"{name:14s} {format_mib(bytes_per_request):>11s}  "
        f"{transfer:8.3f}/{percentile(transfer_samples, 0.9):8.3f}  "
        f"{recompute:8.3f}/{percentile(recompute_samples, 0.9):8.3f}  "
        f"{winner:9s} {ratio:5.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("/data/models/Qwen3.5-27B/config.json")
    )
    parser.add_argument("--concurrency", type=parse_int_list, default=[1, 4, 16, 64])
    parser.add_argument("--prefix-tokens", type=int, default=784)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--transport",
        choices=("mooncake", "pinned"),
        default="mooncake",
        help="Use the real Mooncake CPU segment or a local pinned-memory bound.",
    )
    parser.add_argument(
        "--mooncake-request-mode",
        choices=("batch", "concurrent"),
        default="concurrent",
        help="Batch one get call or issue one matching/get call per request.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")
    if args.prefix_tokens < 1 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("prefix-tokens must be positive; invalid warmup/iterations")
    seed_everything(args.seed)

    config = json.loads(args.config.read_text())
    text_config = config.get("text_config", config)
    full_layout: FullAttentionKVLayout = qwen_full_attention_kv_layout(config)
    linear_layout = qwen_gated_delta_net_state_layout(
        config, temporal_dtype_bytes=4
    )
    full_bytes = full_layout.bytes_for_tokens(args.prefix_tokens)
    linear_bytes = linear_layout.state_bytes_per_request
    print("Qwen3.5 hybrid cache recovery microbenchmark")
    print(
        f"aligned prefix page: {args.prefix_tokens} tokens; Full layers: "
        f"{full_layout.num_full_attention_layers}; Linear layers: "
        f"{linear_layout.num_linear_layers}"
    )
    print(
        f"transport={args.transport}; compute replays all "
        f"{linear_layout.num_linear_layers} GDN layers and excludes "
        "MLP/norm/residual work."
    )
    print(
        "conc  type             bytes/request  transfer p50/p90 ms  "
        "recompute p50/p90 ms  winner    ratio"
    )
    if args.transport == "mooncake":
        transfer_fn = (
            measure_mooncake_concurrent_requests_ms
            if args.mooncake_request_mode == "concurrent"
            else measure_mooncake_get_batch_ms
        )
    else:
        transfer_fn = measure_h2d_batch_ms
    for concurrency in args.concurrency:
        full_transfer = transfer_fn(
            full_bytes, concurrency, warmup=args.warmup, iterations=args.iterations
        )
        full_one_layer = measure_cuda_ms(
            make_full_attention_recompute_operation(
                concurrency=concurrency,
                prefix_tokens=args.prefix_tokens,
                hidden_size=int(text_config["hidden_size"]),
                num_attention_heads=int(text_config["num_attention_heads"]),
                num_kv_heads=int(text_config["num_key_value_heads"]),
                head_dim=int(text_config["head_dim"]),
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        full_recompute = [
            sample * full_layout.num_full_attention_layers
            for sample in full_one_layer
        ]
        linear_transfer = transfer_fn(
            linear_bytes,
            concurrency,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        linear_one_layer = measure_cuda_ms(
            make_linear_recompute_operation(
                linear_layout,
                concurrency=concurrency,
                prefix_tokens=args.prefix_tokens,
                num_layers=linear_layout.num_linear_layers,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        # ``make_linear_recompute_operation`` already replays every linear
        # layer.  Keep these samples as the complete GDN recovery cost;
        # multiplying here would count the 48 layers twice.
        linear_recompute = linear_one_layer
        print(f"{concurrency:4d}  ", end="")
        print_case("full-attention", full_bytes, full_transfer, full_recompute)
        print(f"{'':4s}  ", end="")
        print_case("linear-state", linear_bytes, linear_transfer, linear_recompute)


if __name__ == "__main__":
    main()
