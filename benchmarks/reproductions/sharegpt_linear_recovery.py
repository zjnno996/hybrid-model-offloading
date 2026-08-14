# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure Linear Attention recovery on real ShareGPT multi-turn suffixes.

Each sample represents a user turn after a prior assistant response.  The
suffix is tokenized with Qwen3.5's chat template, so it includes the turn's
role markers.  A transfer restores the entire Gated DeltaNet state for each
concurrent request; recompute runs the GDN recurrent state update/read only
over that request's actual new suffix.  Projection and full-attention work are
not included in recompute, making it a lower bound for total recomputation.

Download the raw dataset first (the repository's multi-turn README uses the
same source):
    curl -L -o /data/datasets/sharegpt/sharegpt_20230401_clean_lang_split.json \\
      https://huggingface.co/datasets/philschmid/sharegpt-raw/resolve/main/sharegpt_20230401_clean_lang_split.json
"""

import argparse
import json
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

try:
    from sharegpt_data import reservoir_sample
except ImportError:  # Imported as benchmarks.reproductions.*
    from benchmarks.reproductions.sharegpt_data import reservoir_sample
from vllm.v1.kv_offload.mooncake_linear_recovery import (
    GatedDeltaNetStateLayout,
    choose_recovery_path,
    qwen_gated_delta_net_state_layout,
)


def parse_int_list(value: str) -> list[int]:
    values = [int(part) for part in value.split(",") if part]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a non-empty list of positive integers"
        )
    return values


def percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("cannot calculate percentile of no samples")
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def measure_cuda_ms(
    operation: Callable[[], None], *, warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**2):.2f} MiB"


@dataclass(frozen=True)
class H2DQueueStats:
    """FIFO device-queue timing for one recovered state descriptor."""

    service_p50_ms: float
    wait_p50_ms: float
    wait_p99_ms: float
    completion_p50_ms: float
    completion_p99_ms: float


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Map the raw ShareGPT speaker labels to chat-template roles."""
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "chatgpt": "assistant",
        "bing": "assistant",
        "bard": "assistant",
        "system": "system",
    }
    messages = []
    for turn in record.get("conversations", []):
        role = role_map.get(turn.get("from"))
        content = turn.get("value")
        if role is not None and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    return messages


def extract_turn_suffix_lengths(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    record_indices: Sequence[int],
    max_suffix_tokens: int,
) -> list[int]:
    """Get chat-template token deltas for non-initial ShareGPT user turns."""
    suffix_lengths: list[int] = []
    for index in record_indices:
        history: list[dict[str, str]] = []
        for message in normalize_messages(records[index]):
            if (
                message["role"] == "user"
                and any(item["role"] == "user" for item in history)
                and any(item["role"] == "assistant" for item in history)
            ):
                previous = tokenizer.apply_chat_template(
                    history, tokenize=True, add_generation_prompt=False
                )
                current = tokenizer.apply_chat_template(
                    [*history, message], tokenize=True, add_generation_prompt=True
                )
                suffix_length = len(current) - len(previous)
                if 0 < suffix_length <= max_suffix_tokens:
                    suffix_lengths.append(suffix_length)
            history.append(message)
    return suffix_lengths


def measure_h2d_queue_ms(
    state_bytes_per_layer: int,
    concurrency: int,
    *,
    warmup: int,
    iterations: int,
) -> H2DQueueStats:
    """Measure FIFO wait and completion for simultaneous H2D recoveries.

    Each request gets an independent state descriptor on the same CUDA stream.
    Completion events therefore include copies queued in front of it. This is
    the local H2D analogue of a saturated Mooncake transfer queue; it excludes
    remote lookup, network and Mooncake worker-thread queueing.
    """
    host = torch.empty(
        (concurrency, state_bytes_per_layer), dtype=torch.uint8, pin_memory=True
    )
    device = torch.empty_like(host, device="cuda")
    host.zero_()

    @torch.inference_mode()
    def transfer_batch() -> None:
        for request_idx in range(concurrency):
            device[request_idx].copy_(host[request_idx], non_blocking=True)

    for _ in range(warmup):
        transfer_batch()
    torch.cuda.synchronize()

    service_times: list[float] = []
    wait_times: list[float] = []
    completion_times: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        completions = [torch.cuda.Event(enable_timing=True) for _ in range(concurrency)]
        start.record()
        for request_idx, completion in enumerate(completions):
            device[request_idx].copy_(host[request_idx], non_blocking=True)
            completion.record()
        completions[-1].synchronize()

        elapsed = [start.elapsed_time(completion) for completion in completions]
        completion_times.extend(elapsed)
        service_times.extend(
            [
                elapsed[0],
                *[
                    current - previous
                    for previous, current in zip(elapsed, elapsed[1:])
                ],
            ]
        )
        # In FIFO order, descriptor i starts after descriptor i - 1 completes.
        wait_times.extend([0.0, *elapsed[:-1]])

    return H2DQueueStats(
        service_p50_ms=statistics.median(service_times),
        wait_p50_ms=statistics.median(wait_times),
        wait_p99_ms=percentile(wait_times, 0.99),
        completion_p50_ms=statistics.median(completion_times),
        completion_p99_ms=percentile(completion_times, 0.99),
    )


def make_variable_gdn_recompute_operation(
    layout: GatedDeltaNetStateLayout, suffix_lengths: Sequence[int]
) -> Callable[[], None]:
    """Create a batched GDN recurrent operation with real variable suffixes."""
    lengths = sorted(suffix_lengths, reverse=True)
    concurrency = len(lengths)
    heads = layout.num_value_heads
    state = torch.randn(
        (concurrency * heads, layout.value_head_dim, layout.key_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        (concurrency * heads, layout.key_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        (concurrency * heads, layout.value_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    query = torch.randn_like(key)
    decay = torch.full(
        (concurrency * heads, 1, 1), 0.99, device="cuda", dtype=torch.bfloat16
    )
    active_counts = [
        sum(length >= step for length in lengths)
        for step in range(1, max(lengths) + 1)
    ]

    @torch.inference_mode()
    def recompute() -> None:
        for active_count in active_counts:
            active_heads = active_count * heads
            active_state = state[:active_heads]
            active_key = key[:active_heads]
            active_value = value[:active_heads]
            active_query = query[:active_heads]
            active_state.mul_(decay[:active_heads])
            active_state.add_(
                torch.bmm(active_value.unsqueeze(2), active_key.unsqueeze(1))
            )
            torch.bmm(active_state, active_query.unsqueeze(2))

    return recompute


def sample_batches(
    suffix_lengths: Sequence[int],
    *,
    concurrency: int,
    num_batches: int,
    rng: random.Random,
) -> list[list[int]]:
    if len(suffix_lengths) < concurrency:
        raise ValueError(f"need {concurrency} user turns; found {len(suffix_lengths)}")
    return [rng.sample(list(suffix_lengths), concurrency) for _ in range(num_batches)]


def run_concurrency(
    layout: GatedDeltaNetStateLayout,
    suffix_lengths: Sequence[int],
    *,
    concurrency: int,
    transfer_state_bytes: int,
    recompute_layers: int,
    num_batches: int,
    warmup: int,
    iterations: int,
    rng: random.Random,
) -> None:
    h2d_queue_stats = []
    recompute_p50s = []
    batch_token_p50s = []
    batch_token_p90s = []
    for batch in sample_batches(
        suffix_lengths, concurrency=concurrency, num_batches=num_batches, rng=rng
    ):
        h2d_queue_stats.append(
            measure_h2d_queue_ms(
                transfer_state_bytes,
                concurrency,
                warmup=warmup,
                iterations=iterations,
            )
        )
        recompute_samples = measure_cuda_ms(
            make_variable_gdn_recompute_operation(layout, batch),
            warmup=warmup,
            iterations=iterations,
        )
        recompute_p50s.append(statistics.median(recompute_samples))
        batch_token_p50s.append(statistics.median(batch))
        batch_token_p90s.append(percentile(batch, 0.9))

    h2d_service_ms = statistics.median(
        stats.service_p50_ms for stats in h2d_queue_stats
    )
    h2d_wait_p50_ms = statistics.median(
        stats.wait_p50_ms for stats in h2d_queue_stats
    )
    h2d_wait_p99_ms = statistics.median(
        stats.wait_p99_ms for stats in h2d_queue_stats
    )
    h2d_completion_p50_ms = statistics.median(
        stats.completion_p50_ms for stats in h2d_queue_stats
    )
    h2d_completion_p99_ms = statistics.median(
        stats.completion_p99_ms for stats in h2d_queue_stats
    )
    recompute_ms = statistics.median(recompute_p50s) * recompute_layers
    decision = choose_recovery_path(h2d_completion_p50_ms, recompute_ms)
    print(
        f"{concurrency:5d}  {statistics.median(batch_token_p50s):8.1f}/"
        f"{statistics.median(batch_token_p90s):6.1f}  "
        f"{h2d_service_ms:6.3f}  {h2d_wait_p50_ms:6.3f}/{h2d_wait_p99_ms:6.3f}  "
        f"{h2d_completion_p50_ms:6.3f}/{h2d_completion_p99_ms:6.3f}  "
        f"{recompute_ms:9.3f}  "
        f"{decision.path:9s} {decision.speedup:6.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "/data/datasets/sharegpt/sharegpt_20230401_clean_lang_split.json"
        ),
    )
    parser.add_argument("--model", default="/data/models/Qwen3.5-27B")
    parser.add_argument("--concurrency", type=parse_int_list, default=[1, 4, 16, 64])
    parser.add_argument(
        "--sampled-records",
        type=int,
        default=128,
        help="Bounded reservoir sample; the full ShareGPT JSON is streamed.",
    )
    parser.add_argument("--max-suffix-tokens", type=int, default=256)
    parser.add_argument(
        "--transfer-state-scope",
        choices=["layer", "request"],
        default="request",
        help=(
            "transfer one GDN layer state or the coalesced full state for a "
            "request; request is the materialization lower bound"
        ),
    )
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")

    rng = random.Random(args.seed)
    print(f"Loading ShareGPT data from {args.dataset_path}")
    sampled = reservoir_sample(
        args.dataset_path, args.sampled_records, seed=args.seed
    )
    record_count = len(sampled)
    # ``extract_turn_suffix_lengths`` indexes the in-memory sample, not the
    # original 126k-record dataset.
    record_indices = list(range(record_count))
    records = [record for _, record in sampled]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    suffix_lengths = extract_turn_suffix_lengths(
        records,
        tokenizer,
        record_indices=record_indices,
        max_suffix_tokens=args.max_suffix_tokens,
    )
    if not suffix_lengths:
        raise ValueError("no eligible multi-turn ShareGPT suffixes were found")

    config = json.loads((Path(args.model) / "config.json").read_text())
    layout = qwen_gated_delta_net_state_layout(config)
    transfer_state_bytes = (
        layout.state_bytes_per_layer
        if args.transfer_state_scope == "layer"
        else layout.state_bytes_per_request
    )
    recompute_layers = (
        1 if args.transfer_state_scope == "layer" else layout.num_linear_layers
    )
    print(
        f"Extracted {len(suffix_lengths):,} eligible user turns from "
        f"{record_count:,} sampled ShareGPT records; token p50/p90/p99 = "
        f"{statistics.median(suffix_lengths):.0f}/"
        f"{percentile(suffix_lengths, 0.9):.0f}/"
        f"{percentile(suffix_lengths, 0.99):.0f}."
    )
    print(
        f"GDN state/layer: {format_mib(layout.state_bytes_per_layer)}; "
        f"linear layers/model: {layout.num_linear_layers}."
    )
    print(
        f"H2D scope: {args.transfer_state_scope}, descriptor size "
        f"{format_mib(transfer_state_bytes)}; recompute covers "
        f"{recompute_layers} layer(s)."
    )
    print(
        "H2D metrics are per request and include FIFO queueing of the "
        "simultaneously admitted descriptors. Request scope assumes that the "
        "48 layer states are coalesced, so it is a transfer lower bound."
    )
    print(
        " conc  suffix p50/p90  service  H2D-wait p50/p99  H2D-e2e p50/p99  "
        "recompute  winner     speedup"
    )
    for concurrency in args.concurrency:
        run_concurrency(
            layout,
            suffix_lengths,
            concurrency=concurrency,
            transfer_state_bytes=transfer_state_bytes,
            recompute_layers=recompute_layers,
            num_batches=args.num_batches,
            warmup=args.warmup,
            iterations=args.iterations,
            rng=rng,
        )


if __name__ == "__main__":
    main()
