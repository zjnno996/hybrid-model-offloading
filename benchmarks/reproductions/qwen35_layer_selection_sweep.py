# SPDX-License-Identifier: Apache-2.0
"""Measure selective Full/Linear layer recovery on the real Mooncake store.

This is a layer-count sweep, not a synthetic bandwidth model.  Each case puts
the selected number of layer-sized descriptors in the standalone Mooncake CPU
segment and recovers them with ``batch_get_into_multi_buffers``.  The matching
recompute operation runs the same attention/GDN kernel count.  It intentionally
reports the attention-only lower bound; the full vLLM TTFT experiment remains
the final end-to-end validation.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from qwen35_hybrid_recovery_crossover import (
    make_full_attention_recompute_operation,
    make_linear_recompute_operation,
    measure_bidirectional_pinned_batch_ms,
    measure_cuda_ms,
    measure_h2d_batch_ms,
    measure_mooncake_concurrent_requests_ms,
    percentile,
    qwen_full_attention_kv_layout,
    qwen_gated_delta_net_state_layout,
)


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "/data/models/Qwen3.5-27B/config.json"
    ))
    parser.add_argument("--prefix-tokens", type=parse_int_list, default=[128, 900])
    parser.add_argument("--concurrency", type=parse_int_list, default=[1, 16])
    parser.add_argument("--linear-layer-counts", type=parse_int_list,
                        default=[1, 4, 8, 16, 32, 48])
    parser.add_argument("--full-layer-counts", type=parse_int_list,
                        default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--transport",
        choices=("pinned", "mooncake"),
        default="pinned",
        help="pinned measures direct CPU->GPU H2D; mooncake measures TCP recovery.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("invalid warmup/iterations")

    seed_everything(args.seed)
    config = json.loads(args.config.read_text())
    text_config = config.get("text_config", config)
    full_layout = qwen_full_attention_kv_layout(config)
    linear_layout = qwen_gated_delta_net_state_layout(
        config, temporal_dtype_bytes=4
    )
    full_bytes_per_layer = full_layout.bytes_for_tokens(1) // full_layout.num_full_attention_layers
    linear_bytes_per_layer = linear_layout.state_bytes_per_request // linear_layout.num_linear_layers

    rows: list[dict[str, object]] = []
    print(f"Qwen3.5 selective layer recovery sweep ({args.transport})")
    print(
        f"per-layer bytes: Full={full_bytes_per_layer / 2**20:.3f} MiB, "
        f"Linear={linear_bytes_per_layer / 2**20:.3f} MiB"
    )
    print(
        "type layers prefix conc bytes load_p50 load_p90 "
        "recompute_p50 recompute_p90 winner"
    )

    for prefix_tokens in args.prefix_tokens:
        for concurrency in args.concurrency:
            for attention_type, counts, max_layers in (
                ("full", args.full_layer_counts, full_layout.num_full_attention_layers),
                ("linear", args.linear_layer_counts, linear_layout.num_linear_layers),
            ):
                for layer_count in counts:
                    if layer_count > max_layers:
                        continue
                    if attention_type == "full":
                        bytes_per_request = full_bytes_per_layer * layer_count * prefix_tokens
                        transfer_fn = (
                            measure_h2d_batch_ms
                            if args.transport == "pinned"
                            else measure_mooncake_concurrent_requests_ms
                        )
                        transfer = transfer_fn(
                            bytes_per_request,
                            concurrency,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        )
                        one_layer = measure_cuda_ms(
                            make_full_attention_recompute_operation(
                                concurrency=concurrency,
                                prefix_tokens=prefix_tokens,
                                hidden_size=int(text_config["hidden_size"]),
                                num_attention_heads=int(text_config["num_attention_heads"]),
                                num_kv_heads=int(text_config["num_key_value_heads"]),
                                head_dim=int(text_config["head_dim"]),
                            ),
                            warmup=args.warmup, iterations=args.iterations,
                        )
                        recompute = [sample * layer_count for sample in one_layer]
                    else:
                        bytes_per_request = linear_bytes_per_layer * layer_count
                        transfer_fn = (
                            measure_h2d_batch_ms
                            if args.transport == "pinned"
                            else measure_mooncake_concurrent_requests_ms
                        )
                        transfer = transfer_fn(
                            bytes_per_request,
                            concurrency,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        )

                    d2h = h2d = roundtrip = None
                    if args.transport == "pinned":
                        d2h, h2d, roundtrip = measure_bidirectional_pinned_batch_ms(
                            bytes_per_request,
                            concurrency,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        )
                        recompute = measure_cuda_ms(
                            make_linear_recompute_operation(
                                linear_layout,
                                concurrency=concurrency,
                                prefix_tokens=prefix_tokens,
                                num_layers=layer_count,
                            ),
                            warmup=args.warmup, iterations=args.iterations,
                        )

                    load_p50 = statistics.median(transfer)
                    load_p90 = percentile(transfer, 0.9)
                    recompute_p50 = statistics.median(recompute)
                    recompute_p90 = percentile(recompute, 0.9)
                    winner = "load" if load_p50 <= recompute_p50 else "recompute"
                    row = {
                        "attention_type": attention_type,
                        "layer_count": layer_count,
                        "prefix_tokens": prefix_tokens,
                        "concurrency": concurrency,
                        "bytes_per_request": bytes_per_request,
                        "load_p50_ms": load_p50,
                        "load_p90_ms": load_p90,
                        "gpu_to_cpu_p50_ms": (
                            statistics.median(d2h) if d2h is not None else None
                        ),
                        "gpu_to_cpu_p90_ms": (
                            percentile(d2h, 0.9) if d2h is not None else None
                        ),
                        "cpu_to_gpu_p50_ms": (
                            statistics.median(h2d) if h2d is not None else None
                        ),
                        "cpu_to_gpu_p90_ms": (
                            percentile(h2d, 0.9) if h2d is not None else None
                        ),
                        "roundtrip_p50_ms": (
                            statistics.median(roundtrip)
                            if roundtrip is not None else None
                        ),
                        "roundtrip_p90_ms": (
                            percentile(roundtrip, 0.9)
                            if roundtrip is not None else None
                        ),
                        "recompute_p50_ms": recompute_p50,
                        "recompute_p90_ms": recompute_p90,
                        "winner": winner,
                        "seed": args.seed,
                    }
                    rows.append(row)
                    print(
                        f"{attention_type:6s} {layer_count:6d} {prefix_tokens:6d} "
                        f"{concurrency:4d} {bytes_per_request / 2**20:8.2f} "
                        f"{load_p50:9.3f} {load_p90:9.3f} "
                        f"{recompute_p50:13.3f} {recompute_p90:13.3f} {winner}"
                    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
