# SPDX-License-Identifier: Apache-2.0
"""Run a deterministic prefix-length x concurrency TTFT matrix.

The vLLM server is intentionally managed outside this script: start it with
the desired ``VLLM_MOONCAKE_HYBRID_POLICY`` (or without a connector for P2),
then run ``warmup`` once and ``load`` after restarting vLLM.  The script keeps
the workload identical across P1--P4 and records one JSON object per cell.
With ``--metrics-url`` it also records native CPU-offload counter deltas per
cell, which verifies that a reported TTFT actually used GPU->CPU/CPU->GPU
offloading rather than a GPU-resident prefix hit.

Policy mapping:

* P1: fetch Full and Linear groups (the all-load endpoint).
* P2: no connector; replay both groups locally.
* P3: fetch only Full groups; replay the Linear prefix.
* P4: fetch only Linear groups; replay the Full-Attention prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

try:
    from sharegpt_data import iter_records
except ImportError:  # Imported as benchmarks.reproductions.*
    from benchmarks.reproductions.sharegpt_data import iter_records


ROLE_NAMES = {
    "human": "User",
    "user": "User",
    "gpt": "Assistant",
    "chatgpt": "Assistant",
    "bing": "Assistant",
    "bard": "Assistant",
}


@dataclass(frozen=True)
class Prefix:
    record_index: int
    target_tokens: int
    actual_tokens: int
    prompt: str


def parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def conversation_text(record: dict[str, Any]) -> str:
    turns: list[str] = []
    for turn in record.get("conversations", []):
        role = ROLE_NAMES.get(turn.get("from"))
        value = turn.get("value")
        if role and isinstance(value, str) and value.strip():
            turns.append(f"{role}: {value}")
    return "\n\n".join(turns)


def make_prefixes(
    records: Any,
    tokenizer: Any,
    *,
    target_tokens: int,
    count: int,
    salt: str,
) -> list[Prefix]:
    salt_ids = tokenizer.encode(salt, add_special_tokens=False)
    prefixes: list[Prefix] = []
    for record_index, record in enumerate(records):
        token_ids = tokenizer.encode(
            conversation_text(record), add_special_tokens=False
        )
        if len(token_ids) + len(salt_ids) < target_tokens:
            continue
        prompt_ids = salt_ids + token_ids[: target_tokens - len(salt_ids)]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        actual = len(tokenizer.encode(prompt, add_special_tokens=False))
        prefixes.append(Prefix(record_index, target_tokens, actual, prompt))
        if len(prefixes) == count:
            return prefixes
    raise ValueError(
        f"could not find {count} ShareGPT prefixes at {target_tokens} tokens"
    )


def issue_request(server: str, model: str, prefix: Prefix, seed: int) -> float:
    started = time.perf_counter()
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        f"{server}/v1/completions",
        json={
            "model": model,
            "prompt": prefix.prompt,
            "max_tokens": 1,
            "temperature": 0,
            "seed": seed,
        },
        timeout=600,
    )
    response.raise_for_status()
    return (time.perf_counter() - started) * 1000


def read_offload_metrics(metrics_url: str | None) -> dict[str, float]:
    """Read native vLLM offload counters for one TTFT cell.

    The counters are optional because Mooncake-only deployments may not expose
    the native ``kv_offload_*`` series. Missing metrics are represented by an
    empty dictionary and never invalidate a TTFT measurement.
    """
    if not metrics_url:
        return {}
    session = requests.Session()
    session.trust_env = False
    response = session.get(metrics_url, timeout=10)
    response.raise_for_status()
    values: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line.startswith("vllm:kv_offload_total_") or "{" not in line:
            continue
        metric, raw_value = line.rsplit(" ", 1)
        if 'transfer_type="' not in metric:
            continue
        transfer_type = metric.split('transfer_type="', 1)[1].split('"', 1)[0]
        if metric.startswith("vllm:kv_offload_total_bytes_total"):
            values[f"{transfer_type}_bytes"] = float(raw_value)
        elif metric.startswith("vllm:kv_offload_total_time_total"):
            values[f"{transfer_type}_time"] = float(raw_value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("warmup", "load"))
    parser.add_argument("--policy", required=True, help="P1, P2, P3, or P4")
    parser.add_argument("--dataset-path", type=Path, default=Path(
        "/data/datasets/sharegpt/sharegpt_20230401_clean_lang_split.json"
    ))
    parser.add_argument("--tokenizer", default="/data/models/Qwen3.5-27B")
    parser.add_argument("--model", default="Qwen3.5-27B")
    parser.add_argument("--server", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--metrics-url",
        default=None,
        help="Optional vLLM /metrics URL; records per-cell offload counter deltas.",
    )
    parser.add_argument("--prefix-lengths", type=parse_ints, default=[256, 512, 768, 900])
    parser.add_argument("--concurrencies", type=parse_ints, default=[1, 4, 16, 32, 64])
    parser.add_argument("--requests-per-cell", type=int, default=16)
    parser.add_argument(
        "--unique-prefixes",
        type=int,
        default=16,
        help="Number of distinct warmed prefixes per length; load requests may repeat them.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/root/qwen35_prefix_matrix.jsonl"))
    args = parser.parse_args()
    if args.requests_per_cell < 1 or args.unique_prefixes < 1:
        raise ValueError("requests-per-cell and unique-prefixes must be positive")
    if args.mode == "warmup" and args.policy not in {"P1", "P3", "P4"}:
        raise ValueError(
            "warmup must use P1 (all groups), P3 (Full groups), or P4 (Linear/Mamba groups)"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prefixes_by_length: dict[int, list[Prefix]] = {}
    for length in args.prefix_lengths:
        prefixes_by_length[length] = make_prefixes(
            iter_records(args.dataset_path), tokenizer,
            target_tokens=length,
            count=args.unique_prefixes,
            salt=f"Agent matrix prefix={length}:\n",
        )

    if args.mode == "warmup":
        for length, prefixes in prefixes_by_length.items():
            for prefix in prefixes:
                issue_request(args.server, args.model, prefix, args.seed)
            print(f"warmup prefix_tokens={length} requests={len(prefixes)}")
        print(
            "warmup complete; for native CPU offload keep this vLLM process alive "
            "and evict GPU prefixes before load; Mooncake requesters may restart "
            "while keeping the owner alive"
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        for length, prefixes in prefixes_by_length.items():
            for concurrency in args.concurrencies:
                workload = [
                    prefixes[index % len(prefixes)]
                    for index in range(args.requests_per_cell)
                ]
                started = time.perf_counter()
                metrics_before = read_offload_metrics(args.metrics_url)
                latencies: list[float] = []
                errors: list[str] = []
                with ThreadPoolExecutor(max_workers=min(concurrency, len(workload))) as pool:
                    futures = [
                        pool.submit(issue_request, args.server, args.model, prefix, args.seed)
                        for prefix in workload
                    ]
                    for future in as_completed(futures):
                        try:
                            latencies.append(future.result())
                        except Exception as exc:  # Preserve failed stress cells.
                            errors.append(type(exc).__name__)
                row = {
                    "policy": args.policy,
                    "prefix_tokens": length,
                    "concurrency": concurrency,
                    "requests": len(latencies),
                    "requests_failed": len(errors),
                    "seed": args.seed,
                    "wall_ms": (time.perf_counter() - started) * 1000,
                }
                metrics_after = read_offload_metrics(args.metrics_url)
                for key in (
                    "CPU_to_GPU_bytes",
                    "CPU_to_GPU_time",
                    "GPU_to_CPU_bytes",
                    "GPU_to_CPU_time",
                ):
                    row[f"{key.lower()}_delta"] = (
                        metrics_after.get(key, 0.0)
                        - metrics_before.get(key, 0.0)
                        if metrics_before or metrics_after
                        else None
                    )
                if latencies:
                    row.update(
                        ttft_p50_ms=statistics.median(latencies),
                        ttft_p90_ms=percentile(latencies, 0.90),
                        ttft_p99_ms=percentile(latencies, 0.99),
                    )
                else:
                    row.update(
                        ttft_p50_ms=None,
                        ttft_p90_ms=None,
                        ttft_p99_ms=None,
                    )
                if errors:
                    row["error_types"] = sorted(set(errors))
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
