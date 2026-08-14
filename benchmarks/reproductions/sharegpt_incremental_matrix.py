# SPDX-License-Identifier: Apache-2.0
"""Measure TTFT after a cached prefix plus a short incremental suffix."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from transformers import AutoTokenizer

try:
    from sharegpt_mooncake_matrix import make_prefixes
    from sharegpt_data import iter_records
except ImportError:  # Imported as benchmarks.reproductions.*
    from benchmarks.reproductions.sharegpt_mooncake_matrix import make_prefixes
    from benchmarks.reproductions.sharegpt_data import iter_records


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def suffix_prompt(tokenizer, prefix: str, suffix_tokens: int) -> str:
    suffix_text = (
        " User: a new tool result arrived. Continue the agent workflow with "
        "the next concrete action and mention any blocker."
    )
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    repeated = suffix_ids * ((suffix_tokens // len(suffix_ids)) + 3)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    # Decoding a concatenated BPE sequence and tokenizing it again can merge
    # one token at the text boundary.  Search a short look-ahead so the prompt
    # sent over HTTP has exactly the requested incremental token count while
    # preserving every cached-prefix token.
    for candidate_length in range(suffix_tokens, suffix_tokens + 16):
        prompt = tokenizer.decode(
            prefix_ids + repeated[:candidate_length], skip_special_tokens=False
        )
        full_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if (
            full_ids[: len(prefix_ids)] == prefix_ids
            and len(full_ids) - len(prefix_ids) == suffix_tokens
        ):
            return prompt
    raise ValueError(
        f"could not construct a {suffix_tokens}-token suffix without changing "
        "the cached prefix"
    )


def request_ttft(server: str, model: str, prompt: str, seed: int) -> float:
    started = time.perf_counter()
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        f"{server}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "seed": seed,
        },
        timeout=600,
    )
    response.raise_for_status()
    return (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("warmup", "load"))
    parser.add_argument("--policy", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument(
        "--unique-prefixes",
        type=int,
        default=None,
        help="Independent base prefixes to warm; defaults to --requests.",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warmup-settle-seconds",
        type=float,
        default=8.0,
        help="Wait for asynchronous Mooncake puts before the producer exits.",
    )
    parser.add_argument("--dataset-path", type=Path, default=Path(
        "/data/datasets/sharegpt/sharegpt_20230401_clean_lang_split.json"
    ))
    parser.add_argument("--tokenizer", default="/data/models/Qwen3.5-27B")
    parser.add_argument("--model", default="Qwen3.5-27B")
    parser.add_argument("--server", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, default=Path("/root/qwen35_incremental.jsonl"))
    args = parser.parse_args()
    unique_prefixes = args.unique_prefixes or args.requests
    if (
        args.requests < 1
        or unique_prefixes < 1
        or args.concurrency < 1
        or args.warmup_settle_seconds < 0
    ):
        raise ValueError("requests, unique-prefixes and concurrency must be positive")
    if args.mode == "warmup" and args.policy != "P1":
        raise ValueError("warmup must use P1/all_load so every group is stored")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prefixes = make_prefixes(
        iter_records(args.dataset_path), tokenizer,
        target_tokens=args.prefix_tokens,
        count=unique_prefixes,
        salt=f"Agent incremental prefix={args.prefix_tokens}:\n",
    )
    suffix_prompts = [
        suffix_prompt(tokenizer, prefix.prompt, args.suffix_tokens)
        for prefix in prefixes
    ]
    prompts = [
        suffix_prompts[index % len(suffix_prompts)]
        for index in range(args.requests)
    ]
    print(
        f"prefix_tokens={args.prefix_tokens} suffix_tokens={args.suffix_tokens} "
        f"requests={len(prompts)} unique_prefixes={len(prefixes)} "
        f"concurrency={args.concurrency}"
    )
    if args.mode == "warmup":
        for prefix in prefixes:
            request_ttft(args.server, args.model, prefix.prompt, args.seed)
        if args.warmup_settle_seconds:
            time.sleep(args.warmup_settle_seconds)
        print("warmup complete; restart vLLM and keep Mooncake CPU owner alive")
        return

    latencies: list[float] = []
    errors: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(prompts))) as pool:
        futures = [
            pool.submit(request_ttft, args.server, args.model, prompt, args.seed)
            for prompt in prompts
        ]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as exc:
                errors.append(type(exc).__name__)
    row = {
        "policy": args.policy,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "requests": len(prompts),
        "requests_failed": len(errors),
        "concurrency": args.concurrency,
        "seed": args.seed,
        "wall_ms": (time.perf_counter() - started) * 1000,
        "ttft_p50_ms": statistics.median(latencies) if latencies else None,
        "ttft_p90_ms": percentile(latencies, 0.9) if latencies else None,
        "ttft_p99_ms": percentile(latencies, 0.99) if latencies else None,
    }
    if errors:
        row["error_types"] = sorted(set(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
