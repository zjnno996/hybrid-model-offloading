# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise CPU-offloaded prefix recovery with real ShareGPT turns.

The same client works with either the native vLLM CPU offloader
(``--kv-offloading-backend native``) or the MooncakeStoreConnector. Run
``warmup`` (or the backwards-compatible ``populate`` alias) on a deterministic
ShareGPT subset, then run ``load`` with the identical arguments. The latter
measures HTTP/request queueing, prefix matching, scheduler admission,
connector lookup, CPU-to-GPU materialization, prefill, and first-token
latency (TTFT). For native offload, warmup and load are intentionally kept in
one vLLM process so the CPU tier remains alive; the metrics endpoint can verify
that requests actually performed CPU->GPU H2D recovery.

The dataset is streamed with ijson and only the selected prefixes are retained
in memory; the 900 MiB source JSON is never loaded as one Python list.
"""

import argparse
import os
import statistics
import time
from collections.abc import Iterable, Sequence
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
    "system": "System",
}


@dataclass(frozen=True)
class ShareGPTPrefix:
    record_index: int
    token_count: int
    prompt: str


def select_partial_prefixes(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    count: int,
    shared_prefix_tokens: int,
    suffix_tokens: int,
    skip: int = 0,
) -> list[ShareGPTPrefix]:
    """Build a two-tier prefix workload from real ShareGPT turns.

    The first ``shared_prefix_tokens`` come from one real ShareGPT prompt and
    are identical for every request.  Each request receives a different
    suffix, also taken from a real ShareGPT prompt.  The resulting workload
    is useful for testing a local-prefix hit followed by an LMCache CPU hit:
    the local worker can retain the shared prefix while the suffix is loaded
    from the external CPU tier.
    """
    if shared_prefix_tokens < 1 or suffix_tokens < 1:
        raise ValueError("shared prefix and suffix lengths must be positive")
    if count < 1 or skip < 0:
        raise ValueError("count must be positive and skip must be non-negative")

    # Use two deterministic streaming passes.  This keeps the large source
    # JSON out of memory while making the base prefix independent of suffix
    # selection.
    candidates = select_prefixes(
        records,
        tokenizer,
        count=1,
        min_tokens=shared_prefix_tokens,
        max_tokens=max(shared_prefix_tokens + 4096, shared_prefix_tokens),
    )
    base_ids = tokenizer.encode(candidates[0].prompt, add_special_tokens=False)
    base_text = tokenizer.decode(
        base_ids[:shared_prefix_tokens],
        clean_up_tokenization_spaces=False,
    )
    suffixes = select_prefixes(
        records,
        tokenizer,
        count=count,
        skip=skip + 1,
        min_tokens=suffix_tokens,
        max_tokens=max(suffix_tokens + 4096, suffix_tokens),
    )

    partial: list[ShareGPTPrefix] = []
    for suffix in suffixes:
        suffix_ids = tokenizer.encode(suffix.prompt, add_special_tokens=False)
        suffix_text = tokenizer.decode(
            suffix_ids[:suffix_tokens],
            clean_up_tokenization_spaces=False,
        )
        # Concatenate at the token boundary without injecting an extra
        # separator token.  The boundary is checked after re-tokenization so
        # a tokenizer change cannot silently destroy the intended prefix.
        prompt = base_text + suffix_text
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        common = 0
        for expected, actual in zip(base_ids[:shared_prefix_tokens], prompt_ids):
            if expected != actual:
                break
            common += 1
        if common != shared_prefix_tokens:
            raise ValueError(
                "constructed partial prompt lost the shared token prefix: "
                f"expected {shared_prefix_tokens}, got {common}"
            )
        partial.append(
            ShareGPTPrefix(
                record_index=suffix.record_index,
                token_count=len(prompt_ids),
                prompt=prompt,
            )
        )
    return partial


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def select_prefixes(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    count: int,
    skip: int = 0,
    min_tokens: int,
    max_tokens: int,
) -> list[ShareGPTPrefix]:
    if count < 1 or skip < 0:
        raise ValueError("count must be positive and skip must be non-negative")
    prefixes: list[ShareGPTPrefix] = []
    matched = 0
    for record_index, record in enumerate(records):
        history: list[str] = []
        saw_user = False
        saw_assistant = False
        for turn in record.get("conversations", []):
            role = ROLE_NAMES.get(turn.get("from"))
            content = turn.get("value")
            if role is None or not isinstance(content, str) or not content.strip():
                continue
            if role == "User" and saw_user and saw_assistant:
                prompt = "\n\n".join([*history, f"{role}: {content}"])
                token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
                if min_tokens <= token_count <= max_tokens:
                    if matched < skip:
                        matched += 1
                        history.append(f"{role}: {content}")
                        saw_user |= role == "User"
                        saw_assistant |= role == "Assistant"
                        continue
                    prefixes.append(
                        ShareGPTPrefix(record_index, token_count, prompt)
                    )
                    if len(prefixes) == count:
                        return prefixes
            history.append(f"{role}: {content}")
            saw_user |= role == "User"
            saw_assistant |= role == "Assistant"
    raise ValueError(
        f"found only {len(prefixes)} ShareGPT prefixes in [{min_tokens}, "
        f"{max_tokens}] tokens; need {count}"
    )


def issue_request(
    server: str, model: str, prefix: ShareGPTPrefix, seed: int
) -> tuple[int, float]:
    started = time.perf_counter()
    # Do not route localhost TTFT measurements through the shell's proxy.
    # Otherwise a proxy error is reported as a false 502 even when vLLM
    # completed the request successfully.
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
        timeout=300,
    )
    response.raise_for_status()
    return prefix.record_index, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["warmup", "populate", "load"])
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "/data/datasets/sharegpt/sharegpt_20230401_clean_lang_split.json"
        ),
    )
    parser.add_argument("--model", default="Qwen3.5-27B")
    parser.add_argument("--tokenizer", default="/data/models/Qwen3.5-27B")
    parser.add_argument("--server", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--requests",
        type=int,
        default=16,
        help="Number of ShareGPT prefixes in the bounded test set.",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--warmup-concurrency",
        type=int,
        default=1,
        help="Concurrency for cache-populating warmup; sequential is safest.",
    )
    parser.add_argument(
        "--warmup-settle-seconds",
        type=float,
        default=2.0,
        help="Wait for asynchronous Mooncake stores after warmup requests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed sampling seed sent with every deterministic request.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many deterministically selected prefixes.",
    )
    parser.add_argument("--min-tokens", type=int, default=784)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=0,
        help="Construct a shared real-ShareGPT prefix followed by unique suffixes.",
    )
    parser.add_argument(
        "--suffix-tokens",
        type=int,
        default=0,
        help="Suffix length for --shared-prefix-tokens workloads.",
    )
    parser.add_argument(
        "--prefix-only",
        action="store_true",
        help="For a partial workload, send only the shared prefix (GPU warmup stage).",
    )
    args = parser.parse_args()
    if (
        args.requests < 1
        or args.concurrency < 1
        or args.warmup_concurrency < 1
        or args.start_index < 0
        or args.warmup_settle_seconds < 0
    ):
        raise ValueError(
            "requests and concurrency must be positive; start-index must be non-negative"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if bool(args.shared_prefix_tokens) != bool(args.suffix_tokens):
        raise ValueError("--shared-prefix-tokens and --suffix-tokens must be used together")
    if args.shared_prefix_tokens:
        prefixes = select_partial_prefixes(
            iter_records(args.dataset_path),
            tokenizer,
            count=args.requests,
            skip=args.start_index,
            shared_prefix_tokens=args.shared_prefix_tokens,
            suffix_tokens=args.suffix_tokens,
        )
        if args.prefix_only:
            shared = prefixes[0].prompt
            shared_ids = tokenizer.encode(shared, add_special_tokens=False)
            shared_text = tokenizer.decode(
                shared_ids[: args.shared_prefix_tokens],
                clean_up_tokenization_spaces=False,
            )
            prefixes = [
                ShareGPTPrefix(p.record_index, args.shared_prefix_tokens, shared_text)
                for p in prefixes
            ]
    else:
        if args.prefix_only:
            raise ValueError("--prefix-only requires a partial prefix workload")
        prefixes = select_prefixes(
            iter_records(args.dataset_path),
            tokenizer,
            count=args.requests,
            skip=args.start_index,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
    token_counts = [prefix.token_count for prefix in prefixes]
    print(
        f"selected {len(prefixes)} real ShareGPT prefixes; tokens p50/p90 = "
        f"{statistics.median(token_counts):.0f}/{percentile(token_counts, 0.9):.0f}"
    )

    # ``warmup`` and ``populate`` deliberately do not report these requests as
    # measurements.  Native CPU offloading keeps the CPU tier in the same vLLM
    # process, so run ``load`` against this process after GPU-cache eviction.
    # Mooncake deployments may restart the requester while keeping the
    # Mooncake owner alive.
    if args.mode in {"warmup", "populate"}:
        policy = os.environ.get("VLLM_MOONCAKE_HYBRID_POLICY", "all_load")
        if policy != "all_load":
            print(
                f"warning: warmup is running with policy={policy!r}; for a fair "
                "four-policy comparison, warm up once with all_load so both "
                "Full and Linear groups are written to Mooncake."
            )
        with ThreadPoolExecutor(
            max_workers=min(args.warmup_concurrency, len(prefixes))
        ) as pool:
            futures = [
                pool.submit(issue_request, args.server, args.model, prefix, args.seed)
                for prefix in prefixes
            ]
            for future in as_completed(futures):
                record_index, latency_ms = future.result()
                print(f"warmup record={record_index} latency_ms={latency_ms:.3f}")
        if args.warmup_settle_seconds:
            time.sleep(args.warmup_settle_seconds)
        print(
            f"warmup complete: {len(prefixes)} prefixes written/attempted; "
            "run load in this process after GPU-cache eviction (or restart only "
            "the Mooncake requester while keeping its owner alive)."
        )
        return

    latencies: list[float] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(prefixes))) as pool:
        futures = [
            pool.submit(issue_request, args.server, args.model, prefix, args.seed)
            for prefix in prefixes
        ]
        for future in as_completed(futures):
            record_index, latency_ms = future.result()
            latencies.append(latency_ms)
            print(f"record={record_index} latency_ms={latency_ms:.3f}")
    wall_ms = (time.perf_counter() - started) * 1000
    print(
        f"mode={args.mode} requests={len(latencies)} concurrency={args.concurrency} "
        f"wall_ms={wall_ms:.3f} latency_ms_p50/p90/p99="
        f"{statistics.median(latencies):.3f}/{percentile(latencies, 0.9):.3f}/"
        f"{percentile(latencies, 0.99):.3f}"
    )


if __name__ == "__main__":
    main()
