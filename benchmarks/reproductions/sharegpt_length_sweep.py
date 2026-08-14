"""Length sweep for hybrid-prefix recovery using real ShareGPT conversations.

The selected conversation is token-truncated to an exact target length.  This
keeps the workload deterministic while allowing lengths that do not end on a
ShareGPT turn boundary.  Run ``load`` against a Mooncake-enabled server and
``recompute`` against a fresh server without a KV connector.
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from transformers import AutoTokenizer

try:
    from sharegpt_data import get_record
except ImportError:  # Imported as benchmarks.reproductions.*
    from benchmarks.reproductions.sharegpt_data import get_record


@dataclass(frozen=True)
class Prefix:
    record_index: int
    target_tokens: int
    actual_tokens: int
    prompt: str


def _conversation_text(record: dict) -> str:
    roles = {
        "human": "User",
        "user": "User",
        "gpt": "Assistant",
        "assistant": "Assistant",
    }
    turns = []
    for message in record.get("conversations", []):
        role = roles.get(message.get("from", ""), message.get("from", ""))
        turns.append(f"{role}: {message.get('value', '')}")
    return "\n\n".join(turns)


def select_prefix(
    records: list[dict],
    tokenizer,
    target_tokens: int,
    record_index: int,
    salt: str,
) -> Prefix:
    text = _conversation_text(records[record_index])
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    salt_ids = tokenizer.encode(salt, add_special_tokens=False)
    if len(token_ids) + len(salt_ids) < target_tokens:
        raise ValueError(
            f"record {record_index} has {len(token_ids)} tokens, "
            f"but target is {target_tokens}"
        )
    # Decode the prefix so the server tokenizes exactly the same text that is
    # used for the real request.  The final turn may be truncated, which is
    # intentional: an agent context can end at an observation boundary.
    prompt_ids = salt_ids + token_ids[: target_tokens - len(salt_ids)]
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    return Prefix(record_index, target_tokens, actual, prompt)


def issue_request(server: str, model: str, prefix: Prefix) -> float:
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
        },
        timeout=600,
    )
    response.raise_for_status()
    return (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["warmup", "populate", "load", "recompute"]
    )
    parser.add_argument("--dataset-path", type=Path, default=Path(
        "/data/datasets/sharegpt/sharegpt_20230401_clean_lang_split.json"
    ))
    parser.add_argument("--tokenizer", default="/data/models/Qwen3.5-27B")
    parser.add_argument("--model", default="Qwen3.5-27B")
    parser.add_argument("--server", default="http://127.0.0.1:8001")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--record-index", type=int, required=True)
    parser.add_argument(
        "--salt",
        default=None,
        help="Unique session prefix to prevent cross-length local-prefix hits.",
    )
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup-settle-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if (
        args.target_tokens < 1
        or args.requests < 1
        or args.concurrency < 1
        or args.warmup_settle_seconds < 0
    ):
        raise ValueError("target-tokens, requests and concurrency must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    salt = args.salt or f"Agent session length={args.target_tokens}:\n"
    prefix = select_prefix(
        [get_record(args.dataset_path, args.record_index)],
        tokenizer,
        args.target_tokens,
        0,
        salt,
    )
    prefix = Prefix(
        args.record_index, prefix.target_tokens, prefix.actual_tokens, prefix.prompt
    )
    print(
        f"mode={args.mode} record={prefix.record_index} "
        f"target_tokens={prefix.target_tokens} actual_tokens={prefix.actual_tokens} "
        f"requests={args.requests} concurrency={args.concurrency}"
    )

    if args.mode in {"warmup", "populate"}:
        for _ in range(args.requests):
            issue_request(args.server, args.model, prefix)
        if args.warmup_settle_seconds:
            time.sleep(args.warmup_settle_seconds)
        print(
            "warmup complete; restart vLLM (keep Mooncake CPU owner alive) "
            "before running load."
        )
        return

    latencies = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(args.concurrency, args.requests)) as pool:
        futures = [pool.submit(issue_request, args.server, args.model, prefix)
                   for _ in range(args.requests)]
        for future in as_completed(futures):
            latency = future.result()
            latencies.append(latency)
            print(f"latency_ms={latency:.3f}")
    wall_ms = (time.perf_counter() - started) * 1000
    print(
        f"wall_ms={wall_ms:.3f} latency_ms_p50={statistics.median(latencies):.3f} "
        f"latency_ms_max={max(latencies):.3f}"
    )


if __name__ == "__main__":
    main()
