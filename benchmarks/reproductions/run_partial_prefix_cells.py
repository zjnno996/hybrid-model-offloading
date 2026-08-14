# SPDX-License-Identifier: Apache-2.0
"""Run a real two-tier prefix workload against LMCache MP.

Each cell first stores ``P + S_i`` requests in the LMCache CPU tier, then
restarts vLLM, warms only ``P`` into the new worker's local GPU cache, and
finally measures ``P + S_i``.  The measured residual is therefore a local
prefix hit followed by a CPU-tier lookup/load, rather than a cold full
request.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from run_native_independent_cells import (
    DATASET,
    MODEL,
    TOKENIZER,
    start_lmcache_server,
    remember_process_group,
    stop_server,
    wait_lmcache_ready,
    wait_ready,
)
import requests


ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "benchmarks/reproductions/sharegpt_mooncake_e2e.py"
POLICIES = {
    "P1": "all_load",
    "P2": "all_replay",
}
SUMMARY_RE = re.compile(
    r"latency_ms_p50/p90/p99=([0-9.]+)/([0-9.]+)/([0-9.]+)"
)


def wait_ready_for_process(
    process: subprocess.Popen, url: str, timeout_s: float
) -> None:
    """Wait for vLLM, surfacing an early worker failure immediately."""
    session = requests.Session()
    session.trust_env = False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited before becoming ready (returncode={process.returncode})"
            )
        try:
            response = session.get(f"{url}/health", timeout=3)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"vLLM did not become ready within {timeout_s:.0f}s")


def run_e2e(
    mode: str,
    server: str,
    *,
    requests: int,
    concurrency: int,
    shared_prefix_tokens: int,
    suffix_tokens: int,
    prefix_only: bool = False,
) -> str:
    command = [
        os.environ.get("PYTHON", "python"),
        str(E2E),
        mode,
        "--server",
        server,
        "--requests",
        str(requests),
        "--concurrency",
        str(concurrency),
        "--warmup-concurrency",
        "1",
        "--model",
        "Qwen3.5-27B",
        "--tokenizer",
        TOKENIZER,
        "--dataset-path",
        DATASET,
        "--shared-prefix-tokens",
        str(shared_prefix_tokens),
        "--suffix-tokens",
        str(suffix_tokens),
        "--warmup-settle-seconds",
        "2",
    ]
    if prefix_only:
        command.append("--prefix-only")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def vllm_command(
    policy: str,
    *,
    server_port: int,
    lmcache_port: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
) -> list[str]:
    return [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(server_port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--enforce-eager",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        "0",
        "--served-model-name",
        "Qwen3.5-27B",
        "--kv-transfer-config",
        json.dumps(
            {
                "kv_connector": "LMCacheMPConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "lmcache.mp.host": "tcp://127.0.0.1",
                    "lmcache.mp.port": lmcache_port,
                },
            }
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICIES, default="P1")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--shared-prefix-tokens", type=int, default=256)
    parser.add_argument(
        "--suffix-tokens",
        type=int,
        default=1312,
        help="CPU-resident residual; default makes a 256+1312=1568-token prompt.",
    )
    parser.add_argument("--server-port", type=int, default=8002)
    parser.add_argument("--lmcache-port", type=int, default=5556)
    parser.add_argument("--lmcache-http-port", type=int, default=8081)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help=(
            "GPU fraction for the two-worker test. LMCache's CUDA IPC staging "
            "pool remains resident while the second worker starts."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/root/qwen35_partial_prefix.jsonl")
    )
    args = parser.parse_args()
    if args.requests is None:
        args.requests = args.concurrency
    if args.requests < args.concurrency:
        raise ValueError("requests must be >= concurrency")

    total_tokens = args.shared_prefix_tokens + args.suffix_tokens
    max_model_len = ((total_tokens + 127) // 128) * 128
    max_num_batched_tokens = max(2 * total_tokens - 1, 2 * 784 - 1)
    config_path = Path(f"/tmp/partial_lmcache_{args.lmcache_port}.yaml")
    config_path.write_text(
        "\n".join(
            [
                "chunk_size: 784",
                "local_device: cpu",
                "local_cpu: true",
                "max_local_cpu_size: 48",
                "enable_hidden_state_cache: true",
                "max_hidden_state_cpu_size: 48",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_USE_SIMPLE_KV_OFFLOAD": "0",
            "VLLM_MOONCAKE_HYBRID_POLICY": POLICIES[args.policy],
            "VLLM_MOONCAKE_HYBRID_ACTIVATION_CHECKPOINT": "0",
            "LMCACHE_CONFIG_FILE": str(config_path),
        }
    )

    lmcache = start_lmcache_server(
        config_path,
        port=args.lmcache_port,
        http_port=args.lmcache_http_port,
        chunk_size=784,
    )
    process = None
    try:
        wait_lmcache_ready(args.lmcache_http_port, 60)
        server = f"http://127.0.0.1:{args.server_port}"
        command = vllm_command(
            "P1",
            server_port=args.server_port,
            lmcache_port=args.lmcache_port,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        seed_log = Path(f"/tmp/partial_seed_c{args.concurrency}.log").open("w")
        process = subprocess.Popen(
            command,
            env={**env, "VLLM_MOONCAKE_HYBRID_POLICY": "all_load"},
            cwd=ROOT,
            stdout=seed_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        remember_process_group(process)
        wait_ready_for_process(process, server, args.startup_timeout)
        run_e2e(
            "warmup",
            server,
            requests=args.requests,
            concurrency=1,
            shared_prefix_tokens=args.shared_prefix_tokens,
            suffix_tokens=args.suffix_tokens,
        )
        stop_server(process)
        process = None
        seed_log.close()

        test_log = Path(f"/tmp/partial_{args.policy}_c{args.concurrency}.log").open("w")
        process = subprocess.Popen(
            vllm_command(
                args.policy,
                server_port=args.server_port,
                lmcache_port=args.lmcache_port,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
            ),
            env=env,
            cwd=ROOT,
            stdout=test_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        remember_process_group(process)
        wait_ready_for_process(process, server, args.startup_timeout)
        run_e2e(
            "warmup",
            server,
            requests=args.requests,
            concurrency=1,
            shared_prefix_tokens=args.shared_prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            prefix_only=True,
        )
        output = run_e2e(
            "load",
            server,
            requests=args.requests,
            concurrency=args.concurrency,
            shared_prefix_tokens=args.shared_prefix_tokens,
            suffix_tokens=args.suffix_tokens,
        )
        match = SUMMARY_RE.search(output)
        if match is None:
            raise RuntimeError(f"could not parse TTFT summary: {output}")
        row = {
            "workload": "local_prefix_hit_cpu_residual_hit",
            "backend": "lmcache_mp",
            "policy": args.policy,
            "shared_prefix_tokens": args.shared_prefix_tokens,
            "cpu_residual_tokens": args.suffix_tokens,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "seed": 0,
            "ttft_p50_ms": float(match.group(1)),
            "ttft_p90_ms": float(match.group(2)),
            "ttft_p99_ms": float(match.group(3)),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    finally:
        if process is not None:
            stop_server(process)
        stop_server(lmcache)
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
