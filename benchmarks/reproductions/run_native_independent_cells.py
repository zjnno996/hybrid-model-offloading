# SPDX-License-Identifier: Apache-2.0
"""Run isolated native H2D TTFT cells with a fresh vLLM process per cell."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "benchmarks/reproductions/sharegpt_mooncake_e2e.py"
MODEL = "/data/models/Qwen3.5-27B"
DATASET = "/data/datasets/sharegpt/sharegpt_20230401_clean_lang_split.json"
TOKENIZER = MODEL
POLICIES = {
    "P1": ("all_load", True),
    "P2": ("all_load", False),
    "P3": ("full_load_linear_replay", True),
    "P4": ("full_replay_linear_load", True),
}
SUMMARY_RE = re.compile(
    r"latency_ms_p50/p90/p99=([0-9.]+)/([0-9.]+)/([0-9.]+)"
)
SUFFIX_RE = re.compile(
    r"Hybrid suffix-only replay .*?local_prefix=(\d+) loaded_tokens=(\d+)"
)
KV_METRIC_RE = re.compile(r"KV Transfer metrics: (?P<body>.*)")
KV_FIELD_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[0-9.eE+-]+)")


def wait_ready(url: str, timeout_s: float, path: str = "/health") -> None:
    session = requests.Session()
    session.trust_env = False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = session.get(f"{url}{path}", timeout=3)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"vLLM did not become ready within {timeout_s:.0f}s")


def wait_lmcache_ready(port: int, timeout_s: float) -> None:
    wait_ready(f"http://127.0.0.1:{port}", timeout_s, path="/healthcheck")


def read_h2d_metrics(log_path: Path) -> dict[str, float]:
    """Aggregate native connector metrics emitted by the vLLM server log."""
    totals: dict[str, float] = {}
    if not log_path.exists():
        return totals
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = KV_METRIC_RE.search(line)
        if match is None:
            continue
        for field in KV_FIELD_RE.finditer(match.group("body")):
            key = field.group("key")
            if key.endswith(("_total_bytes", "_total_time", "_total_queue_time")):
                totals[key] = totals.get(key, 0.0) + float(field.group("value"))
    return totals


def remember_process_group(process: subprocess.Popen) -> None:
    """Remember the session/process-group created for a child process.

    ``Popen`` can already report an exited leader by the time cleanup starts,
    while EngineCore descendants are still alive.  In that case looking up
    the PGID from ``process.pid`` fails, so retain it immediately after spawn.
    All callers use ``start_new_session=True``; therefore the leader PID is
    also a private process-group ID.
    """
    try:
        process._process_group_id = os.getpgid(process.pid)  # type: ignore[attr-defined]
    except ProcessLookupError:
        process._process_group_id = process.pid  # type: ignore[attr-defined]


def stop_server(process: subprocess.Popen) -> None:
    # vLLM starts EngineCore and multiprocessing helper processes below the
    # API server.  The server is launched with ``start_new_session=True``, so
    # signal the whole process group; signalling only the API PID can leave a
    # CUDA-owning EngineCore orphaned and keep most GPU memory allocated.
    process_group = getattr(process, "_process_group_id", None)
    if process_group is None:
        try:
            process_group = os.getpgid(process.pid)
        except ProcessLookupError:
            # The leader may be gone while EngineCore is still in the group.
            process_group = process.pid

    def signal_group(sig: signal.Signals) -> None:
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            # The group is already gone.  Do not signal an unrelated PID.
            pass

    def group_alive() -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        return True

    signal_group(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        signal_group(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            signal_group(signal.SIGKILL)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # A child can remain in uninterruptible kernel sleep.  The
                # group kill is still the correct last resort; report the
                # timeout to the caller instead of silently continuing.
                raise
    finally:
        # The API leader can exit cleanly while EngineCore survives.  Always
        # reap any remaining descendants in the private group before the next
        # cell starts, otherwise they retain CUDA contexts and GPU memory.
        signal_group(signal.SIGKILL)
        # Do not start another GPU cell if a child is stuck in D-state (or
        # otherwise ignored the kill).  Continuing would overlap CUDA
        # contexts and recreate the original "GPU memory is still allocated"
        # failure; the caller must stop and repair the host instead.
        deadline = time.monotonic() + 5
        while group_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        group_survived = group_alive()
        log_file = getattr(process, "_lmcache_log_file", None)
        if log_file is not None:
            log_file.close()
        if group_survived:
            raise RuntimeError(
                f"process group {process_group} survived SIGKILL; "
                "refusing to start another GPU cell"
            )


def start_lmcache_server(
    config: Path, port: int, http_port: int, chunk_size: int
) -> subprocess.Popen:
    """Start the real LMCache MP server used by the hybrid backend."""
    values = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in config.read_text(encoding="utf-8").splitlines()
        if ":" in line and not line.lstrip().startswith("#")
    }
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.http_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(http_port),
        "--chunk-size",
        str(chunk_size),
        "--l1-size-gb",
        values.get("max_local_cpu_size", "32"),
        "--eviction-policy",
        "LRU",
        "--separate-object-groups",
        "--disable-observability",
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    server_log = Path(f"/tmp/lmcache_mp_{port}.log").open("w")
    process = subprocess.Popen(
        command,
        env=env,
        cwd=ROOT,
        # Keep the server log drained and inspectable.  An unread PIPE can
        # fill during startup and block worker registration.
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    remember_process_group(process)
    process._lmcache_log_file = server_log  # type: ignore[attr-defined]
    return process


def run_e2e(
    mode: str,
    server: str,
    concurrency: int,
    requests_count: int,
    *,
    shared_prefix_tokens: int = 0,
    suffix_tokens: int = 0,
    prefix_only: bool = False,
) -> str:
    command = [
        sys.executable,
        str(E2E),
        mode,
        "--server",
        server,
        "--requests",
        str(requests_count),
        "--concurrency",
        str(concurrency),
        "--warmup-concurrency",
        "1",
        "--min-tokens",
        "784",
        "--max-tokens",
        "900",
        "--model",
        "Qwen3.5-27B",
        "--tokenizer",
        TOKENIZER,
        "--dataset-path",
        DATASET,
        "--warmup-settle-seconds",
        "1",
    ]
    if shared_prefix_tokens or suffix_tokens:
        command += [
            "--shared-prefix-tokens",
            str(shared_prefix_tokens),
            "--suffix-tokens",
            str(suffix_tokens),
        ]
        if prefix_only:
            command.append("--prefix-only")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/qwen35_native_independent_cells.jsonl"),
    )
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=72)
    parser.add_argument(
        "--activation-checkpoint",
        action="store_true",
        help="Enable hybrid activation checkpoints for P3/P4 replay.",
    )
    parser.add_argument(
        "--activation-max-bytes",
        type=int,
        default=48 * 1024**3,
        help="Host-memory bound for activation checkpoints (0=unbounded).",
    )
    parser.add_argument(
        "--suffix-only",
        action="store_true",
        help=(
            "Keep a local prefix hit and replay only the externally missing "
            "suffix for mixed P3/P4 policies. Activation checkpoints are "
            "optional and should be disabled for a fair KV-only comparison."
        ),
    )
    parser.add_argument(
        "--retain-shared-prefix",
        action="store_true",
        help=(
            "After warmup, touch the shared prefix once more so it remains a "
            "local GPU prefix hit while unique suffixes are evicted."
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900,
        help="Server startup timeout; Qwen3.5-27B may need several minutes to load.",
    )
    parser.add_argument(
        "--backend",
        choices=("native", "lmcache"),
        default="native",
        help="KV backend for load cells; native is the control baseline.",
    )
    parser.add_argument(
        "--lmcache-config",
        type=Path,
        default=None,
        help="Existing LMCache YAML. If omitted, a local CPU/H2D config is created.",
    )
    parser.add_argument(
        "--lmcache-kv-gb", type=float, default=48.0,
        help="LMCache local CPU KV pool size when creating a config.",
    )
    parser.add_argument(
        "--lmcache-hidden-state-gb", type=float, default=48.0,
        help="LMCache hidden-state pinned pool size when creating a config.",
    )
    parser.add_argument(
        "--native-kv-gb",
        type=float,
        default=48.0,
        help="Native CPU offload capacity in GiB for the H2D baseline.",
    )
    parser.add_argument(
        "--lmcache-chunk-size",
        type=int,
        default=784,
        help="LMCache/vLLM unified block size (784 for Qwen3.5-27B).",
    )
    parser.add_argument("--concurrencies", default="1,16,32")
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=0,
        help="Use a shared-prefix plus unique-suffix workload instead of 784-900 tokens.",
    )
    parser.add_argument(
        "--suffix-tokens",
        type=int,
        default=0,
        help="Unique suffix length for the partial-prefix workload.",
    )
    parser.add_argument(
        "--policies",
        default="P1,P2,P3,P4",
        help="Comma-separated policy names to run.",
    )
    args = parser.parse_args()
    if bool(args.shared_prefix_tokens) != bool(args.suffix_tokens):
        raise ValueError(
            "--shared-prefix-tokens and --suffix-tokens must be provided together"
        )
    partial_tokens = args.shared_prefix_tokens + args.suffix_tokens
    # Token-boundary concatenation can add a small number of tokens when the
    # decoded ShareGPT prefix and suffix are re-tokenized.  Keep a safety
    # margin (and a 1K floor) so short partial workloads do not turn into
    # HTTP 400 "prompt too long" failures merely because of that boundary.
    max_model_len = (
        max(1024, ((partial_tokens + 127) // 128) * 128)
        if partial_tokens
        else 1024
    )
    max_num_batched_tokens = (
        max(2 * partial_tokens - 1, 2 * args.lmcache_chunk_size - 1)
        if partial_tokens
        else 2 * args.lmcache_chunk_size - 1
    )
    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    selected_policies = [
        value.strip() for value in args.policies.split(",") if value.strip()
    ]
    unknown_policies = set(selected_policies) - set(POLICIES)
    if unknown_policies:
        raise ValueError(f"unknown policies: {sorted(unknown_policies)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    generated_lmcache_config = None
    if args.backend == "lmcache" and args.lmcache_config is None:
        config_fd, config_path = tempfile.mkstemp(
            prefix="vllm_lmcache_hybrid_", suffix=".yaml"
        )
        os.close(config_fd)
        generated_lmcache_config = Path(config_path)
        generated_lmcache_config.write_text(
            "\n".join(
                [
                    f"chunk_size: {args.lmcache_chunk_size}",
                    "local_device: cpu",
                    "local_cpu: true",
                    f"max_local_cpu_size: {args.lmcache_kv_gb}",
                    "enable_hidden_state_cache: true",
                    f"max_hidden_state_cpu_size: {args.lmcache_hidden_state_gb}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    lmcache_config = args.lmcache_config or generated_lmcache_config

    for policy in selected_policies:
        policy_env, use_offload = POLICIES[policy]
        # P2 is the no-offload/native replay control.  When the real LMCache
        # MP connector is selected, keep the connector enabled but explicitly
        # suppress every external group load so the residual is recomputed.
        if args.backend == "lmcache" and policy == "P2":
            policy_env = "all_replay"
        for concurrency in concurrencies:
            server = f"http://127.0.0.1:{args.port}"
            lmcache_process = None
            env = os.environ.copy()
            env.update(
                {
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                    "VLLM_USE_SIMPLE_KV_OFFLOAD": "0",
                    "VLLM_MOONCAKE_HYBRID_POLICY": policy_env,
                    "VLLM_MOONCAKE_HYBRID_ACTIVATION_CHECKPOINT": (
                        "1" if args.activation_checkpoint else "0"
                    ),
                    "VLLM_MOONCAKE_HYBRID_ACTIVATION_MAX_BYTES": str(
                        args.activation_max_bytes
                    ),
                    "VLLM_MOONCAKE_HYBRID_SUFFIX_ONLY": (
                        "1" if args.suffix_only else "0"
                    ),
                    # Emit connector transfer counters frequently enough that
                    # short TTFT cells still expose the actual H2D operation.
                    "VLLM_LOG_STATS_INTERVAL": "1",
                }
            )
            if args.backend == "lmcache":
                assert lmcache_config is not None
                env["LMCACHE_CONFIG_FILE"] = str(lmcache_config)
            command = [
                "vllm",
                "serve",
                MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
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
                "0.8",
                "--seed",
                "0",
                "--served-model-name",
                "Qwen3.5-27B",
            ]
            if args.backend == "lmcache":
                lmcache_process = start_lmcache_server(
                    lmcache_config,
                    port=5555 + concurrency,
                    http_port=8080 + concurrency,
                    chunk_size=args.lmcache_chunk_size,
                )
                wait_lmcache_ready(8080 + concurrency, 60)
                command += [
                    "--kv-transfer-config",
                    json.dumps(
                        {
                            "kv_connector": "LMCacheMPConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {
                                "lmcache.mp.host": "tcp://127.0.0.1",
                                "lmcache.mp.port": 5555 + concurrency,
                            },
                        }
                    ),
                ]
            elif use_offload:
                command += [
                    "--kv-offloading-size",
                    str(args.native_kv_gb),
                    "--kv-offloading-backend",
                    "native",
                ]
            log_path = Path(f"/tmp/native_independent_{policy}_c{concurrency}.log")
            print(f"starting policy={policy} concurrency={concurrency}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                remember_process_group(process)
            try:
                wait_ready(server, args.startup_timeout)
                run_e2e(
                    "warmup",
                    server,
                    1,
                    args.warmup_requests,
                    shared_prefix_tokens=args.shared_prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                )
                if args.retain_shared_prefix and args.shared_prefix_tokens:
                    run_e2e(
                        "warmup",
                        server,
                        1,
                        1,
                        shared_prefix_tokens=args.shared_prefix_tokens,
                        suffix_tokens=args.suffix_tokens,
                        prefix_only=True,
                    )
                output = run_e2e(
                    "load",
                    server,
                    concurrency,
                    args.requests,
                    shared_prefix_tokens=args.shared_prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                )
                match = SUMMARY_RE.search(output)
                if match is None:
                    raise RuntimeError(
                        f"could not parse TTFT summary for {policy}/C{concurrency}: "
                        f"{output}"
                    )
                # The suffix-only transition is logged by the engine process.
                # Stop it before reading the file so buffered logger output is
                # flushed and the validation fields cannot silently disappear.
                stop_server(process)
                process = None
                h2d_metrics = read_h2d_metrics(log_path)
                h2d_bytes = h2d_metrics.get("CPU_to_GPU_total_bytes", 0.0)
                if args.backend == "native" and use_offload and h2d_bytes <= 0:
                    raise RuntimeError(
                        f"{policy}/C{concurrency} produced no CPU_to_GPU bytes; "
                        "the request hit GPU prefix cache or did not complete an "
                        "H2D restore, so refusing to record it"
                    )
                row = {
                    "backend": (
                        "lmcache"
                        if args.backend == "lmcache"
                        else ("native" if use_offload else "none")
                    ),
                    "policy": policy,
                    "policy_env": policy_env,
                    "independent_process": True,
                    "prefix_tokens_min": 784,
                    "prefix_tokens_max": 900,
                    "shared_prefix_tokens": args.shared_prefix_tokens,
                    "suffix_tokens": args.suffix_tokens,
                    "warmup_requests": args.warmup_requests,
                    "requests": args.requests,
                    "concurrency": concurrency,
                    "seed": 0,
                    "activation_checkpoint": args.activation_checkpoint,
                    "activation_max_bytes": args.activation_max_bytes,
                    "suffix_only": args.suffix_only,
                    "retain_shared_prefix": args.retain_shared_prefix,
                    "ttft_p50_ms": float(match.group(1)),
                    "ttft_p90_ms": float(match.group(2)),
                    "ttft_p99_ms": float(match.group(3)),
                }
                row["h2d_cpu_to_gpu_bytes"] = h2d_bytes
                row["h2d_cpu_to_gpu_service_ms"] = 1000.0 * h2d_metrics.get(
                    "CPU_to_GPU_total_time", 0.0
                )
                row["h2d_cpu_to_gpu_queue_ms"] = 1000.0 * h2d_metrics.get(
                    "CPU_to_GPU_total_queue_time", 0.0
                )
                # Validate the intended suffix-only state transition from the
                # engine log instead of assuming that the flag was effective.
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                suffix_matches = SUFFIX_RE.findall(log_text)
                row["suffix_only_replay_count"] = len(suffix_matches)
                if suffix_matches:
                    row["suffix_only_local_prefix_tokens"] = int(
                        suffix_matches[-1][0]
                    )
                    row["suffix_only_loaded_tokens"] = int(suffix_matches[-1][1])
                with args.output.open("a", encoding="utf-8") as output_file:
                    output_file.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            finally:
                if process is not None:
                    stop_server(process)
                if lmcache_process is not None:
                    stop_server(lmcache_process)

    if generated_lmcache_config is not None:
        generated_lmcache_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
