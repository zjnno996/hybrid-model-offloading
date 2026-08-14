# SPDX-License-Identifier: Apache-2.0
"""Plot the native CPU-offload P1--P4 endpoint measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/root/qwen35_native_policy_results.jsonl"))
    parser.add_argument("--output-prefix", type=Path, default=Path("/root/qwen35_native_policy"))
    args = parser.parse_args()
    rows = load_rows(args.input)
    rows.sort(key=lambda row: row["policy"])
    policies = [row["policy"] for row in rows]
    colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea"]
    x = np.arange(len(rows))
    width = 0.24

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    p50 = [row["ttft_p50_ms"] for row in rows]
    p90 = [row["ttft_p90_ms"] for row in rows]
    p99 = [row["ttft_p99_ms"] for row in rows]
    axes[0].bar(x - width, p50, width, label="p50", color=colors)
    axes[0].bar(x, p90, width, label="p90", color="#94a3b8")
    axes[0].bar(x + width, p99, width, label="p99", color="#cbd5e1")
    axes[0].set_title("Native CPU/H2D endpoint TTFT")
    axes[0].set_ylabel("TTFT (ms)")
    axes[0].set_xticks(x, policies)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    h2d_mb = [row["cpu_to_gpu_bytes"] / 1024**2 for row in rows]
    d2h_mb = [row["gpu_to_cpu_bytes"] / 1024**2 for row in rows]
    axes[1].bar(x - width / 2, h2d_mb, width, label="CPU→GPU H2D", color="#0f766e")
    axes[1].bar(x + width / 2, d2h_mb, width, label="GPU→CPU D2H", color="#f59e0b")
    axes[1].set_title("Measured transfer volume")
    axes[1].set_ylabel("MiB")
    axes[1].set_xticks(x, policies)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    h2d_ms = [row["cpu_to_gpu_time_s"] * 1000 for row in rows]
    d2h_ms = [row["gpu_to_cpu_time_s"] * 1000 for row in rows]
    axes[2].bar(x - width / 2, h2d_ms, width, label="H2D DMA", color="#0f766e")
    axes[2].bar(x + width / 2, d2h_ms, width, label="D2H DMA", color="#f59e0b")
    axes[2].set_title("Measured DMA time")
    axes[2].set_ylabel("seconds")
    axes[2].set_xticks(x, policies)
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Qwen3.5-27B native offloading: P1/P2/P3/P4\n"
        "ShareGPT prefixes p50≈830, p90≈860 tokens; 72-prefix eviction; C=4; seed=0"
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=180)
    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    print(args.output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

