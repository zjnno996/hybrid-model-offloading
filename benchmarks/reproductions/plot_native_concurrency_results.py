# SPDX-License-Identifier: Apache-2.0
"""Plot native TTFT while increasing request concurrency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/root/qwen35_native_concurrency_results.jsonl"))
    parser.add_argument("--output-prefix", type=Path, default=Path("/root/qwen35_native_concurrency"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    policies = ["P1", "P2", "P3", "P4"]
    labels = {
        "P1": "P1 all Load",
        "P2": "P2 replay",
        "P3": "P3 Full Load + Linear replay",
        "P4": "P4 Linear Load + Full replay",
    }
    colors = {"P1": "#2563eb", "P2": "#ea580c", "P3": "#16a34a", "P4": "#9333ea"}
    concurrencies = sorted({row["concurrency"] for row in rows})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for policy in policies:
        subset = sorted((row for row in rows if row["policy"] == policy), key=lambda r: r["concurrency"])
        axes[0].plot(
            [r["concurrency"] for r in subset],
            [r["ttft_p50_ms"] for r in subset],
            marker="o", linewidth=2.2, color=colors[policy], label=labels[policy],
        )
        axes[0].fill_between(
            [r["concurrency"] for r in subset],
            [r["ttft_p50_ms"] for r in subset],
            [r["ttft_p90_ms"] for r in subset],
            color=colors[policy], alpha=0.10,
        )
    axes[0].set_title("Native end-to-end TTFT vs concurrency")
    axes[0].set_xlabel("concurrency")
    axes[0].set_ylabel("TTFT (ms), p50; shaded=p50–p90")
    axes[0].set_xticks(concurrencies)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    width = 0.18
    x = list(range(len(concurrencies)))
    for i, policy in enumerate(policies):
        subset = {row["concurrency"]: row for row in rows if row["policy"] == policy}
        xx = [value + (i - 1.5) * width for value in x]
        p50 = [subset[c]["ttft_p50_ms"] for c in concurrencies]
        p90 = [subset[c]["ttft_p90_ms"] for c in concurrencies]
        axes[1].bar(xx, p50, width, color=colors[policy], label=policy)
        axes[1].errorbar(xx, p50, yerr=[[0] * len(p50), [p90[j] - p50[j] for j in range(len(p50))]], fmt="none", ecolor="#111827", capsize=3, linewidth=1)
    axes[1].set_title("Load vs replay crossover")
    axes[1].set_xlabel("concurrency")
    axes[1].set_ylabel("TTFT (ms), p50; whisker=p90")
    axes[1].set_xticks(x, [f"C{c}" for c in concurrencies])
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        "Qwen3.5-27B native CPU/H2D offloading\n"
        "ShareGPT p50≈830/p90≈860 tokens; 72-prefix GPU eviction; 32 requests/cell; seed=0"
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=180)
    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    print(args.output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
