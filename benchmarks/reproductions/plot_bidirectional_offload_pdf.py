# SPDX-License-Identifier: Apache-2.0
"""Create PDF figures for the bidirectional Qwen3.5 offload sweep.

The input JSONL files are produced by qwen35_layer_selection_sweep.py after
the bidirectional D2H/H2D measurement was added.  The PDF contains:

1. Full-Attention D2H, H2D, and round-trip bars;
2. Linear-Attention D2H, H2D, and round-trip bars;
3. round-trip heatmaps; and
4. load-versus-recompute crossover curves.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


PREFIXES = [128, 256, 512, 900]
CONCURRENCIES = [1, 4, 16, 32, 64]
COLORS = {
    "gpu_to_cpu_p50_ms": "#3568a8",
    "cpu_to_gpu_p50_ms": "#e28b32",
    "roundtrip_p50_ms": "#3b9b65",
}


def load_rows(pattern: str) -> list[dict]:
    rows = []
    for filename in sorted(glob.glob(pattern)):
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"no JSONL files matched {pattern!r}")
    return rows


def index_rows(rows: list[dict], attention_type: str, layer_count: int) -> dict:
    return {
        (row["prefix_tokens"], row["concurrency"]): row
        for row in rows
        if row.get("attention_type") == attention_type
        and row.get("layer_count") == layer_count
    }


def add_common_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Time (ms, p50)")
    ax.set_xticks(range(len(CONCURRENCIES)), [str(x) for x in CONCURRENCIES])
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_yscale("log")


def bar_page(pdf: PdfPages, data: dict, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    width = 0.24
    x = np.arange(len(CONCURRENCIES))
    fields = list(COLORS)
    labels = ["GPU→CPU offload", "CPU→GPU restore", "Round-trip"]
    for ax, prefix in zip(axes.flat, PREFIXES):
        for offset, field, label in zip(
            [-width, 0, width], fields, labels
        ):
            values = [
                data[(prefix, concurrency)][field]
                for concurrency in CONCURRENCIES
            ]
            ax.bar(
                x + offset,
                values,
                width,
                label=label,
                color=COLORS[field],
                edgecolor="white",
                linewidth=0.4,
            )
        add_common_axis(ax, f"Prefix length = {prefix} tokens")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def heatmap_page(pdf: PdfPages, full: dict, linear: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Bidirectional offload round-trip (p50, ms)", fontsize=15,
                 fontweight="bold")
    for ax, data, title in zip(
        axes, [full, linear], ["Full Attention (16 layers)", "Linear Attention (48 layers)"]
    ):
        matrix = np.array([
            [data[(prefix, concurrency)]["roundtrip_p50_ms"]
             for concurrency in CONCURRENCIES]
            for prefix in PREFIXES
        ])
        image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Prefix tokens")
        ax.set_xticks(range(len(CONCURRENCIES)), CONCURRENCIES)
        ax.set_yticks(range(len(PREFIXES)), PREFIXES)
        for row in range(len(PREFIXES)):
            for col in range(len(CONCURRENCIES)):
                ax.text(col, row, f"{matrix[row, col]:.1f}",
                        ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def crossover_page(pdf: PdfPages, rows: list[dict]) -> None:
    full = index_rows(rows, "full", 16)
    linear = index_rows(rows, "linear", 48)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    fig.suptitle("Restore load versus replay (p50, ms)", fontsize=15,
                 fontweight="bold")
    for ax, data, title in zip(
        axes, [full, linear], ["Full Attention (16 layers)", "Linear Attention (48 layers)"]
    ):
        for prefix in PREFIXES:
            load = [data[(prefix, c)]["load_p50_ms"] for c in CONCURRENCIES]
            replay = [data[(prefix, c)]["recompute_p50_ms"] for c in CONCURRENCIES]
            color = plt.cm.viridis((prefix - min(PREFIXES)) /
                                   (max(PREFIXES) - min(PREFIXES)))
            ax.plot(CONCURRENCIES, load, marker="o", color=color,
                    label=f"{prefix} load")
            ax.plot(CONCURRENCIES, replay, marker="x", linestyle="--",
                    color=color, alpha=0.85, label=f"{prefix} replay")
        ax.set_title(title)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Time (ms, p50)")
        ax.set_xticks(CONCURRENCIES)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default="/tmp/qwen35_bidirectional_[0-9]*.jsonl",
    )
    parser.add_argument(
        "--comparison-glob",
        default="/tmp/qwen35_layer_selection_clean_[0-9]*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/qwen35_bidirectional_offload.pdf"),
    )
    args = parser.parse_args()
    rows = load_rows(args.input_glob)
    comparison_rows = load_rows(args.comparison_glob)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    full = index_rows(rows, "full", 16)
    linear = index_rows(rows, "linear", 48)
    with PdfPages(args.output) as pdf:
        bar_page(pdf, full, "Full Attention: GPU↔CPU offload")
        bar_page(pdf, linear, "Linear Attention: GPU↔CPU offload")
        heatmap_page(pdf, full, linear)
        crossover_page(pdf, comparison_rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
