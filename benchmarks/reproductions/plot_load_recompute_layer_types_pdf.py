# SPDX-License-Identifier: Apache-2.0
"""Plot load-vs-recompute evidence separately for Full and Linear layers."""

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
LOAD_COLOR = "#2878c8"
RECOMPUTE_COLOR = "#e58b2a"


def read_rows(pattern: str) -> list[dict]:
    rows = []
    for filename in sorted(glob.glob(pattern)):
        with open(filename, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError(f"no files matched {pattern!r}")
    return rows


def select(rows: list[dict], attention_type: str, layers: int) -> dict:
    return {
        (row["prefix_tokens"], row["concurrency"]): row
        for row in rows
        if row.get("attention_type") == attention_type
        and row.get("layer_count") == layers
    }


def bar_page(pdf: PdfPages, data: dict, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    x = np.arange(len(CONCURRENCIES))
    width = 0.34
    for ax, prefix in zip(axes.flat, PREFIXES):
        load = [data[(prefix, c)]["load_p50_ms"] for c in CONCURRENCIES]
        recompute = [
            data[(prefix, c)]["recompute_p50_ms"] for c in CONCURRENCIES
        ]
        ax.bar(x - width / 2, load, width, color=LOAD_COLOR,
               label="Load (CPU→GPU)", edgecolor="white")
        ax.bar(x + width / 2, recompute, width, color=RECOMPUTE_COLOR,
               label="Recompute", edgecolor="white")
        ax.set_title(f"Prefix = {prefix} tokens")
        ax.set_xticks(x, [str(c) for c in CONCURRENCIES])
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Time (ms, p50)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig.text(
        0.5, 0.015,
        "Lower is better. Bars include batched copy serialization; the layer "
        "microbenchmark excludes vLLM prefix-match/scheduler overhead.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def ratio_page(pdf: PdfPages, full: dict, linear: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        "Recompute / Load ratio (p50): >1 means Load is faster",
        fontsize=15,
        fontweight="bold",
    )
    for ax, data, title in zip(
        axes,
        [full, linear],
        ["Full Attention (16 layers)", "Linear Attention (48 layers)"],
    ):
        matrix = np.array([
            [
                data[(prefix, c)]["recompute_p50_ms"]
                / data[(prefix, c)]["load_p50_ms"]
                for c in CONCURRENCIES
            ]
            for prefix in PREFIXES
        ])
        # Center the diverging map at 1.  The logarithm makes both sides
        # readable because the ratios span more than one order of magnitude.
        image = ax.imshow(np.log2(matrix), cmap="RdBu_r", vmin=-4, vmax=4)
        ax.set_title(title)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Prefix tokens")
        ax.set_xticks(range(len(CONCURRENCIES)), CONCURRENCIES)
        ax.set_yticks(range(len(PREFIXES)), PREFIXES)
        for row in range(len(PREFIXES)):
            for col in range(len(CONCURRENCIES)):
                ratio = matrix[row, col]
                winner = "L" if ratio >= 1 else "R"
                ax.text(col, row, f"{ratio:.1f}× {winner}",
                        ha="center", va="center", fontsize=8)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("log₂(recompute/load)")
    fig.text(
        0.5, 0.015,
        "L = Load wins, R = Recompute wins. This is the layer-level crossover "
        "evidence; full TTFT adds matching and scheduler costs.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default="/tmp/qwen35_layer_selection_clean_[0-9]*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/qwen35_load_recompute_layer_types.pdf"),
    )
    args = parser.parse_args()
    rows = read_rows(args.input_glob)
    full = select(rows, "full", 16)
    linear = select(rows, "linear", 48)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        bar_page(pdf, full, "Full Attention: Load vs Recompute")
        bar_page(pdf, linear, "Linear Attention: Load vs Recompute")
        ratio_page(pdf, full, linear)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
