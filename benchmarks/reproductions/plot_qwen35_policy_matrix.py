# SPDX-License-Identifier: Apache-2.0
"""Plot measured Qwen3.5 four-policy TTFT by prefix length and concurrency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {"P1": "#2563eb", "P2": "#ea580c", "P3": "#16a34a", "P4": "#9333ea"}
LABELS = {
    "P1": "Full load + Linear load",
    "P2": "Full replay + Linear replay",
    "P3": "Full load + Linear replay",
    "P4": "Full replay + Linear load",
}


def font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def load_jsonl(paths: list[Path]) -> dict[tuple[str, int, int], tuple[float, float]]:
    values: dict[tuple[str, int, int], tuple[float, float]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("ttft_p50_ms") is not None:
                values[(row["policy"], row["prefix_tokens"], row["concurrency"])] = (
                    row["ttft_p50_ms"], row["ttft_p90_ms"]
                )
    return values


def draw_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    title: str,
    concurrency: list[int],
    values: dict[tuple[str, int, int], tuple[float, float]],
    prefix_tokens: int,
    y_max: float,
) -> None:
    x0, y0 = origin
    left, top, right, bottom = x0 + 72, y0 + 55, x0 + 620, y0 + 410
    draw.text((x0, y0), title, fill="#111827", font=font(25))
    draw.rectangle((left, top, right, bottom), outline="#6b7280", width=2)

    def y(value: float) -> float:
        return bottom - min(value, y_max) / y_max * (bottom - top)

    for tick in (0, 2000, 4000, 6000, 8000, 10000):
        if tick > y_max:
            continue
        yy = y(tick)
        draw.line((left, yy, right, yy), fill="#e5e7eb", width=1)
        draw.text((left - 66, yy - 10), f"{tick/1000:.0f}s", fill="#4b5563", font=font(14))

    group_width = (right - left) / max(1, len(concurrency))
    bar_width = min(30, (group_width - 22) / 4)
    for ci, conc in enumerate(concurrency):
        center = left + (ci + 0.5) * group_width
        draw.text((center - 22, bottom + 14), f"C{conc}", fill="#111827", font=font(17))
        for pi, policy in enumerate(("P1", "P2", "P3", "P4")):
            point = values.get((policy, prefix_tokens, conc))
            if point is None:
                continue
            p50, p90 = point
            xx = center + (pi - 1.5) * (bar_width + 5) - bar_width / 2
            yy = y(p50)
            draw.rectangle((xx, yy, xx + bar_width, bottom), fill=COLORS[policy])
            whisker_x = xx + bar_width / 2
            whisker_y = y(p90)
            draw.line((whisker_x, whisker_y, whisker_x, yy), fill="#111827", width=2)
            draw.line((whisker_x - 5, whisker_y, whisker_x + 5, whisker_y), fill="#111827", width=2)

    draw.text((left, bottom + 45), "concurrency", fill="#4b5563", font=font(15))
    draw.text((x0 + 4, top - 28), "TTFT (p50 bars; p90 whiskers)", fill="#4b5563", font=font(15))


def draw_advantage_map(
    values: dict[tuple[str, int, int], tuple[float, float]],
    output: Path,
) -> None:
    cells = [("784--900", 784, [1, 4, 16]), ("512", 512, [32, 64]), ("900", 900, [32, 64])]
    image = Image.new("RGB", (1120, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 24), "Winning recovery policy by prefix length and concurrency", fill="#111827", font=font(27))
    draw.text((35, 66), "Cell text: winner and p50 improvement over P2 replay baseline", fill="#4b5563", font=font(16))
    x0, y0 = 55, 125
    cell_w, cell_h = 230, 100
    for row, (label, length, concurrencies) in enumerate(cells):
        yy = y0 + row * (cell_h + 30)
        draw.text((x0, yy + 34), f"{label} tokens", fill="#111827", font=font(19))
        for col, conc in enumerate(concurrencies):
            xx = x0 + 160 + col * cell_w
            points = {p: values.get((p, length, conc)) for p in ("P1", "P2", "P3", "P4")}
            points = {p: point for p, point in points.items() if point is not None}
            if not points:
                continue
            winner = min(points, key=lambda p: points[p][0])
            p2 = points.get("P2", (None, None))[0]
            gain = 0.0 if p2 is None else (p2 - points[winner][0]) / p2 * 100
            draw.rounded_rectangle((xx, yy, xx + cell_w - 18, yy + cell_h), radius=8, fill=COLORS[winner], outline="#374151", width=1)
            draw.text((xx + 20, yy + 17), f"{winner}  {gain:+.1f}%", fill="white", font=font(24))
            draw.text((xx + 20, yy + 56), f"C{conc}  p50={points[winner][0]/1000:.2f}s", fill="white", font=font(15))
    draw.text((x0 + 160, 510), "Blue=P1  Orange=P2  Green=P3  Purple=P4", fill="#4b5563", font=font(15))
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/root"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values = load_jsonl([
        Path("/root/qwen35_matrix_p1_final.jsonl"),
        Path("/root/qwen35_matrix_p2_final.jsonl"),
        Path("/root/qwen35_matrix_p3_final.jsonl"),
        Path("/root/qwen35_matrix_p3_diag.jsonl"),
        Path("/root/qwen35_matrix_p4_final.jsonl"),
    ])
    # Earlier real endpoint runs used the same deterministic ShareGPT workload
    # at 784--900 tokens and provide the low-concurrency comparison points.
    low = {
        ("P1", 784, 1): (115.780, 142.753), ("P1", 784, 4): (350.921, 406.274), ("P1", 784, 16): (781.051, 1162.822),
        ("P2", 784, 1): (202.822, 206.101), ("P2", 784, 4): (555.213, 688.157), ("P2", 784, 16): (1951.838, 1957.115),
        ("P3", 784, 1): (227.551, 230.879), ("P3", 784, 4): (741.505, 845.383), ("P3", 784, 16): (2323.675, 2505.382),
        ("P4", 784, 1): (280.232, 288.516), ("P4", 784, 4): (755.844, 882.862), ("P4", 784, 16): (2328.866, 2621.470),
    }
    values.update(low)

    image = Image.new("RGB", (1450, 1120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((45, 25), "Qwen3.5-27B: Hybrid recovery TTFT matrix", fill="#111827", font=font(31))
    draw.text((45, 68), "Real ShareGPT + Mooncake CPU offload; bars=p50, whiskers=p90; seed=0", fill="#4b5563", font=font(16))
    legend_x = 55
    for policy in ("P1", "P2", "P3", "P4"):
        draw.rectangle((legend_x, 102, legend_x + 20, 122), fill=COLORS[policy])
        draw.text((legend_x + 28, 101), f"{policy} {LABELS[policy]}", fill="#111827", font=font(14))
        legend_x += 330

    draw_panel(draw, (45, 150), "A. ShareGPT prefixes: 784--900 tokens", [1, 4, 16], values, 784, 3000)
    draw_panel(draw, (750, 150), "B. Exact 512-token prefixes", [32, 64], values, 512, 6000)
    draw_panel(draw, (45, 610), "C. Exact 900-token prefixes", [32, 64], values, 900, 11000)
    draw.text((750, 690), "Interpretation", fill="#111827", font=font(23))
    draw.multiline_text(
        (750, 732),
        "At 900 tokens/C64, P4 has lower p50 than P3,\n"
        "but its p90/p99 tail is worse. This is a local\n"
        "contention crossover, not a universal Linear-load win.",
        fill="#374151", font=font(18), spacing=8,
    )
    output = args.output_dir / "qwen35_policy_length_concurrency.png"
    image.save(output)
    image.save(
        args.output_dir / "qwen35_policy_length_concurrency.pdf",
        "PDF",
        resolution=150.0,
    )
    draw_advantage_map(values, args.output_dir / "qwen35_policy_advantage_map.png")
    # Keep a vector-friendly PDF copy for paper figures.
    advantage = Image.open(args.output_dir / "qwen35_policy_advantage_map.png").convert(
        "RGB"
    )
    advantage.save(
        args.output_dir / "qwen35_policy_advantage_map.pdf",
        "PDF",
        resolution=150.0,
    )
    print(output)


if __name__ == "__main__":
    main()
