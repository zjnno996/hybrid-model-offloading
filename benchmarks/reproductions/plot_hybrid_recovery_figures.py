# SPDX-License-Identifier: Apache-2.0
"""Draw provisional paper figures from measured Qwen3.5 recovery points.

The table intentionally contains only measurements produced by
qwen35_hybrid_recovery_crossover.py.  Missing length/concurrency cells stay
empty instead of being interpolated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CONCURRENCIES = [1, 4, 16, 64]
LENGTHS = [512, 784, 1024, 2048]

# (transfer_ms, recompute_ms), p50 values from real Mooncake concurrent runs.
MEASUREMENTS: dict[str, dict[tuple[int, int], tuple[float, float]]] = {
    "Full Attention": {
        (512, 1): (8.348, 3.948),
        (512, 4): (28.461, 10.132),
        (512, 16): (89.136, 35.444),
        (512, 64): (1178.060, 131.780),
        (784, 1): (10.341, 4.991),
        (784, 4): (33.572, 15.050),
        (784, 16): (134.341, 54.028),
        (784, 64): (1578.695, 1547.740),
        (1024, 1): (12.481, 6.583),
        (1024, 4): (42.959, 19.212),
        (1024, 16): (175.366, 71.217),
        (1024, 64): (2502.228, 274.241),
        (2048, 1): (28.438, 1384.472),
        (2048, 4): (75.977, 41.684),
        (2048, 16): (376.227, 156.084),
        (2048, 64): (5349.518, 608.398),
    },
    "Linear/GDN": {
        (512, 1): (28.033, 8.274),
        (512, 4): (86.480, 12.898),
        (512, 16): (497.766, 71.729),
        (512, 64): (4214.722, 288.117),
        (784, 1): (31.298, 8.097),
        (784, 4): (85.089, 24.559),
        (784, 16): (444.672, 110.489),
        (784, 64): (6057.119, 585.874),
        (1024, 1): (29.174, 8.085),
        (1024, 4): (86.424, 33.793),
        (1024, 16): (434.706, 141.906),
        (1024, 64): (6317.627, 568.466),
        (2048, 1): (37.671, 154.505),
        (2048, 4): (85.665, 71.401),
        (2048, 16): (457.196, 290.787),
        (2048, 64): (5439.703, 1140.861),
    },
}


def font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_heatmap(draw, origin, title, values, formatter, cell_w=190, cell_h=80):
    x0, y0 = origin
    title_font = font(25)
    label_font = font(19)
    value_font = font(17)
    draw.text((x0, y0), title, fill="#111827", font=title_font)
    y = y0 + 48
    draw.text((x0 + 120, y), "prefix tokens", fill="#4b5563", font=label_font)
    for col, length in enumerate(LENGTHS):
        x = x0 + 120 + col * cell_w
        draw.text((x + 48, y + 27), str(length), fill="#111827", font=label_font)
    y += 62
    draw.text((x0, y + 24), "concurrency", fill="#4b5563", font=label_font)
    for row, concurrency in enumerate(CONCURRENCIES):
        yy = y + row * cell_h
        draw.text((x0 + 38, yy + 28), str(concurrency), fill="#111827", font=label_font)
        for col, length in enumerate(LENGTHS):
            xx = x0 + 120 + col * cell_w
            point = values.get((length, concurrency))
            if point is None:
                fill = "#f3f4f6"
                text = "not run"
            else:
                transfer, recompute = point
                ratio = transfer / recompute
                # Blue means transfer is cheaper; orange means replay is cheaper.
                if ratio <= 1:
                    fill = "#bfdbfe"
                else:
                    fill = "#fed7aa"
                text = formatter(transfer, recompute, ratio)
            draw.rounded_rectangle(
                (xx, yy, xx + cell_w - 12, yy + cell_h - 10),
                radius=8,
                fill=fill,
                outline="#9ca3af",
                width=1,
            )
            bbox = draw.multiline_textbbox((0, 0), text, font=value_font, spacing=2)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.multiline_text(
                (xx + (cell_w - 12 - tw) / 2, yy + (cell_h - 10 - th) / 2),
                text,
                fill="#111827",
                font=value_font,
                align="center",
                spacing=2,
            )


def draw_strategy(draw, origin, cell_w=190, cell_h=80):
    x0, y0 = origin
    title_font = font(25)
    label_font = font(19)
    value_font = font(17)
    draw.text(
        (x0, y0),
        "Candidate hybrid policy: Full KV load + Linear choice",
        fill="#111827",
        font=title_font,
    )
    y = y0 + 48
    draw.text((x0 + 120, y), "prefix tokens", fill="#4b5563", font=label_font)
    for col, length in enumerate(LENGTHS):
        x = x0 + 120 + col * cell_w
        draw.text((x + 48, y + 27), str(length), fill="#111827", font=label_font)
    y += 62
    draw.text((x0, y + 24), "concurrency", fill="#4b5563", font=label_font)
    for row, concurrency in enumerate(CONCURRENCIES):
        yy = y + row * cell_h
        draw.text((x0 + 38, yy + 28), str(concurrency), fill="#111827", font=label_font)
        for col, length in enumerate(LENGTHS):
            xx = x0 + 120 + col * cell_w
            full = MEASUREMENTS["Full Attention"].get((length, concurrency))
            linear = MEASUREMENTS["Linear/GDN"].get((length, concurrency))
            if full is None or linear is None:
                fill, text = "#f3f4f6", "not run"
            else:
                linear_load, linear_recompute = linear
                linear_choice = "L" if linear_load <= linear_recompute else "R"
                text = f"Full: L\nLinear: {linear_choice}"
                fill = "#dcfce7" if linear_choice == "R" else "#dbeafe"
            draw.rounded_rectangle(
                (xx, yy, xx + cell_w - 12, yy + cell_h - 10),
                radius=8,
                fill=fill,
                outline="#9ca3af",
                width=1,
            )
            bbox = draw.multiline_textbbox((0, 0), text, font=value_font, spacing=2)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.multiline_text(
                (xx + (cell_w - 12 - tw) / 2, yy + (cell_h - 10 - th) / 2),
                text,
                fill="#111827",
                font=value_font,
                align="center",
                spacing=2,
            )
    legend_y = y + len(CONCURRENCIES) * cell_h + 20
    draw.text(
        (x0, legend_y),
        "L = Mooncake load, R = prefix replay; Linear load does not remove replay compute in this endpoint path",
        fill="#4b5563",
        font=label_font,
    )


def draw_line_chart(draw, origin, size, title, type_name):
    """Draw transfer/replay curves on a logarithmic millisecond axis."""
    x0, y0 = origin
    width, height = size
    title_font = font(25)
    label_font = font(17)
    small_font = font(14)
    draw.text((x0, y0), title, fill="#111827", font=title_font)
    left, top = x0 + 72, y0 + 55
    right, bottom = x0 + width - 24, y0 + height - 56
    draw.rectangle((left, top, right, bottom), outline="#6b7280", width=2)
    y_min, y_max = 1.0, 10000.0

    def x_pos(concurrency):
        index = CONCURRENCIES.index(concurrency)
        return left + index * (right - left) / (len(CONCURRENCIES) - 1)

    def y_pos(value):
        import math

        fraction = (math.log10(value) - math.log10(y_min)) / (
            math.log10(y_max) - math.log10(y_min)
        )
        return bottom - fraction * (bottom - top)

    for tick in (1, 10, 100, 1000, 10000):
        yy = y_pos(tick)
        draw.line((left, yy, right, yy), fill="#e5e7eb", width=1)
        draw.text((left - 58, yy - 9), str(tick), fill="#4b5563", font=small_font)
    for concurrency in CONCURRENCIES:
        xx = x_pos(concurrency)
        draw.line((xx, top, xx, bottom), fill="#f3f4f6", width=1)
        draw.text((xx - 10, bottom + 12), str(concurrency), fill="#111827", font=label_font)
    draw.text((left + 4, bottom + 39), "concurrency", fill="#4b5563", font=label_font)
    draw.text((x0 + 5, top - 25), "ms (log scale)", fill="#4b5563", font=small_font)

    load_colors = ["#2563eb", "#16a34a", "#0f766e", "#7c3aed"]
    replay_colors = ["#ea580c", "#dc2626", "#c026d3", "#92400e"]
    colors = {
        (length, metric): (load_colors[index] if metric == "load" else replay_colors[index])
        for index, length in enumerate(LENGTHS)
        for metric in ("load", "replay")
    }
    labels = {
        (length, metric): f"{length} {metric}"
        for length in LENGTHS
        for metric in ("load", "replay")
    }
    legend_x, legend_y = left + 12, top + 10
    legend_index = 0
    for key, color in colors.items():
        length, metric = key
        points = []
        for concurrency in CONCURRENCIES:
            measured = MEASUREMENTS[type_name].get((length, concurrency))
            if measured is None:
                continue
            value = measured[0] if metric == "load" else measured[1]
            points.append((x_pos(concurrency), y_pos(value), value))
        if points:
            for first, second in zip(points, points[1:]):
                draw.line((first[0], first[1], second[0], second[1]), fill=color, width=4)
            for xx, yy, value in points:
                draw.ellipse((xx - 6, yy - 6, xx + 6, yy + 6), fill=color, outline="white")
                draw.text((xx + 8, yy - 17), f"{value:.1f}", fill=color, font=small_font)
        lx = legend_x + (legend_index % 2) * 170
        ly = legend_y + (legend_index // 2) * 25
        draw.line((lx, ly + 9, lx + 22, ly + 9), fill=color, width=4)
        draw.text((lx + 28, ly), labels[key], fill="#111827", font=small_font)
        legend_index += 1


def draw_detail_table(draw, origin):
    x0, y0 = origin
    title_font = font(23)
    header_font = font(15)
    body_font = font(14)
    draw.text((x0, y0), "Measured points and candidate hybrid cost", fill="#111827", font=title_font)
    headers = ["type", "prefix", "conc", "bytes", "load ms", "replay ms", "load/replay", "winner"]
    widths = [170, 80, 65, 105, 105, 115, 120, 100]
    row_h = 29
    y = y0 + 42
    x = x0
    for header, cell_w in zip(headers, widths):
        draw.rectangle((x, y, x + cell_w, y + row_h), fill="#dbeafe", outline="#9ca3af")
        draw.text((x + 5, y + 7), header, fill="#111827", font=header_font)
        x += cell_w
    y += row_h
    bytes_by_type = {
        "Full Attention": {length: f"{length * 64 / 1024:.1f}" for length in LENGTHS},
        "Linear/GDN": {length: "146.8" for length in LENGTHS},
    }
    ordered = []
    for type_name in ("Full Attention", "Linear/GDN"):
        for length in LENGTHS:
            for concurrency in CONCURRENCIES:
                point = MEASUREMENTS[type_name].get((length, concurrency))
                if point is not None:
                    ordered.append((type_name, length, concurrency, point))
    for type_name, length, concurrency, (load, replay) in ordered:
        ratio = load / replay
        row = [
            type_name,
            str(length),
            str(concurrency),
            bytes_by_type[type_name][length],
            f"{load:.1f}",
            f"{replay:.1f}",
            f"{ratio:.2f}x",
            "load" if ratio <= 1 else "replay",
        ]
        x = x0
        fill = "#eff6ff" if ratio <= 1 else "#fff7ed"
        for value, cell_w in zip(row, widths):
            draw.rectangle((x, y, x + cell_w, y + row_h), fill=fill, outline="#d1d5db")
            draw.text((x + 5, y + 7), value, fill="#111827", font=body_font)
            x += cell_w
        y += row_h

    draw.text(
        (x0, y + 14),
        "Legacy old-P3 = Full KV load + Linear replay; rerun for current P3/P5",
        fill="#166534",
        font=header_font,
    )
    candidate_rows = [
        (784, 1, 10.341 + 8.097, 10.341 + 31.298),
        (784, 4, 33.572 + 24.559, 33.572 + 85.089),
        (784, 16, 134.341 + 110.489, 134.341 + 444.672),
        (784, 64, 1578.695 + 585.874, 1578.695 + 6057.119),
        (2048, 1, 28.438 + 154.505, 28.438 + 37.671),
        (2048, 16, 376.227 + 290.787, 376.227 + 457.196),
    ]
    y += 43
    for length, concurrency, old_p3, load_both in candidate_rows:
        saving = (load_both - old_p3) / load_both * 100
        draw.text(
            (x0, y),
            f"{length:4d} tokens / C{concurrency:<2d}: old-P3={old_p3:7.1f} ms, "
            f"load-both={load_both:7.1f} ms, saving={saving:5.1f}%",
            fill="#111827",
            font=body_font,
        )
        y += 23


def draw_type_grid(draw, origin, type_name, cell_w=250, cell_h=112):
    x0, y0 = origin
    title_font = font(25)
    label_font = font(18)
    value_font = font(17)
    draw.text((x0, y0), f"{type_name}: measured load/replay", fill="#111827", font=title_font)
    y = y0 + 52
    draw.text((x0, y + 36), "prefix", fill="#4b5563", font=label_font)
    for col, concurrency in enumerate(CONCURRENCIES):
        xx = x0 + 110 + col * cell_w
        draw.text((xx + 92, y + 36), f"C{concurrency}", fill="#111827", font=label_font)
    y += 78
    bytes_by_type = {
        "Full Attention": {
            length: f"{length * 64 / 1024:.1f} MiB" for length in LENGTHS
        },
        "Linear/GDN": {length: "146.8 MiB" for length in LENGTHS},
    }
    for length in LENGTHS:
        yy = y + LENGTHS.index(length) * cell_h
        draw.text((x0 + 35, yy + 38), str(length), fill="#111827", font=label_font)
        draw.text((x0 + 14, yy + 66), bytes_by_type[type_name][length], fill="#6b7280", font=font(13))
        for col, concurrency in enumerate(CONCURRENCIES):
            xx = x0 + 110 + col * cell_w
            point = MEASUREMENTS[type_name].get((length, concurrency))
            if point is None:
                fill, text = "#f3f4f6", "not measured"
            else:
                load, replay = point
                ratio = load / replay
                winner = "LOAD" if ratio <= 1 else "REPLAY"
                fill = "#bfdbfe" if ratio <= 1 else "#fed7aa"
                text = f"load {load:.1f} ms\nreplay {replay:.1f} ms\n{winner}  ({ratio:.2f}x)"
            draw.rounded_rectangle(
                (xx, yy, xx + cell_w - 14, yy + cell_h - 12),
                radius=9,
                fill=fill,
                outline="#9ca3af",
                width=1,
            )
            bbox = draw.multiline_textbbox((0, 0), text, font=value_font, spacing=3)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.multiline_text(
                (xx + (cell_w - 14 - tw) / 2, yy + (cell_h - 12 - th) / 2),
                text,
                fill="#111827",
                font=value_font,
                align="center",
                spacing=3,
            )
    legend_y = y + len(LENGTHS) * cell_h + 22
    draw.text(
        (x0, legend_y),
        "Blue = load faster; orange = replay faster; values are p50 and include direct Mooncake matching/transfer contention.",
        fill="#4b5563",
        font=font(15),
    )


def draw_bar_figure(type_name, output_path: Path) -> None:
    """Draw a four-panel grouped bar chart with only load/replay legends."""
    image = Image.new("RGB", (1680, 1240), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(29)
    panel_title_font = font(22)
    label_font = font(17)
    value_font = font(14)
    draw.text(
        (45, 25),
        f"{type_name}: Mooncake load vs prefix replay",
        fill="#111827",
        font=title_font,
    )
    # Exactly two legend entries, as requested.
    legend_y = 75
    draw.rectangle((55, legend_y, 78, legend_y + 23), fill="#2563eb")
    draw.text((88, legend_y + 2), "load", fill="#111827", font=label_font)
    draw.rectangle((180, legend_y, 203, legend_y + 23), fill="#ea580c")
    draw.text((213, legend_y + 2), "compute/replay", fill="#111827", font=label_font)
    draw.text(
        (420, legend_y + 2),
        "bar height uses logarithmic milliseconds; labels show measured p50",
        fill="#4b5563",
        font=label_font,
    )

    panel_w, panel_h = 780, 500
    panel_origins = [(45, 125), (855, 125), (45, 675), (855, 675)]
    y_min, y_max = 1.0, 10000.0

    def y_pos(value, top, bottom):
        import math

        fraction = (math.log10(value) - math.log10(y_min)) / (
            math.log10(y_max) - math.log10(y_min)
        )
        return bottom - fraction * (bottom - top)

    for length, (x0, y0) in zip(LENGTHS, panel_origins):
        draw.text((x0, y0), f"prefix = {length} tokens", fill="#111827", font=panel_title_font)
        left, top = x0 + 72, y0 + 48
        right, bottom = x0 + panel_w - 24, y0 + panel_h - 60
        draw.rectangle((left, top, right, bottom), outline="#6b7280", width=2)
        for tick in (1, 10, 100, 1000, 10000):
            yy = y_pos(tick, top, bottom)
            draw.line((left, yy, right, yy), fill="#e5e7eb", width=1)
            draw.text((left - 55, yy - 9), str(tick), fill="#4b5563", font=value_font)
        draw.text((x0 + 5, top - 24), "ms (log)", fill="#4b5563", font=value_font)
        group_width = (right - left) / len(CONCURRENCIES)
        bar_width = min(45, group_width * 0.24)
        for index, concurrency in enumerate(CONCURRENCIES):
            center = left + (index + 0.5) * group_width
            draw.text((center - 14, bottom + 13), f"C{concurrency}", fill="#111827", font=label_font)
            point = MEASUREMENTS[type_name].get((length, concurrency))
            if point is None:
                draw.text((center - 27, top + 26), "not run", fill="#6b7280", font=value_font)
                continue
            load, replay = point
            for offset, value, color in ((-bar_width * 0.60, load, "#2563eb"), (bar_width * 0.60, replay, "#ea580c")):
                xx = center + offset - bar_width / 2
                yy = y_pos(value, top, bottom)
                draw.rectangle((xx, yy, xx + bar_width, bottom), fill=color)
                label_y = max(yy - 19, top + 2)
                draw.text((xx - 3, label_y), f"{value:.1f}", fill=color, font=value_font)
        draw.text((left + 4, bottom + 39), "concurrency", fill="#4b5563", font=value_font)
    image.save(output_path)


def draw_four_policy_figure(output_path: Path) -> None:
    """Compare four layer-group recovery policies (not endpoint TTFT).

    P1/P2 are the all-load and all-replay baselines. P3 is the P5-style
    Full+Linear load path, and P4 is the remaining mixed hybrid. Each total is
    the sum of the independently measured Full and Linear p50 costs for the
    same prefix/concurrency point.
    """
    policies = (
        ("P1  load + load", "#2563eb"),
        ("P2  replay + replay", "#ea580c"),
        ("P3  Full load + Linear load", "#16a34a"),
        ("P4  Full replay + Linear load", "#9333ea"),
    )
    image = Image.new("RGB", (1760, 1320), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(29)
    panel_title_font = font(22)
    label_font = font(16)
    value_font = font(12)
    draw.text((45, 24), "Qwen3.5-27B: four hybrid recovery policies", fill="#111827", font=title_font)
    draw.text(
        (45, 61),
        "layer-group recovery cost, not endpoint TTFT; total p50 = Full + Linear; log-scale ms",
        fill="#4b5563",
        font=label_font,
    )
    draw.text(
        (45, 80),
        "P4 transfers Linear state, but full prefix replay still recomputes Linear outputs (current endpoint path)",
        fill="#7c2d12",
        font=font(14),
    )
    legend_x = 45
    for label, color in policies:
        draw.rectangle((legend_x, 96, legend_x + 22, 118), fill=color)
        draw.text((legend_x + 30, 97), label, fill="#111827", font=label_font)
        legend_x += 300

    panel_w, panel_h = 820, 535
    panel_origins = [(45, 145), (895, 145), (45, 700), (895, 700)]
    y_min, y_max = 1.0, 30000.0

    def y_pos(value: float, top: int, bottom: int) -> float:
        import math

        fraction = (math.log10(value) - math.log10(y_min)) / (
            math.log10(y_max) - math.log10(y_min)
        )
        return bottom - fraction * (bottom - top)

    for length, (x0, y0) in zip(LENGTHS, panel_origins):
        draw.text((x0, y0), f"prefix = {length} tokens", fill="#111827", font=panel_title_font)
        left, top = x0 + 70, y0 + 48
        right, bottom = x0 + panel_w - 25, y0 + panel_h - 65
        draw.rectangle((left, top, right, bottom), outline="#6b7280", width=2)
        for tick in (1, 10, 100, 1000, 10000):
            yy = y_pos(tick, top, bottom)
            draw.line((left, yy, right, yy), fill="#e5e7eb", width=1)
            draw.text((left - 56, yy - 8), str(tick), fill="#4b5563", font=value_font)
        draw.text((x0 + 4, top - 22), "total ms (log)", fill="#4b5563", font=value_font)
        group_width = (right - left) / len(CONCURRENCIES)
        bar_width = min(34, group_width * 0.17)
        for index, concurrency in enumerate(CONCURRENCIES):
            center = left + (index + 0.5) * group_width
            draw.text((center - 14, bottom + 13), f"C{concurrency}", fill="#111827", font=label_font)
            full = MEASUREMENTS["Full Attention"].get((length, concurrency))
            linear = MEASUREMENTS["Linear/GDN"].get((length, concurrency))
            if full is None or linear is None:
                continue
            full_load, full_replay = full
            linear_load, linear_replay = linear
            totals = (
                full_load + linear_load,
                full_replay + linear_replay,
                full_load + linear_replay,
                full_replay + linear_load,
            )
            best = min(totals)
            for policy_index, ((_, color), value) in enumerate(zip(policies, totals)):
                offset = (policy_index - 1.5) * (bar_width + 5)
                xx = center + offset - bar_width / 2
                yy = y_pos(value, top, bottom)
                draw.rectangle((xx, yy, xx + bar_width, bottom), fill=color)
                draw.text((xx - 3, max(yy - 17, top + 2)), f"{value:.0f}", fill=color, font=value_font)
                if value == best:
                    draw.text((xx + 7, max(yy - 32, top + 2)), "*", fill="#111827", font=font(18))
        draw.text((left + 4, bottom + 39), "concurrency; * = fastest policy", fill="#4b5563", font=value_font)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure = Image.new("RGB", (1040, 1400), "white")
    draw = ImageDraw.Draw(figure)
    draw_heatmap(
        draw,
        (40, 35),
        "Full Attention: transfer / recompute (ms)",
        MEASUREMENTS["Full Attention"],
        lambda t, r, ratio: f"{t:.1f} / {r:.1f}\n{'load' if ratio <= 1 else 'replay'} {ratio:.2f}x",
    )
    draw_heatmap(
        draw,
        (40, 700),
        "Linear/GDN: transfer / recompute (ms)",
        MEASUREMENTS["Linear/GDN"],
        lambda t, r, ratio: f"{t:.1f} / {r:.1f}\n{'load' if ratio <= 1 else 'replay'} {ratio:.2f}x",
    )
    draw.text(
        (40, 1320),
        "Blue: load wins   Orange: replay wins   Gray: not measured",
        fill="#4b5563",
        font=font(18),
    )
    figure.save(args.output_dir / "qwen35_recovery_cost_heatmaps.png")

    policy = Image.new("RGB", (1040, 570), "white")
    policy_draw = ImageDraw.Draw(policy)
    draw_strategy(policy_draw, (40, 30))
    policy.save(args.output_dir / "qwen35_hybrid_policy_heatmap.png")

    detailed = Image.new("RGB", (1680, 1700), "white")
    detailed_draw = ImageDraw.Draw(detailed)
    draw_line_chart(detailed_draw, (40, 30), (760, 510), "Full Attention recovery", "Full Attention")
    draw_line_chart(detailed_draw, (860, 30), (760, 510), "Linear/GDN recovery", "Linear/GDN")
    draw_detail_table(detailed_draw, (40, 590))
    detailed.save(args.output_dir / "qwen35_hybrid_recovery_detailed.png")

    for type_name, filename in (
        ("Full Attention", "full_attention_recovery_detailed.png"),
        ("Linear/GDN", "linear_attention_recovery_detailed.png"),
    ):
        single = Image.new("RGB", (1440, 1370), "white")
        single_draw = ImageDraw.Draw(single)
        draw_type_grid(single_draw, (40, 35), type_name)
        draw_line_chart(single_draw, (40, 700), (1360, 560), f"{type_name}: latency versus concurrency", type_name)
        single.save(args.output_dir / filename)
    draw_bar_figure("Full Attention", args.output_dir / "full_attention_recovery_bars.png")
    draw_bar_figure("Linear/GDN", args.output_dir / "linear_attention_recovery_bars.png")
    draw_four_policy_figure(args.output_dir / "qwen35_four_policy_comparison.png")
    print(args.output_dir / "qwen35_recovery_cost_heatmaps.png")
    print(args.output_dir / "qwen35_hybrid_policy_heatmap.png")
    print(args.output_dir / "qwen35_hybrid_recovery_detailed.png")
    print(args.output_dir / "full_attention_recovery_detailed.png")
    print(args.output_dir / "linear_attention_recovery_detailed.png")
    print(args.output_dir / "full_attention_recovery_bars.png")
    print(args.output_dir / "linear_attention_recovery_bars.png")
    print(args.output_dir / "qwen35_four_policy_comparison.png")


if __name__ == "__main__":
    main()
