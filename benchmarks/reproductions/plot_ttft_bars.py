# SPDX-License-Identifier: Apache-2.0
"""Plot reproducible ShareGPT TTFT measurements for four recovery policies."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def f(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    # p50/p90 from real endpoint runs; seed=0, ShareGPT 784--900 tokens.
    # P1/P2 are the all-load/all-replay endpoints. P3 is now the P5-style
    # Full+Linear load path, kept under the old label for script compatibility.
    # P4 still measures the mixed Mooncake replay path.
    concurrency = [1, 4, 16, 64]
    policies = [
        ("P1 Full+Linear load", "#2563eb", [106.800, 369.349, 944.585, 2703.681], [191.034, 371.986, 1344.801, 4573.574]),
        ("P2 Full+Linear replay", "#ea580c", [204.402, 590.720, 2225.067, 7626.911], [289.861, 595.881, 2232.356, 8924.420]),
        ("P3 Full+Linear load", "#16a34a", [343.323, 686.567, 2497.585, 8779.295], [343.323, 689.384, 2508.445, 9394.214]),
        ("P4 Full replay+Linear load*", "#9333ea", [364.998, 769.888, 2048.732, 5928.578], [364.998, 776.466, 2623.796, 9773.300]),
    ]

    image = Image.new("RGB", (1280, 820), "white")
    draw = ImageDraw.Draw(image)
    draw.text((45, 30), "Qwen3.5-27B ShareGPT: four-policy TTFT", fill="#111827", font=f(30))
    legend_x = 55
    for label, color, _, _ in policies:
        draw.rectangle((legend_x, 82, legend_x + 22, 107), fill=color)
        draw.text((legend_x + 30, 84), label, fill="#111827", font=f(15))
        legend_x += 250
    draw.text(
        (1030, 84),
        "bars=p50, whisker=p90; seed=0; endpoint TTFT",
        fill="#4b5563",
        font=f(13),
    )

    left, top, right, bottom = 105, 145, 1190, 700
    draw.rectangle((left, top, right, bottom), outline="#6b7280", width=2)
    y_min, y_max = 1.0, 10000.0

    def y(value: float) -> float:
        import math

        fraction = (math.log10(value) - math.log10(y_min)) / (
            math.log10(y_max) - math.log10(y_min)
        )
        return bottom - fraction * (bottom - top)

    for tick in (1, 10, 100, 1000, 10000):
        yy = y(tick)
        draw.line((left, yy, right, yy), fill="#e5e7eb", width=1)
        draw.text((left - 50, yy - 9), str(tick), fill="#4b5563", font=f(15))
    draw.text((left + 4, top - 26), "TTFT (ms, log)", fill="#4b5563", font=f(16))

    group_width = (right - left) / len(concurrency)
    bar_width = 38
    for idx, conc in enumerate(concurrency):
        center = left + (idx + 0.5) * group_width
        draw.text((center - 24, bottom + 18), f"C{conc}", fill="#111827", font=f(20))
        for policy_idx, (_, color, p50_values, p90_values) in enumerate(policies):
            offset = (policy_idx - 1.5) * (bar_width + 8)
            p50, p90 = p50_values[idx], p90_values[idx]
            xx = center + offset - bar_width / 2
            yy = y(p50)
            draw.rectangle((xx, yy, xx + bar_width, bottom), fill=color)
            draw.text((xx + 10, yy - 25), f"{p50:.1f}", fill=color, font=f(17))
            whisker_x = xx + bar_width / 2
            whisker_y = y(p90)
            draw.line((whisker_x, whisker_y, whisker_x, yy), fill="#111827", width=3)
            draw.line((whisker_x - 10, whisker_y, whisker_x + 10, whisker_y), fill="#111827", width=3)
            draw.text((whisker_x + 12, whisker_y - 8), f"p90 {p90:.1f}", fill="#111827", font=f(13))
    draw.text((left + 4, bottom + 58), "concurrency", fill="#4b5563", font=f(17))
    draw.text(
        (45, 755),
        "Real ShareGPT prefixes (4/4/16/64); p50/p90 include HTTP, scheduler, cache match, queue, Mooncake transfer and first-token generation. *P4 loads Linear state but still recomputes Linear during full prefix replay.",
        fill="#4b5563",
        font=f(15),
    )
    output = Path("/root/qwen35_four_policy_ttft.png")
    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
