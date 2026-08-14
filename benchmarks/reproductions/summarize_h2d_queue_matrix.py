# SPDX-License-Identifier: Apache-2.0
"""Join isolated TTFT cells with native offloader H2D queue metrics.

The native offloader emits periodic ``KV Transfer metrics`` lines.  This
utility aggregates those lines from each cell log and joins them with the
JSONL produced by ``run_native_independent_cells.py``.  The queue value is
the CUDA copy-stream wait (waiting for the previous H2D job), not the HTTP or
vLLM scheduler queue; those queues remain included in TTFT.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"KV Transfer metrics: (?P<body>.*)"
)
FIELD_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[0-9.eE+-]+)"
)


def read_metrics(path: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    lines = 0
    if not path.exists():
        return {"metric_lines": 0}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = METRIC_RE.search(line)
        if match is None:
            continue
        lines += 1
        for field in FIELD_RE.finditer(match.group("body")):
            key = field.group("key")
            if key.endswith(("_total_bytes", "_total_time", "_total_queue_time")):
                totals[key] = totals.get(key, 0.0) + float(field.group("value"))
    totals["metric_lines"] = lines
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for line in args.results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    enriched = []
    for row in rows:
        policy = row["policy"]
        concurrency = row["concurrency"]
        metrics = read_metrics(
            args.log_dir / f"native_independent_{policy}_c{concurrency}.log"
        )
        row = dict(row)
        row.update({f"h2d_{key}": value for key, value in metrics.items()})
        requests = max(int(row.get("requests", 1)), 1)
        row["h2d_avg_queue_ms_per_request"] = (
            1000.0 * row.get("h2d_CPU_to_GPU_total_queue_time", 0.0) / requests
        )
        row["h2d_avg_service_ms_per_request"] = (
            1000.0 * row.get("h2d_CPU_to_GPU_total_time", 0.0) / requests
        )
        enriched.append(row)

    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in enriched)
    if text:
        text += "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
