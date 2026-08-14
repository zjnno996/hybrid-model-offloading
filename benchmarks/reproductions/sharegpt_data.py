# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory readers for the large ShareGPT JSON-list dataset."""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records without materializing the ~GB JSON list in RAM."""
    try:
        import ijson
    except ImportError as exc:  # pragma: no cover - benchmark dependency
        raise RuntimeError(
            "ShareGPT benchmarks require the 'ijson' package for bounded-memory "
            "streaming (pip install ijson)."
        ) from exc

    with path.open("rb") as dataset_file:
        for record in ijson.items(dataset_file, "item"):
            if isinstance(record, dict):
                yield record


def get_record(path: Path, record_index: int) -> dict[str, Any]:
    """Read one record by index while keeping memory bounded."""
    if record_index < 0:
        raise ValueError("record_index must be non-negative")
    for index, record in enumerate(iter_records(path)):
        if index == record_index:
            return record
    raise IndexError(f"ShareGPT record index {record_index} is out of range")


def reservoir_sample(
    path: Path, sample_count: int, *, seed: int
) -> list[tuple[int, dict[str, Any]]]:
    """Uniformly sample records in one streaming pass.

    The returned record indices make the selected workload reproducible while
    storing only ``sample_count`` records instead of the complete dataset.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = random.Random(seed)
    sample: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(iter_records(path)):
        item = (index, record)
        if len(sample) < sample_count:
            sample.append(item)
            continue
        replacement = rng.randrange(index + 1)
        if replacement < sample_count:
            sample[replacement] = item
    sample.sort(key=lambda item: item[0])
    return sample
