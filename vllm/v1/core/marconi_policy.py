# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marconi-style admission and eviction policy for hybrid KV/state caches.

Marconi's key observation is that a recurrent state and an attention KV block
have different reuse and memory economics.  This module intentionally keeps
the policy independent of the block allocator: the allocator supplies cache
entries with their state size, expected reuse, and compute savings, and uses
the selected victims when it needs to reclaim capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable


@dataclass(frozen=True)
class MarconiEntry:
    """A cached hybrid state candidate.

    ``compute_savings`` is the work avoided by a future hit.  ``state_bytes``
    includes both attention KV bytes and recurrent-state bytes for this entry.
    ``reuse_probability`` is updated by the serving layer from its request
    trace, and is kept separate from recency so cold-but-expensive states are
    not automatically retained forever.
    """

    key: Hashable
    state_bytes: int
    compute_savings: float
    last_access: int
    reuse_probability: float = 0.0

    @property
    def compute_efficiency(self) -> float:
        """Expected saved compute per byte occupied by this entry."""
        if self.state_bytes <= 0:
            return 0.0
        return (
            self.reuse_probability * max(self.compute_savings, 0.0)
        ) / self.state_bytes


@dataclass(frozen=True)
class MarconiPolicyConfig:
    """Tunable weights matching Marconi's policy surface."""

    # 0 is pure recency; larger values prefer compute-efficient entries.
    efficiency_weight: float = 1.0
    # Do not admit entries whose expected savings per byte is below this value.
    min_admission_efficiency: float = 0.0


class MarconiPolicy:
    """Stateless scoring policy used by a hybrid cache manager."""

    def __init__(self, config: MarconiPolicyConfig | None = None):
        self.config = config or MarconiPolicyConfig()

    def admission_score(self, entry: MarconiEntry) -> float:
        """Return the expected compute savings per byte for an entry."""
        return entry.compute_efficiency

    def should_admit(self, entry: MarconiEntry) -> bool:
        """Whether an entry is worth placing in the finite cache."""
        return self.admission_score(entry) >= self.config.min_admission_efficiency

    def eviction_scores(
        self, entries: list[MarconiEntry], now: int | None = None
    ) -> dict[Hashable, float]:
        """Score entries for retention; the lowest score is evicted first.

        Both terms are normalized over the current candidate set, as in the
        artifact's policy.  Logical timestamps make this deterministic in
        tests and avoid wall-clock noise in a serving process.
        """
        if not entries:
            return {}
        current = max(entry.last_access for entry in entries) if now is None else now
        recencies = {
            entry.key: 1.0 / max(current - entry.last_access, 1)
            for entry in entries
        }
        efficiencies = {entry.key: entry.compute_efficiency for entry in entries}
        normalized_recency = self._normalize(recencies)
        normalized_efficiency = self._normalize(efficiencies)
        weight = self.config.efficiency_weight
        return {
            entry.key: normalized_recency[entry.key]
            + weight * normalized_efficiency[entry.key]
            for entry in entries
        }

    def choose_victims(
        self,
        entries: list[MarconiEntry],
        bytes_to_free: int,
        now: int | None = None,
    ) -> list[MarconiEntry]:
        """Choose the lowest-utility entries until enough bytes are freed."""
        if bytes_to_free <= 0 or not entries:
            return []
        scores = self.eviction_scores(entries, now=now)
        victims: list[MarconiEntry] = []
        freed = 0
        for entry in sorted(entries, key=lambda item: scores[item.key]):
            victims.append(entry)
            freed += max(entry.state_bytes, 0)
            if freed >= bytes_to_free:
                break
        return victims

    @staticmethod
    def _normalize(values: dict[Hashable, float]) -> dict[Hashable, float]:
        if not values:
            return {}
        low = min(values.values())
        high = max(values.values())
        if high <= low:
            return {key: 0.0 for key in values}
        return {key: (value - low) / (high - low) for key, value in values.items()}
