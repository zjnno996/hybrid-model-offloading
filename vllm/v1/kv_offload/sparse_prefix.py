# SPDX-License-Identifier: Apache-2.0
"""Sparse recurrent-state checkpoints for hybrid prefix caching.

Sparse Prefix Caching (2026) observes that a recurrent/linear-attention state
is a checkpoint, rather than a dense per-token KV tensor.  This module keeps
the placement and hit decision independent from a particular state tensor
implementation, so Mamba and other recurrent layers can share the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SparsePrefixConfig:
    """Placement constraints for recurrent-state checkpoints."""

    checkpoint_interval: int = 256
    state_bytes: int = 0
    memory_budget_bytes: int = 0

    def max_checkpoints(self) -> int | None:
        if self.state_bytes <= 0 or self.memory_budget_bytes <= 0:
            return None
        return self.memory_budget_bytes // self.state_bytes


@dataclass(frozen=True)
class SparseCheckpointPlan:
    """Selected checkpoint positions and the expected replay cost."""

    positions: tuple[int, ...]
    expected_recompute_tokens: float
    checkpoint_bytes: int


@dataclass(frozen=True)
class SparseResumePlan:
    """The deepest checkpoint usable for one prefix-cache lookup."""

    request_tokens: int
    checkpoint_tokens: int
    recompute_tokens: int


class SparsePrefixPlanner:
    """Select sparse checkpoints using weighted marginal replay savings."""

    def __init__(self, config: SparsePrefixConfig | None = None):
        self.config = config or SparsePrefixConfig()
        if self.config.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.config.state_bytes < 0 or self.config.memory_budget_bytes < 0:
            raise ValueError("state_bytes and memory_budget_bytes must be non-negative")

    def select_checkpoints(
        self,
        prefix_tokens: int,
        query_lengths: Sequence[int],
        query_probabilities: Sequence[float] | None = None,
    ) -> SparseCheckpointPlan:
        """Place checkpoints for an observed prefix-length distribution.

        Position zero is implicit (the model's initial state).  Each greedy
        step picks the candidate with the largest reduction in expected suffix
        replay.  This is equivalent to the paper's sparse checkpoint objective
        when state sizes are uniform, while also respecting a byte budget.
        """
        if prefix_tokens < 0:
            raise ValueError("prefix_tokens must be non-negative")
        if not query_lengths:
            return SparseCheckpointPlan((0,), 0.0, 0)
        if any(q < 0 or q > prefix_tokens for q in query_lengths):
            raise ValueError("query lengths must lie within prefix_tokens")
        if query_probabilities is None:
            weights = [1.0 / len(query_lengths)] * len(query_lengths)
        else:
            if len(query_probabilities) != len(query_lengths):
                raise ValueError("query_probabilities must match query_lengths")
            if any(p < 0 for p in query_probabilities) or sum(query_probabilities) <= 0:
                raise ValueError("query_probabilities must be non-negative and non-zero")
            total = float(sum(query_probabilities))
            weights = [float(p) / total for p in query_probabilities]

        candidates = list(
            range(self.config.checkpoint_interval, prefix_tokens + 1,
                 self.config.checkpoint_interval)
        )
        max_extra = self.config.max_checkpoints()
        if max_extra is not None:
            max_extra = max(0, max_extra - 1)  # position zero is implicit

        selected: list[int] = [0]

        def expected_recompute(positions: Sequence[int]) -> float:
            return sum(
                weight * (q - max(p for p in positions if p <= q))
                for q, weight in zip(query_lengths, weights)
            )

        while candidates and (max_extra is None or len(selected) - 1 < max_extra):
            current = expected_recompute(selected)
            best = max(
                candidates,
                key=lambda candidate: current
                - expected_recompute((*selected, candidate)),
            )
            gain = current - expected_recompute((*selected, best))
            if gain <= 0:
                break
            selected.append(best)
            selected.sort()
            candidates.remove(best)

        return SparseCheckpointPlan(
            positions=tuple(selected),
            expected_recompute_tokens=expected_recompute(selected),
            checkpoint_bytes=len(selected) * self.config.state_bytes,
        )

    @staticmethod
    def resume(checkpoints: Sequence[int], request_tokens: int) -> SparseResumePlan:
        """Return the deepest checkpoint and suffix that must be replayed."""
        if request_tokens < 0:
            raise ValueError("request_tokens must be non-negative")
        valid = sorted({p for p in checkpoints if 0 <= p <= request_tokens})
        checkpoint = valid[-1] if valid else 0
        return SparseResumePlan(request_tokens, checkpoint, request_tokens - checkpoint)
