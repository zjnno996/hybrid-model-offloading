# SPDX-License-Identifier: Apache-2.0
"""Position-independent linear-state composition used by HYPIC (2026).

For a linear/recurrent attention layer, a token (or segment) can be written as
an affine transition ``s' = A s + b``.  HYPIC caches segment transitions and
composes them at a prefix hit; only the seam tokens after the last complete
segment need to run again.  The implementation accepts torch tensors without
owning a model-specific state layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HypicTransition:
    """Affine state transition ``state -> matrix @ state + bias``."""

    matrix: object
    bias: object

    def apply(self, state: object) -> object:
        return self.matrix @ state + self.bias

    def compose(self, previous: "HypicTransition") -> "HypicTransition":
        """Return ``self(previous(state))``."""
        return HypicTransition(
            self.matrix @ previous.matrix,
            self.matrix @ previous.bias + self.bias,
        )


@dataclass(frozen=True)
class HypicRecoveryPlan:
    prefix_tokens: int
    cached_segments: int
    boundary_tokens: int
    seam_recompute_tokens: int


class HypicPlanner:
    """Plan HYPIC segment composition and seam recomputation."""

    def __init__(self, segment_tokens: int = 128):
        if segment_tokens <= 0:
            raise ValueError("segment_tokens must be positive")
        self.segment_tokens = segment_tokens

    def recovery_plan(self, prefix_tokens: int) -> HypicRecoveryPlan:
        if prefix_tokens < 0:
            raise ValueError("prefix_tokens must be non-negative")
        cached_segments, boundary = divmod(prefix_tokens, self.segment_tokens)
        boundary_tokens = cached_segments * self.segment_tokens
        return HypicRecoveryPlan(
            prefix_tokens=prefix_tokens,
            cached_segments=cached_segments,
            boundary_tokens=boundary_tokens,
            seam_recompute_tokens=boundary,
        )

    @staticmethod
    def compose_segments(
        transitions: Sequence[HypicTransition],
    ) -> HypicTransition | None:
        """Compose cached segments in prefix order."""
        result: HypicTransition | None = None
        for transition in transitions:
            result = transition if result is None else transition.compose(result)
        return result

