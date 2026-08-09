# SPDX-License-Identifier: Apache-2.0
"""Position-independent linear-state composition used by HYPIC (2026).

For a linear/recurrent attention layer, a token (or segment) can be written as
an affine transition ``s' = A s + b``.  HYPIC caches segment transitions and
composes them at a prefix hit; only the seam tokens after the last complete
segment need to run again.  The implementation accepts torch tensors without
owning a model-specific state layout.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
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


class HypicComposeMode(str, Enum):
    """Cache payloads used by the official HYPIC/PIC policies."""

    ADDITION = "addition"
    TRANSITION = "transition"


@dataclass(frozen=True)
class HypicPolicy:
    """Position-independent segment-cache policy.

    ``recompute_seam`` corresponds to HYPIC's
    ``transition_rope_recompute`` mode. RoPE re-rotation itself is model
    kernel-specific, so this generic policy reports the seam tokens that must
    be replayed rather than attempting a model-agnostic rotation.
    """

    compose_mode: HypicComposeMode = HypicComposeMode.TRANSITION
    recompute_seam: bool = True
    seam_tokens: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.seam_tokens < 0:
            raise ValueError("seam_tokens must be non-negative")

    @classmethod
    def from_environment(cls) -> "HypicPolicy":
        """Build the opt-in policy used by a vLLM integration."""
        from vllm import envs

        return cls(enabled=envs.VLLM_HYPIC_ENABLED)


@dataclass(frozen=True)
class HypicSegmentEntry:
    """One position-independent cached segment and its zero-start state."""

    key: bytes
    token_ids: tuple[int, ...]
    transition: HypicTransition


@dataclass(frozen=True)
class HypicMatch:
    """Lookup result for a segmented prompt."""

    ranges: tuple[tuple[int, int], ...]
    entries: tuple[HypicSegmentEntry | None, ...]
    seam_ranges: tuple[tuple[int, int], ...]

    @property
    def hit_segments(self) -> int:
        return sum(entry is not None for entry in self.entries)

    @property
    def hit_tokens(self) -> int:
        return sum(
            end - start
            for (start, end), entry in zip(self.ranges, self.entries)
            if entry is not None
        )


class HypicSegmentCache:
    """In-memory segment cache matching HYPIC's PIC hit semantics.

    Any non-final segment may hit independently. The token tuple is retained
    alongside the hash to make collisions harmless. A caller obtains
    ``seam_ranges`` and sends only those tokens through its model-specific
    re-rotation/recompute path.
    """

    def __init__(self, segment_tokens: int = 128, policy: HypicPolicy | None = None):
        if segment_tokens <= 0:
            raise ValueError("segment_tokens must be positive")
        self.segment_tokens = segment_tokens
        self.policy = policy or HypicPolicy()
        self._entries: dict[bytes, HypicSegmentEntry] = {}

    @classmethod
    def from_environment(cls) -> "HypicSegmentCache":
        """Create the explicitly enabled/disabled vLLM experiment cache."""
        from vllm import envs

        return cls(
            segment_tokens=envs.VLLM_HYPIC_SEGMENT_TOKENS,
            policy=HypicPolicy.from_environment(),
        )

    def segment_ranges(self, token_count: int) -> tuple[tuple[int, int], ...]:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        return tuple(
            (start, min(start + self.segment_tokens, token_count))
            for start in range(0, token_count, self.segment_tokens)
        )

    def put(self, token_ids: Sequence[int], transition: HypicTransition) -> HypicSegmentEntry:
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("cannot cache an empty segment")
        key = self._segment_key(tokens)
        existing = self._entries.get(key)
        if existing is not None and existing.token_ids == tokens:
            return existing
        entry = HypicSegmentEntry(key, tokens, transition)
        self._entries[key] = entry
        return entry

    def match(self, token_ids: Sequence[int]) -> HypicMatch:
        tokens = tuple(int(token) for token in token_ids)
        ranges = self.segment_ranges(len(tokens))
        if not self.policy.enabled:
            return HypicMatch(ranges, tuple(None for _ in ranges), ())
        entries: list[HypicSegmentEntry | None] = []
        seams: list[tuple[int, int]] = []
        for index, (start, end) in enumerate(ranges):
            # The last segment is the active query tail and is recomputed.
            if index == len(ranges) - 1:
                entries.append(None)
                continue
            segment = tokens[start:end]
            entry = self._entries.get(self._segment_key(segment))
            if entry is None or entry.token_ids != segment:
                entries.append(None)
                continue
            entries.append(entry)
            if self.policy.recompute_seam and index > 0:
                seam_end = min(end, start + self.policy.seam_tokens)
                if seam_end > start:
                    seams.append((start, seam_end))
        return HypicMatch(ranges, tuple(entries), tuple(seams))

    def compose(self, entries: Sequence[HypicSegmentEntry]) -> HypicTransition | None:
        """Compose a contiguous ordered run of cached segment transitions."""
        return HypicPlanner.compose_segments([entry.transition for entry in entries])

    @staticmethod
    def _segment_key(tokens: tuple[int, ...]) -> bytes:
        hasher = hashlib.blake2b(digest_size=16)
        for token in tokens:
            hasher.update(token.to_bytes(8, "little", signed=True))
        return hasher.digest()


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
