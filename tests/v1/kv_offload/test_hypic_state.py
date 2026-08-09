# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.kv_offload.hypic_state import (
    HypicPolicy,
    HypicSegmentCache,
    HypicPlanner,
    HypicTransition,
)


def test_hypic_affine_segments_compose_in_prefix_order():
    first = HypicTransition(torch.tensor([[2.0]]), torch.tensor([1.0]))
    second = HypicTransition(torch.tensor([[3.0]]), torch.tensor([4.0]))
    combined = HypicPlanner.compose_segments((first, second))
    assert combined is not None
    # second(first(2)) == 3 * (2 * 2 + 1) + 4 == 19.
    assert torch.equal(combined.apply(torch.tensor([2.0])), torch.tensor([19.0]))


def test_hypic_recovery_only_recomputes_the_seam():
    plan = HypicPlanner(segment_tokens=128).recovery_plan(300)
    assert plan.cached_segments == 2
    assert plan.boundary_tokens == 256
    assert plan.seam_recompute_tokens == 44


def test_hypic_segment_cache_matches_independent_segments_and_reports_seams():
    cache = HypicSegmentCache(segment_tokens=4, policy=HypicPolicy(seam_tokens=1))
    transition = HypicTransition(torch.eye(1), torch.ones(1))
    cache.put((1, 2, 3, 4), transition)
    cache.put((5, 6, 7, 8), transition)

    match = cache.match((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

    assert match.hit_segments == 2
    assert match.hit_tokens == 8
    # The first cached segment is safe; the second needs one replayed seam.
    assert match.seam_ranges == ((4, 5),)
    combined = cache.compose([entry for entry in match.entries if entry is not None])
    assert combined is not None
    assert torch.equal(combined.apply(torch.zeros(1)), torch.tensor([2.0]))


def test_hypic_segment_cache_validates_hash_with_tokens():
    cache = HypicSegmentCache(segment_tokens=2)
    entry = cache.put((1, 2), HypicTransition(torch.eye(1), torch.ones(1)))
    # Simulate a digest collision; the token tuple prevents a false hit.
    cache._entries[cache._segment_key((9, 9))] = entry
    assert cache.match((9, 9, 3)).hit_segments == 0


def test_hypic_disabled_policy_keeps_cache_entries_inert():
    cache = HypicSegmentCache(
        segment_tokens=3,
        policy=HypicPolicy(enabled=False),
    )
    transition = HypicTransition(torch.eye(2), torch.zeros(2))
    cache.put((1, 2, 3), transition)
    assert cache.match((1, 2, 3, 4)).hit_segments == 0
