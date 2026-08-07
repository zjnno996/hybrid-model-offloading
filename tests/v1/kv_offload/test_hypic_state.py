# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.kv_offload.hypic_state import HypicPlanner, HypicTransition


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

