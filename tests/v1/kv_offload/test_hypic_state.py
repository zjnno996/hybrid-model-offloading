# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.model_executor.layers.mamba.linear.minimax_linear_attn import (
    MiniMaxText01LinearAttention,
    _hypic_transition,
)
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata
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


def test_hypic_minimax_transition_matches_decayed_state_update():
    torch.manual_seed(1)
    keys = torch.randn(4, 2, 3)
    values = torch.randn(4, 2, 3)
    slopes = torch.tensor([0.1, 0.2])
    transition = _hypic_transition(keys, values, slopes)
    initial = torch.randn(2, 3, 3)
    expected = initial.clone()
    ratio = torch.exp(-slopes).view(2, 1, 1)
    for key, value in zip(keys, values):
        expected = ratio * expected + torch.einsum("hd,he->hde", key, value)
    assert torch.allclose(transition.apply(initial), expected, atol=1e-5)


def test_hypic_minimax_prefill_composes_cached_segments(monkeypatch):
    cache = HypicSegmentCache(segment_tokens=2)
    layer = object.__new__(MiniMaxText01LinearAttention)
    layer._hypic_cache = cache
    layer._hypic_enabled = True
    layer.tp_slope = torch.tensor([0.1])
    layer.BLOCK = 2
    layer.layer_idx = 0

    context = type("Context", (), {"additional_kwargs": {"hypic_request_tokens": {"r": (1, 2, 3, 4, 5)}}})()
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.linear.minimax_linear_attn.get_forward_context",
        lambda: context,
    )

    def prefix(q, k, v, cache_state, slope, block_size, layer_idx=None):
        ratio = torch.exp(-slope).view(1, 1, 1)
        outputs = []
        for q_t, k_t, v_t in zip(q, k, v):
            cache_state.mul_(ratio)
            cache_state.add_(torch.einsum("hd,he->hde", k_t, v_t))
            outputs.append(torch.einsum("hd,hde->he", q_t, cache_state).reshape(-1))
        return torch.stack(outputs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.linear.minimax_linear_attn.MiniMaxText01LinearKernel.jit_linear_forward_prefix",
        staticmethod(prefix),
    )
    q = torch.ones(5, 1, 2)
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    metadata = LinearAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=5,
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc=torch.tensor([0, 5]),
        seq_lens=torch.tensor([5]),
        state_indices_tensor=torch.tensor([0]),
        request_ids=("r",),
    )
    kv_cache = torch.zeros(1, 1, 2, 2)
    first = layer._hypic_prefill_and_mix_infer(q, k, v, kv_cache, metadata.state_indices_tensor, metadata)
    assert first.shape == (5, 2)
    assert cache.match((1, 2, 3, 4, 5)).hit_segments == 2

    kv_cache.zero_()
    second = layer._hypic_prefill_and_mix_infer(q, k, v, kv_cache, metadata.state_indices_tensor, metadata)
    assert second.shape == (5, 2)
    assert torch.allclose(second[-1], first[-1], atol=1e-5)
