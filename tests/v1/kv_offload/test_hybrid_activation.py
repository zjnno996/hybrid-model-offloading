# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.kv_offload.hybrid_activation import (
    ActivationSegment,
    HybridActivationContext,
    LMCacheActivationBackend,
    HybridActivationStore,
)


def test_activation_store_round_trip_for_chunked_prefix():
    store = HybridActivationStore()
    key = "prefix"
    first_hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    first_residual = first_hidden + 100
    second_hidden = first_hidden + 10
    second_residual = first_residual + 10
    store.capture(key, 3, first_hidden, first_residual, 0, 2, 4)
    store.capture(key, 3, second_hidden, second_residual, 2, 2, 4)

    hidden, residual = store.restore(key, 3, 1, 2, torch.device("cpu"), torch.float32)
    assert torch.equal(hidden, torch.stack((first_hidden[1], second_hidden[0])))
    assert torch.equal(
        residual, torch.stack((first_residual[1], second_residual[0]))
    )


def test_activation_context_refuses_mixed_batch_restore():
    store = HybridActivationStore()
    hidden = torch.ones(2, 4)
    store.capture("a", 0, hidden, hidden, 0, 2, 2)
    context = HybridActivationContext(
        store,
        [ActivationSegment("a", 0, 2, 0, 2, False)],
        expected_tokens=2,
        restore_layer_type="full_attention",
    )
    assert (
        context.restore(
            0,
            "full_attention",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        is None
    )


class _FakeRecoveryStore:
    def __init__(self):
        self.saved = {}

    def store_boundary(self, token_ids, hidden_states, residual, *, layer_idx):
        self.saved[(tuple(token_ids), layer_idx)] = (
            hidden_states.clone(),
            residual.clone(),
        )
        return 1

    def retrieve_boundary(self, token_ids, *, layer_idx):
        value = self.saved.get((tuple(token_ids), layer_idx))
        if value is None:
            return None
        return type("Boundary", (), {
            "hidden_states": value[0],
            "residual": value[1],
        })()


def test_lmcache_backend_round_trip():
    recovery = _FakeRecoveryStore()
    backend = LMCacheActivationBackend(recovery)
    key = "prefix"
    token_ids = [11, 12, 13, 14]
    backend.register_tokens(key, token_ids)
    hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    residual = hidden + 10
    backend.capture(key, 3, hidden[:2], residual[:2], 0, 2, 4)
    backend.capture(key, 3, hidden[2:], residual[2:], 2, 2, 4)
    restored = backend.restore(
        key, 3, 1, 2, torch.device("cpu"), torch.float32
    )
    assert restored is not None
    assert torch.equal(restored[0], hidden[1:3])
    assert torch.equal(restored[1], residual[1:3])
