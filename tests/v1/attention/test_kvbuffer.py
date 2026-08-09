# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.attention.backends.kvbuffer import (
    KVBuffer,
    KVBufferBank,
    KVBufferConfig,
)


def test_kvbuffer_materialization_matches_direct_linear_attention():
    keys = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
    values = torch.tensor([[2.0], [4.0], [3.0]])
    queries = torch.tensor([[1.0, 2.0]])
    buf = KVBuffer(2, 1, KVBufferConfig(buffer_tokens=2, update_chunk_tokens=1))
    buf.append(keys, values)
    expected_state = keys.T @ values
    expected_z = keys.sum(dim=0)
    expected = (queries @ expected_state) / (queries @ expected_z).unsqueeze(-1)
    assert torch.allclose(buf.materialize(queries), expected)
    assert buf.keys.shape[0] <= 2


def test_kvbuffer_speculative_rollback_restores_state_and_buffer():
    buf = KVBuffer(2, 1)
    buf.append(torch.ones(2, 2), torch.ones(2, 1))
    snapshot = buf.snapshot()
    buf.append(torch.full((3, 2), 2.0), torch.full((3, 1), 3.0))
    buf.rollback(snapshot)
    assert torch.equal(buf.state_kv, snapshot.state_kv)
    assert torch.equal(buf.keys, snapshot.keys)
    assert torch.equal(buf.values, snapshot.values)


def test_kvbuffer_large_append_flushes_until_capacity_is_respected():
    buf = KVBuffer(2, 1, KVBufferConfig(buffer_tokens=2, update_chunk_tokens=1))
    buf.append(torch.ones(10, 2), torch.ones(10, 1))
    assert buf.keys.shape[0] == 2


def test_kvbuffer_bank_commits_or_rolls_back_per_state_slot():
    bank = KVBufferBank(2, 1, KVBufferConfig(buffer_tokens=2))
    bank.get(7).append(torch.ones(1, 2), torch.ones(1, 1))
    baseline = bank.get(7).snapshot()

    bank.begin_speculative((7,))
    bank.get(7).append(torch.full((2, 2), 2.0), torch.full((2, 1), 3.0))
    bank.reject_speculative((7,))
    assert torch.equal(bank.get(7).keys, baseline.keys)
    assert torch.equal(bank.get(7).state_kv, baseline.state_kv)

    bank.begin_speculative((7,))
    bank.get(7).append(torch.full((1, 2), 4.0), torch.full((1, 1), 5.0))
    bank.accept_speculative((7,))
    assert bank.get(7).keys.shape[0] == 2
    bank.release((7,))
    assert bank.get(7).keys.shape[0] == 0


def test_kvbuffer_slot_export_and_restore_preserves_unflushed_tail():
    source = KVBufferBank(2, 1, KVBufferConfig(buffer_tokens=4))
    source.get(7).append(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0], [6.0]])
    )
    expected = source.get(7).materialize(torch.tensor([[1.0, 1.0]]))

    target = KVBufferBank(2, 1, KVBufferConfig(buffer_tokens=4))
    target.restore_slots(source.export_slots((7,)))
    assert torch.allclose(
        target.get(7).materialize(torch.tensor([[1.0, 1.0]])), expected
    )
    assert torch.equal(target.get(7).keys, source.get(7).keys)
