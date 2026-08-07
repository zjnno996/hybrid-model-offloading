# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.attention.backends.kvbuffer import KVBuffer, KVBufferConfig


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
