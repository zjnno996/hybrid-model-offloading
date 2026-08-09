# KVBuffer (2026)

Branch: `repro/kvbuffer-2026`

This branch reproduces the paper's I/O-aware linear-attention primitive.  A
`KVBuffer` keeps recent K/V tokens in a small materialized buffer, flushes old
tokens to the recurrent `(KᵀV, ΣK)` state in configurable chunks, and can
materialize queries from both state and buffer without forcing an update.
Snapshots implement speculative decoding verification: rejected tokens are
discarded by rollback while accepted tokens remain in the buffer.

The implementation is torch-native and can be connected to vLLM's existing
`LinearAttentionBackend` state indices without changing model-specific kernels.
No author code was available for direct integration.

`KVBufferBank` is the runtime-facing adapter: it owns buffers by
`LinearAttentionMetadata.state_indices_tensor` slot ID and exposes
`begin_speculative`, `accept_speculative`, `reject_speculative`, and `release`.
This gives a model-specific linear-attention kernel a bounded, exact state
container without changing vLLM's underlying Mamba state allocation.

Validation:

```text
python -m compileall -q vllm/v1/attention/backends/kvbuffer.py \
  tests/v1/attention/test_kvbuffer.py
python -m pytest tests/v1/attention/test_kvbuffer.py
```

Benchmark: `python benchmarks/reproductions/kvbuffer_2026.py`.
