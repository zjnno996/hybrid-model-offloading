# Marconi (MLSys 2025)

This branch ports Marconi's hybrid-cache policy into
`vllm.v1.core.marconi_policy`. The policy is allocator-independent so the same
decision works for a dense attention KV block or a recurrent/linear state:

1. **Admission:** retain a candidate when
   `reuse_probability * compute_savings / state_bytes` clears the configured
   threshold.
2. **Eviction:** normalize recency and expected compute savings per byte, then
   evict the lowest-utility entries until the requested capacity is released.
3. **State awareness:** `state_bytes` and `compute_savings` are supplied by the
   model/cache manager, so a large recurrent state is not treated like one
   ordinary KV block.

The implementation is covered by
`tests/v1/core/test_marconi_policy.py`. The original artifact used for the
behavioral baseline is in `/data/vllm-paper-references/Marconi`.

Benchmark: `python benchmarks/reproductions/marconi_mlsys25.py`.

## Experimental vLLM integration

Set `VLLM_MARCONI_CACHE_POLICY=1` to enable the policy in `BlockPool`.  Newly
completed cache blocks are admitted using their KV/state page bytes and an
estimated prefix-recompute saving; released cache blocks are reordered by the
Marconi eviction score. `VLLM_MARCONI_EFFICIENCY_WEIGHT` and
`VLLM_MARCONI_MIN_ADMISSION_EFFICIENCY` tune retention and admission. The
default is disabled and retains vLLM's original queue behavior.
