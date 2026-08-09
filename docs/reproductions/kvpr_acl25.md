# KVPR (ACL Findings 2025)

KVPR overlaps two operations that ordinary CPU KV offload serializes:

1. transfer a small prefix activation to GPU and recompute its KV values;
2. transfer the remaining KV suffix from CPU while that recomputation runs.

This branch ports the paper's hardware-aware split-point planner to
`vllm.v1.kv_offload.kvpr`. It uses the same cost model as the official KVPR
artifact: activation-transfer time plus the maximum of GPU recompute time and
remaining KV-transfer time. The resulting `KVPRPlan` exposes the exact token
split and byte counts needed by a transfer worker.

The planner is covered by `tests/v1/kv_offload/test_kvpr.py`. The original
FlexGen/Transformers implementation is available at
`/data/vllm-paper-references/KVPR`.

`KVPRPlan.timeline()` exposes the worker schedule directly: activation DMA is
submitted first, then GPU prefix recomputation and KV-suffix DMA begin at the
same timestamp. This makes the overlap and critical path testable without
requiring a particular DMA backend; a CPU offload worker can map these three
stages to its existing streams.

`KVPRExecutor` now implements this dependency protocol with backend callbacks.
It is suitable for a connector that can provide activation DMA and prefix
prefill callbacks; it does not silently alter vLLM's current all-or-nothing
offloaded-prefix scheduling.

Benchmark: `python benchmarks/reproductions/kvpr_acl25.py`.
