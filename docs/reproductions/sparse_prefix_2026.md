# Sparse Prefix Caching (2026)

Branch: `repro/sparse-prefix-2026`

This branch reproduces the paper's core hybrid-cache idea: recurrent/linear
attention state is saved at sparse prefix positions, then a hit restores the
deepest checkpoint and recomputes only the suffix.  `SparsePrefixPlanner`
selects positions from an observed prefix-length distribution using weighted
marginal replay savings and a byte budget.  `resume()` is the runtime hit
decision used by a state loader before scheduling suffix tokens.

There was no author implementation available to integrate.  The planner is
deliberately tensor-agnostic so it can be used by vLLM's existing Mamba state
manager and future linear-attention managers without copying model-specific
state layouts.

Validation:

```text
python -m compileall -q vllm/v1/kv_offload/sparse_prefix.py \
  tests/v1/kv_offload/test_sparse_prefix.py
python -m pytest tests/v1/kv_offload/test_sparse_prefix.py
```

Benchmark: `python benchmarks/reproductions/sparse_prefix_2026.py`.
