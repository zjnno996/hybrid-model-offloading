# HYPIC (2026)

Branch: `repro/hypic-2026`

The official HYPIC artifact is an SGLang fork (`/data/vllm-paper-references/HYPIC`),
so its kernels are not directly portable to vLLM.  This branch reproduces the
portable core: a linear-attention state transition is represented as an affine
operator `(A, b)`, segment operators compose as
`(A2 @ A1, A2 @ b1 + b2)`, and a prefix hit loads all complete segment
operators while recomputing only the boundary seam.

`HypicTransition` works with torch tensors and does not assume a particular
Mamba/GDN state shape.  `HypicPlanner.recovery_plan()` gives the scheduler the
number of complete cached segments and seam tokens to execute.

`HypicSegmentCache` now ports the PICache lookup contract from the official
artifact: non-final segments may hit independently, cached token IDs are
checked after hashing, and `transition_rope_recompute` style policies report
only the interior seam ranges that the model kernel must replay. This is the
runtime-facing cache component; applying RoPE re-rotation remains specific to
the supported model kernel.

Validation:

```text
python -m compileall -q vllm/v1/kv_offload/hypic_state.py \
  tests/v1/kv_offload/test_hypic_state.py
python -m pytest tests/v1/kv_offload/test_hypic_state.py
```

Benchmark: `python benchmarks/reproductions/hypic_2026.py`.
