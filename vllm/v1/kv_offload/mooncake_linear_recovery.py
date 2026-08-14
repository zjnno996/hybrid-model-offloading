# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost model shared by the Mooncake Linear Attention recovery benchmark.

This module deliberately has no dependency on Mooncake.  It describes the
Gated DeltaNet state which a Mooncake-backed recovery path will eventually
move.  Keeping the layout calculation separate lets the benchmark use a model
configuration even when its checkpoints are not present locally.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GatedDeltaNetStateLayout:
    """Per-layer Gated DeltaNet state layout for one tensor-parallel rank."""

    num_linear_layers: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_state_elements: int
    temporal_state_elements: int
    dtype_bytes: int
    temporal_dtype_bytes: int | None = None

    @property
    def state_elements_per_layer(self) -> int:
        return self.conv_state_elements + self.temporal_state_elements

    @property
    def state_bytes_per_layer(self) -> int:
        return (
            self.conv_state_elements * self.dtype_bytes
            + self.temporal_state_elements * self.effective_temporal_dtype_bytes
        )

    @property
    def effective_temporal_dtype_bytes(self) -> int:
        return self.temporal_dtype_bytes or self.dtype_bytes

    @property
    def state_bytes_per_request(self) -> int:
        return self.num_linear_layers * self.state_bytes_per_layer

    def bytes_for_concurrency(self, concurrency: int) -> int:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        return self.state_bytes_per_request * concurrency


@dataclass(frozen=True)
class FullAttentionKVLayout:
    """Full-Attention K/V cache layout for one tensor-parallel rank."""

    num_full_attention_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int

    @property
    def kv_bytes_per_token(self) -> int:
        # K and V for every full-attention layer.
        return (
            self.num_full_attention_layers
            * self.num_kv_heads
            * self.head_dim
            * 2
            * self.dtype_bytes
        )

    def bytes_for_tokens(self, tokens: int) -> int:
        if tokens < 1:
            raise ValueError("tokens must be positive")
        return self.kv_bytes_per_token * tokens


@dataclass(frozen=True)
class RecoveryDecision:
    """Comparison of a state transfer and state recomputation measurement."""

    path: str
    transfer_ms: float
    recompute_ms: float
    speedup: float


@dataclass(frozen=True)
class RecoveryCosts:
    """Measured costs for one attention group's recovery decision.

    ``transfer_ms`` is the Mooncake RPC/data movement portion.  Queueing and
    lookup are kept separate because they are the parts that change with
    concurrency; ``materialize_ms`` covers the local GPU-side completion of a
    successful load.  ``recompute_ms`` is only the group's replay cost, not a
    full-model prefill.
    """

    queue_ms: float
    lookup_ms: float
    transfer_ms: float
    materialize_ms: float
    recompute_ms: float

    @property
    def load_ms(self) -> float:
        return self.queue_ms + self.lookup_ms + self.transfer_ms + self.materialize_ms

    def choose(self) -> RecoveryDecision:
        return choose_recovery_path(self.load_ms, self.recompute_ms)


def qwen_gated_delta_net_state_layout(
    model_config: Mapping[str, Any],
    *,
    tensor_parallel_size: int = 1,
    dtype_bytes: int = 2,
    temporal_dtype_bytes: int | None = None,
) -> GatedDeltaNetStateLayout:
    """Build the Qwen3.5 Gated DeltaNet state layout from ``config.json``.

    ``model_config`` can be the outer Qwen3.5 config or its ``text_config``.
    The calculation mirrors ``MambaStateShapeCalculator.gated_delta_net_state_shape``.
    """
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if dtype_bytes < 1 or (
        temporal_dtype_bytes is not None and temporal_dtype_bytes < 1
    ):
        raise ValueError("state dtype byte widths must be positive")

    text_config = model_config.get("text_config", model_config)
    required = (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    )
    missing = [name for name in required if name not in text_config]
    if missing:
        raise ValueError(
            "config is not a Qwen Gated DeltaNet configuration; missing "
            + ", ".join(missing)
        )

    num_key_heads = int(text_config["linear_num_key_heads"])
    num_value_heads = int(text_config["linear_num_value_heads"])
    key_head_dim = int(text_config["linear_key_head_dim"])
    value_head_dim = int(text_config["linear_value_head_dim"])
    conv_kernel = int(text_config["linear_conv_kernel_dim"])
    if num_value_heads % tensor_parallel_size:
        raise ValueError("linear_num_value_heads must divide tensor_parallel_size")
    conv_dim = key_head_dim * num_key_heads * 2 + value_head_dim * num_value_heads
    if conv_dim % tensor_parallel_size:
        raise ValueError(
            "Gated DeltaNet conv dimension must divide tensor_parallel_size"
        )

    layer_types = text_config.get("layer_types")
    if layer_types is None:
        raise ValueError("Qwen3.5 config has no layer_types field")
    num_linear_layers = sum(
        layer_type == "linear_attention" for layer_type in layer_types
    )
    if not num_linear_layers:
        raise ValueError("Qwen3.5 config has no linear_attention layers")

    # vLLM's GDN conv cache stores kernel_size - 1 previous tokens.
    conv_state_elements = (conv_dim // tensor_parallel_size) * (conv_kernel - 1)
    temporal_state_elements = (
        (num_value_heads // tensor_parallel_size) * value_head_dim * key_head_dim
    )
    return GatedDeltaNetStateLayout(
        num_linear_layers=num_linear_layers,
        num_value_heads=num_value_heads // tensor_parallel_size,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_state_elements=conv_state_elements,
        temporal_state_elements=temporal_state_elements,
        dtype_bytes=dtype_bytes,
        temporal_dtype_bytes=temporal_dtype_bytes,
    )


def qwen_full_attention_kv_layout(
    model_config: Mapping[str, Any],
    *,
    tensor_parallel_size: int = 1,
    dtype_bytes: int = 2,
) -> FullAttentionKVLayout:
    """Build the Qwen3.5 Full-Attention K/V cache layout.

    The calculation covers only K/V cache bytes, rather than Q projections or
    attention outputs. This is exactly the state a prefix-cache hit restores.
    """
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if dtype_bytes < 1:
        raise ValueError("dtype_bytes must be positive")

    text_config = model_config.get("text_config", model_config)
    required = ("num_key_value_heads", "head_dim", "layer_types")
    missing = [name for name in required if name not in text_config]
    if missing:
        raise ValueError(
            "config is not a Qwen Full-Attention configuration; missing "
            + ", ".join(missing)
        )
    num_kv_heads = int(text_config["num_key_value_heads"])
    if num_kv_heads % tensor_parallel_size:
        raise ValueError("num_key_value_heads must divide tensor_parallel_size")
    num_full_attention_layers = sum(
        layer_type == "full_attention" for layer_type in text_config["layer_types"]
    )
    if not num_full_attention_layers:
        raise ValueError("Qwen3.5 config has no full_attention layers")
    return FullAttentionKVLayout(
        num_full_attention_layers=num_full_attention_layers,
        num_kv_heads=num_kv_heads // tensor_parallel_size,
        head_dim=int(text_config["head_dim"]),
        dtype_bytes=dtype_bytes,
    )


def choose_recovery_path(transfer_ms: float, recompute_ms: float) -> RecoveryDecision:
    """Select the faster materialization path from measured latencies."""
    if transfer_ms < 0 or recompute_ms < 0:
        raise ValueError("recovery times must be non-negative")
    if transfer_ms <= recompute_ms:
        speedup = float("inf") if transfer_ms == 0 else recompute_ms / transfer_ms
        return RecoveryDecision("transfer", transfer_ms, recompute_ms, speedup)
    speedup = float("inf") if recompute_ms == 0 else transfer_ms / recompute_ms
    return RecoveryDecision("recompute", transfer_ms, recompute_ms, speedup)
