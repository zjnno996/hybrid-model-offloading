# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""I/O-aware KV partial-recompute planner from KVPR (ACL 2025).

The planner is intentionally separate from the transfer worker.  A worker can
use its result to submit a small activation transfer first, recompute that
prefix on the GPU, and overlap the remaining KV transfer with the recompute.
This keeps the policy testable without requiring a particular DMA backend.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVPRConfig:
    """Hardware model used to select the recompute/transfer split."""

    cpu_gpu_bandwidth_bytes_per_second: float = 32e9
    gpu_compute_flops_per_second: float = 312e12
    dtype_bytes: int = 2
    token_step: int = 1


@dataclass(frozen=True)
class KVPRPlan:
    """A partial-recompute execution plan for one prompt batch."""

    total_tokens: int
    recompute_tokens: int
    transfer_tokens: int
    estimated_seconds: float
    activation_bytes: int
    kv_bytes: int

    @property
    def is_partial(self) -> bool:
        return 0 < self.recompute_tokens < self.total_tokens

    def timeline(self, planner: "KVPRPlanner", batch_size: int, hidden_size: int):
        """Return the overlapped transfer/recompute stages.

        The activation DMA occupies the PCIe link first.  Once it completes,
        GPU recomputation and suffix KV DMA start together, which is the
        critical path used by KVPR's analytical model.
        """
        activation = self.activation_bytes / planner.config.cpu_gpu_bandwidth_bytes_per_second
        suffix_bytes = self.kv_bytes
        suffix = suffix_bytes / planner.config.cpu_gpu_bandwidth_bytes_per_second
        recompute = (
            4 * batch_size * self.recompute_tokens * hidden_size * hidden_size
        ) / planner.config.gpu_compute_flops_per_second
        return (
            ("activation_transfer", 0.0, activation),
            ("prefix_recompute", activation, activation + recompute),
            ("suffix_transfer", activation, activation + suffix),
        )


class KVPRPlanner:
    """Select a split that overlaps prefix recomputation with KV transfer."""

    def __init__(self, config: KVPRConfig | None = None):
        self.config = config or KVPRConfig()
        if self.config.token_step <= 0:
            raise ValueError("token_step must be positive")
        if self.config.cpu_gpu_bandwidth_bytes_per_second <= 0:
            raise ValueError("CPU/GPU bandwidth must be positive")
        if self.config.gpu_compute_flops_per_second <= 0:
            raise ValueError("GPU compute throughput must be positive")

    def estimate_seconds(
        self, total_tokens: int, recompute_tokens: int, batch_size: int, hidden_size: int
    ) -> float:
        """Estimate the overlapped load/recompute time for a split."""
        self._validate(total_tokens, recompute_tokens, batch_size, hidden_size)
        dtype_bytes = self.config.dtype_bytes
        activation_bytes = batch_size * recompute_tokens * hidden_size * dtype_bytes
        # K and V are both transferred for the suffix not recomputed on GPU.
        kv_bytes = (
            2
            * batch_size
            * (total_tokens - recompute_tokens)
            * hidden_size
            * dtype_bytes
        )
        transfer_prefix = activation_bytes / self.config.cpu_gpu_bandwidth_bytes_per_second
        transfer_suffix = kv_bytes / self.config.cpu_gpu_bandwidth_bytes_per_second
        recompute = (
            4 * batch_size * recompute_tokens * hidden_size * hidden_size
        ) / self.config.gpu_compute_flops_per_second
        return transfer_prefix + max(recompute, transfer_suffix)

    def plan(
        self,
        prompt_tokens: int,
        generation_tokens: int,
        batch_size: int,
        hidden_size: int,
    ) -> KVPRPlan:
        """Choose the minimum-cost split over the prompt prefix."""
        if prompt_tokens < 0 or generation_tokens < 0:
            raise ValueError("token counts must be non-negative")
        total_tokens = prompt_tokens + generation_tokens
        candidates = list(range(0, prompt_tokens + 1, self.config.token_step))
        if candidates[-1] != prompt_tokens:
            candidates.append(prompt_tokens)
        recompute_tokens = min(
            candidates,
            key=lambda n: self.estimate_seconds(
                total_tokens, n, batch_size, hidden_size
            ),
        )
        estimated_seconds = self.estimate_seconds(
            total_tokens, recompute_tokens, batch_size, hidden_size
        )
        activation_bytes = batch_size * recompute_tokens * hidden_size * self.config.dtype_bytes
        kv_bytes = (
            2
            * batch_size
            * (total_tokens - recompute_tokens)
            * hidden_size
            * self.config.dtype_bytes
        )
        return KVPRPlan(
            total_tokens=total_tokens,
            recompute_tokens=recompute_tokens,
            transfer_tokens=total_tokens - recompute_tokens,
            estimated_seconds=estimated_seconds,
            activation_bytes=activation_bytes,
            kv_bytes=kv_bytes,
        )

    @staticmethod
    def _validate(
        total_tokens: int, recompute_tokens: int, batch_size: int, hidden_size: int
    ) -> None:
        if total_tokens < 0 or not 0 <= recompute_tokens <= total_tokens:
            raise ValueError("recompute_tokens must be within total_tokens")
        if batch_size <= 0 or hidden_size <= 0:
            raise ValueError("batch_size and hidden_size must be positive")
