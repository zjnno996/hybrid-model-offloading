# SPDX-License-Identifier: Apache-2.0
"""IO-aware linear-attention K/V buffer from KVBuffer (2026).

The committed state is updated in chunks while a small recent K/V buffer stays
materialized.  Queries can attend to the buffer directly, avoiding an HBM
state update for every token.  Snapshots provide speculative verification
rollback without allocating a temporary full state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import torch


@dataclass(frozen=True)
class KVBufferConfig:
    buffer_tokens: int = 64
    update_chunk_tokens: int = 64
    enabled: bool = True

    @classmethod
    def from_environment(cls) -> "KVBufferConfig":
        """Read the opt-in settings used by a vLLM linear-attention adapter."""
        from vllm import envs

        return cls(
            buffer_tokens=envs.VLLM_KVBUFFER_BUFFER_TOKENS,
            update_chunk_tokens=envs.VLLM_KVBUFFER_UPDATE_CHUNK_TOKENS,
            enabled=envs.VLLM_KVBUFFER_ENABLED,
        )


@dataclass
class KVBufferSnapshot:
    state_kv: torch.Tensor
    state_z: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


class KVBuffer:
    """Chunked state updater with a recent-token K/V buffer."""

    def __init__(self, key_dim: int, value_dim: int, config: KVBufferConfig | None = None,
                 *, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
        self.config = config or KVBufferConfig()
        if self.config.buffer_tokens <= 0 or self.config.update_chunk_tokens <= 0:
            raise ValueError("buffer_tokens and update_chunk_tokens must be positive")
        if key_dim <= 0 or value_dim <= 0:
            raise ValueError("key_dim and value_dim must be positive")
        self.state_kv = torch.zeros((key_dim, value_dim), device=device, dtype=dtype)
        self.state_z = torch.zeros((key_dim,), device=device, dtype=dtype)
        self.keys = torch.empty((0, key_dim), device=device, dtype=dtype)
        self.values = torch.empty((0, value_dim), device=device, dtype=dtype)

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append tokens and flush old buffer chunks into the committed state."""
        if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must be [tokens, dim] with equal length")
        if keys.shape[1] != self.keys.shape[1] or values.shape[1] != self.values.shape[1]:
            raise ValueError("K/V dimensions do not match the buffer")
        self.keys = torch.cat((self.keys, keys.to(self.keys)), dim=0)
        self.values = torch.cat((self.values, values.to(self.values)), dim=0)
        self._flush_old_chunks()

    def flush(self) -> None:
        """Commit every buffered token to the linear state."""
        self._update_state(self.keys, self.values)
        self.keys = self.keys[:0]
        self.values = self.values[:0]

    def materialize(self, queries: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        """Attend to committed state plus recent buffered K/V without flushing."""
        state_kv = self.state_kv
        state_z = self.state_z
        if self.keys.shape[0]:
            state_kv = state_kv + self.keys.transpose(0, 1) @ self.values
            state_z = state_z + self.keys.sum(dim=0)
        out = queries @ state_kv
        if normalize:
            out = out / (queries @ state_z).clamp_min(1e-6).unsqueeze(-1)
        return out

    def snapshot(self) -> KVBufferSnapshot:
        return KVBufferSnapshot(
            self.state_kv.clone(), self.state_z.clone(), self.keys.clone(), self.values.clone()
        )

    def rollback(self, snapshot: KVBufferSnapshot) -> None:
        self.state_kv = snapshot.state_kv.clone()
        self.state_z = snapshot.state_z.clone()
        self.keys = snapshot.keys.clone()
        self.values = snapshot.values.clone()

    def export_state(self, *, device: torch.device | str = "cpu") -> KVBufferSnapshot:
        """Return an exact, transportable state including the unflushed tail.

        Keeping the tail is essential: flushing it merely to offload would
        defeat KVBuffer's I/O-saving policy and can change the update cadence
        of an attention kernel that consumes the state incrementally.
        """
        target = torch.device(device)
        return KVBufferSnapshot(
            self.state_kv.to(target).clone(),
            self.state_z.to(target).clone(),
            self.keys.to(target).clone(),
            self.values.to(target).clone(),
        )

    def restore_state(self, snapshot: KVBufferSnapshot) -> None:
        """Restore a state previously produced by :meth:`export_state`."""
        if snapshot.state_kv.shape != self.state_kv.shape:
            raise ValueError("state_kv shape does not match the buffer")
        if snapshot.state_z.shape != self.state_z.shape:
            raise ValueError("state_z shape does not match the buffer")
        self.state_kv.copy_(snapshot.state_kv.to(self.state_kv))
        self.state_z.copy_(snapshot.state_z.to(self.state_z))
        self.keys = snapshot.keys.to(self.state_kv).clone()
        self.values = snapshot.values.to(self.state_kv).clone()

    def _flush_old_chunks(self) -> None:
        while self.keys.shape[0] > self.config.buffer_tokens:
            overflow = self.keys.shape[0] - self.config.buffer_tokens
            commit_tokens = min(overflow, self.config.update_chunk_tokens)
            self._update_state(self.keys[:commit_tokens], self.values[:commit_tokens])
            self.keys = self.keys[commit_tokens:]
            self.values = self.values[commit_tokens:]

    def _update_state(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.shape[0] == 0:
            return
        self.state_kv.add_(keys.transpose(0, 1) @ values)
        self.state_z.add_(keys.sum(dim=0))


@dataclass
class DecayedKVBufferSnapshot:
    state_kv: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


class DecayedKVBuffer:
    """Recent-token buffer for the MiniMax/Bailing linear recurrence.

    These models update each head as ``S_t = exp(-slope) * S_{t-1} +
    outer(k_t, v_t)``.  The committed state is kept in the caller-provided
    tensor (normally a vLLM Mamba cache slot); only the recent tail is owned by
    this object.  Thus enabling KVBuffer does not allocate a second full state
    per request.
    """

    def __init__(
        self,
        state_kv: torch.Tensor,
        slope_rate: torch.Tensor,
        config: KVBufferConfig | None = None,
    ) -> None:
        if state_kv.ndim != 3 or state_kv.shape[1] != state_kv.shape[2]:
            raise ValueError("state_kv must have shape [heads, dim, dim]")
        slopes = slope_rate.reshape(-1).to(device=state_kv.device, dtype=torch.float32)
        if slopes.numel() != state_kv.shape[0]:
            raise ValueError("slope_rate must have one value per head")
        self.state_kv = state_kv
        self.slope_rate = slopes
        self.ratios = torch.exp(-slopes).to(dtype=state_kv.dtype)
        self.config = config or KVBufferConfig()
        if self.config.buffer_tokens <= 0 or self.config.update_chunk_tokens <= 0:
            raise ValueError("buffer_tokens and update_chunk_tokens must be positive")
        self.keys = torch.empty(
            (0, state_kv.shape[0], state_kv.shape[1]),
            device=state_kv.device,
            dtype=state_kv.dtype,
        )
        self.values = torch.empty_like(self.keys)

    def reset(self) -> None:
        self.state_kv.zero_()
        self.keys = self.keys[:0]
        self.values = self.values[:0]

    def append(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Update the recurrence and return one output per input token."""
        if (
            queries.ndim != 3
            or keys.shape != queries.shape
            or values.shape != queries.shape
        ):
            raise ValueError("queries, keys, and values must all be [tokens, heads, dim]")
        outputs: list[torch.Tensor] = []
        for token_idx in range(queries.shape[0]):
            self.keys = torch.cat((self.keys, keys[token_idx : token_idx + 1]), dim=0)
            self.values = torch.cat((self.values, values[token_idx : token_idx + 1]), dim=0)
            self._flush_old_chunks()
            outputs.append(torch.einsum("hd,hde->he", queries[token_idx], self.materialize()))
        return torch.stack(outputs, dim=0) if outputs else queries.new_empty(queries.shape)

    def materialize(self) -> torch.Tensor:
        """Return the state after the committed state and buffered tail."""
        count = self.keys.shape[0]
        if count == 0:
            return self.state_kv
        powers = self.ratios.unsqueeze(0).pow(
            torch.arange(
                count - 1, -1, -1, device=self.state_kv.device, dtype=torch.float32
            ).unsqueeze(1)
        )
        state = self.state_kv * self.ratios.pow(count).view(-1, 1, 1)
        state = state + torch.einsum("thd,the,th->hde", self.keys, self.values, powers)
        return state

    def flush(self) -> None:
        if self.keys.shape[0]:
            self.state_kv.copy_(self.materialize())
            self.keys = self.keys[:0]
            self.values = self.values[:0]

    def snapshot(self) -> DecayedKVBufferSnapshot:
        return DecayedKVBufferSnapshot(
            self.state_kv.clone(), self.keys.clone(), self.values.clone()
        )

    def rollback(self, snapshot: DecayedKVBufferSnapshot) -> None:
        self.state_kv.copy_(snapshot.state_kv.to(self.state_kv))
        self.keys = snapshot.keys.to(self.state_kv).clone()
        self.values = snapshot.values.to(self.state_kv).clone()

    def export_state(
        self, *, device: torch.device | str = "cpu"
    ) -> DecayedKVBufferSnapshot:
        target = torch.device(device)
        return DecayedKVBufferSnapshot(
            self.state_kv.to(target).clone(),
            self.keys.to(target).clone(),
            self.values.to(target).clone(),
        )

    def restore_state(self, snapshot: DecayedKVBufferSnapshot) -> None:
        if snapshot.state_kv.shape != self.state_kv.shape:
            raise ValueError("state_kv shape does not match the buffer")
        self.state_kv.copy_(snapshot.state_kv.to(self.state_kv))
        self.keys = snapshot.keys.to(self.state_kv).clone()
        self.values = snapshot.values.to(self.state_kv).clone()

    def _flush_old_chunks(self) -> None:
        while self.keys.shape[0] > self.config.buffer_tokens:
            overflow = self.keys.shape[0] - self.config.buffer_tokens
            commit_tokens = min(overflow, self.config.update_chunk_tokens)
            commit_keys = self.keys[:commit_tokens]
            commit_values = self.values[:commit_tokens]
            weights = self.ratios.unsqueeze(0).pow(
                torch.arange(
                    commit_tokens - 1,
                    -1,
                    -1,
                    device=self.state_kv.device,
                    dtype=torch.float32,
                ).unsqueeze(1)
            )
            self.state_kv.mul_(self.ratios.pow(commit_tokens).view(-1, 1, 1))
            self.state_kv.add_(
                torch.einsum("thd,the,th->hde", commit_keys, commit_values, weights)
            )
            self.keys = self.keys[commit_tokens:]
            self.values = self.values[commit_tokens:]


class KVBufferBank:
    """Own KVBuffer instances by vLLM linear-state slot ID.

    ``LinearAttentionMetadata.state_indices_tensor`` can be used directly as
    the key. The bank keeps speculative snapshots only for slots participating
    in a draft step, so normal decode does not duplicate recurrent states.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        config: KVBufferConfig | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.config = config or KVBufferConfig()
        self.device = device
        self.dtype = dtype
        self._buffers: dict[Hashable, KVBuffer] = {}
        self._speculative: dict[Hashable, KVBufferSnapshot] = {}

    @classmethod
    def from_environment(
        cls,
        key_dim: int,
        value_dim: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "KVBufferBank":
        """Construct the bank with explicit environment-controlled policy."""
        return cls(
            key_dim,
            value_dim,
            KVBufferConfig.from_environment(),
            device=device,
            dtype=dtype,
        )

    def get(self, state_id: Hashable) -> KVBuffer:
        buffer = self._buffers.get(state_id)
        if buffer is None:
            buffer = KVBuffer(
                self.key_dim,
                self.value_dim,
                self.config,
                device=self.device,
                dtype=self.dtype,
            )
            self._buffers[state_id] = buffer
        return buffer

    def begin_speculative(self, state_ids: Sequence[Hashable]) -> None:
        """Snapshot each slot once before draft-token updates."""
        for state_id in state_ids:
            self._speculative.setdefault(state_id, self.get(state_id).snapshot())

    def accept_speculative(self, state_ids: Sequence[Hashable]) -> None:
        """Commit accepted draft-token updates by dropping their snapshots."""
        for state_id in state_ids:
            self._speculative.pop(state_id, None)

    def reject_speculative(self, state_ids: Sequence[Hashable]) -> None:
        """Restore slots whose draft tokens were rejected."""
        for state_id in state_ids:
            snapshot = self._speculative.pop(state_id, None)
            if snapshot is not None:
                self.get(state_id).rollback(snapshot)

    def release(self, state_ids: Sequence[Hashable]) -> None:
        """Drop state when vLLM releases its corresponding cache slots."""
        for state_id in state_ids:
            self._buffers.pop(state_id, None)
            self._speculative.pop(state_id, None)

    def export_slots(
        self, state_ids: Sequence[Hashable], *, device: torch.device | str = "cpu"
    ) -> dict[Hashable, KVBufferSnapshot]:
        """Export selected vLLM state slots for an offload connector."""
        return {
            state_id: self.get(state_id).export_state(device=device)
            for state_id in state_ids
        }

    def restore_slots(self, snapshots: dict[Hashable, KVBufferSnapshot]) -> None:
        """Restore slots fetched by an offload connector before compute."""
        for state_id, snapshot in snapshots.items():
            self.get(state_id).restore_state(snapshot)
