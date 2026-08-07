# SPDX-License-Identifier: Apache-2.0
"""IO-aware linear-attention K/V buffer from KVBuffer (2026).

The committed state is updated in chunks while a small recent K/V buffer stays
materialized.  Queries can attend to the buffer directly, avoiding an HBM
state update for every token.  Snapshots provide speculative verification
rollback without allocating a temporary full state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVBufferConfig:
    buffer_tokens: int = 64
    update_chunk_tokens: int = 64


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
