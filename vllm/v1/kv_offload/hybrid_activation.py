# SPDX-License-Identifier: Apache-2.0
"""Activation checkpoints used by hybrid prefix recovery.

KV tensors alone are not sufficient to skip a decoder layer during a replay:
the output hidden state and residual of that layer are also required by the
next layer.  This module provides a deliberately small, process-local CPU
checkpoint store for the experimental hybrid recovery path.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch

from vllm.utils.platform_utils import is_pin_memory_available


def activation_key(prompt_token_ids: list[int] | None) -> str | None:
    if prompt_token_ids is None:
        return None
    payload = bytearray()
    for token_id in prompt_token_ids:
        payload.extend(int(token_id).to_bytes(8, "little", signed=True))
    return hashlib.sha256(payload).hexdigest()


@dataclass
class _LayerCheckpoint:
    hidden_states: torch.Tensor
    residual: torch.Tensor
    valid_tokens: torch.Tensor

    @property
    def nbytes(self) -> int:
        return (
            self.hidden_states.nbytes
            + self.residual.nbytes
            + self.valid_tokens.nbytes
        )


class LMCacheActivationBackend:
    """Adapter for LMCache's chunked hidden-state store.

    LMCache stores complete chunks, while vLLM may execute a long prefix in
    several scheduler slices.  We therefore assemble one layer boundary in a
    temporary CPU tensor and publish it only after every token in the prefix
    has been observed.  A partial/evicted boundary is simply a miss and the
    model falls back to replay.
    """

    def __init__(self, recovery_store: Any) -> None:
        self.recovery_store = recovery_store
        self._token_ids: dict[str, list[int]] = {}
        self._pending: dict[
            tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        # Hidden-state recovery is a separate H2D path from the KV offloader.
        # Keep its staging buffers pinned as well; otherwise ``Tensor.to``
        # silently falls back to a synchronous pageable-memory copy and the
        # KV-transfer metrics do not account for that cost.
        self._pin_memory = is_pin_memory_available() and torch.cuda.is_available()

    def register_tokens(self, key: str, token_ids: list[int]) -> None:
        self._token_ids[key] = list(token_ids)

    def capture(
        self,
        key: str,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        start: int,
        count: int,
        total_tokens: int,
    ) -> None:
        token_ids = self._token_ids.get(key)
        if token_ids is None or len(token_ids) != total_tokens:
            return
        if start < 0 or count <= 0 or start + count > total_tokens:
            return
        pending_key = (key, layer_idx)
        pending = self._pending.get(pending_key)
        if pending is None or pending[0].shape[0] != total_tokens:
            hidden = torch.empty(
                (total_tokens, *hidden_states.shape[1:]),
                device="cpu",
                dtype=hidden_states.dtype,
                pin_memory=self._pin_memory,
            )
            residual_cpu = torch.empty_like(hidden, pin_memory=self._pin_memory)
            valid = torch.zeros(
                total_tokens, dtype=torch.bool, pin_memory=self._pin_memory
            )
            pending = (hidden, residual_cpu, valid)
            self._pending[pending_key] = pending
        hidden_cpu, residual_cpu, valid = pending
        hidden_cpu[start : start + count].copy_(
            hidden_states[:count].detach().to(device="cpu")
        )
        residual_cpu[start : start + count].copy_(
            residual[:count].detach().to(device="cpu")
        )
        valid[start : start + count] = True
        if bool(valid.all()):
            self.recovery_store.store_boundary(
                token_ids,
                hidden_cpu,
                residual_cpu,
                layer_idx=layer_idx,
            )
            self._pending.pop(pending_key, None)

    def restore(
        self,
        key: str,
        layer_idx: int,
        start: int,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        token_ids = self._token_ids.get(key)
        if token_ids is None:
            return None
        boundary = self.recovery_store.retrieve_boundary(token_ids, layer_idx=layer_idx)
        if (
            boundary is None
            or start < 0
            or start + count > boundary.hidden_states.shape[0]
        ):
            return None
        hidden = boundary.hidden_states[start : start + count].to(
            device=device, dtype=dtype, non_blocking=boundary.hidden_states.is_pinned()
        )
        residual = boundary.residual[start : start + count].to(
            device=device, dtype=dtype, non_blocking=boundary.residual.is_pinned()
        )
        return hidden, residual


class HybridActivationStore:
    """Bounded CPU store for per-prefix layer-boundary activations."""

    def __init__(self, max_bytes: int = 0) -> None:
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, dict[int, _LayerCheckpoint]] = (
            OrderedDict()
        )
        self._bytes = 0
        self._backend: LMCacheActivationBackend | None = None
        self._pin_memory = is_pin_memory_available() and torch.cuda.is_available()

    def register_tokens(self, key: str, token_ids: list[int]) -> None:
        """Associate a vLLM activation key with its full token prefix."""
        if self._backend is not None:
            self._backend.register_tokens(key, token_ids)

    def attach_backend(self, backend: LMCacheActivationBackend) -> None:
        """Use an LMCache hidden-state store instead of the local fallback."""
        self._backend = backend

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def _evict(self) -> None:
        while self.max_bytes > 0 and self._bytes > self.max_bytes and self._entries:
            key, layers = self._entries.popitem(last=False)
            self._bytes -= sum(item.nbytes for item in layers.values())

    def has(self, key: str, layer_idx: int) -> bool:
        layers = self._entries.get(key)
        return layers is not None and layer_idx in layers

    def capture(
        self,
        key: str,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        start: int,
        count: int,
        total_tokens: int,
    ) -> None:
        if self._backend is not None:
            self._backend.capture(
                key,
                layer_idx,
                hidden_states,
                residual,
                start,
                count,
                total_tokens,
            )
            return
        """Capture a token slice at a decoder-layer boundary."""
        if residual is None:
            return
        end = start + count
        if start < 0 or count <= 0 or end > total_tokens:
            return
        layers = self._entries.setdefault(key, {})
        old = layers.get(layer_idx)
        if old is None or old.hidden_states.shape[0] != total_tokens:
            if old is not None:
                self._bytes -= old.nbytes
            hidden_cpu = torch.empty(
                (total_tokens, *hidden_states.shape[1:]),
                device="cpu",
                dtype=hidden_states.dtype,
                pin_memory=self._pin_memory,
            )
            residual_cpu = torch.empty_like(
                hidden_cpu, pin_memory=self._pin_memory
            )
            valid_tokens = torch.zeros(
                total_tokens, dtype=torch.bool, pin_memory=self._pin_memory
            )
            old = _LayerCheckpoint(hidden_cpu, residual_cpu, valid_tokens)
            layers[layer_idx] = old
            self._bytes += old.nbytes
        old.hidden_states[start:end].copy_(
            hidden_states.detach()[:count].to(
                device="cpu", dtype=hidden_states.dtype
            )
        )
        old.residual[start:end].copy_(
            residual.detach()[:count].to(device="cpu", dtype=residual.dtype)
        )
        old.valid_tokens[start:end] = True
        self._entries.move_to_end(key)
        self._evict()

    def restore(
        self,
        key: str,
        layer_idx: int,
        start: int,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._backend is not None:
            return self._backend.restore(
                key, layer_idx, start, count, device, dtype
            )
        layers = self._entries.get(key)
        checkpoint = layers.get(layer_idx) if layers is not None else None
        if checkpoint is None:
            return None
        end = start + count
        if (
            start < 0
            or end > checkpoint.hidden_states.shape[0]
            or not bool(checkpoint.valid_tokens[start:end].all())
        ):
            return None
        self._entries.move_to_end(key)
        hidden = checkpoint.hidden_states[start:end].to(
            device=device,
            dtype=dtype,
            non_blocking=checkpoint.hidden_states.is_pinned(),
        )
        residual = checkpoint.residual[start:end].to(
            device=device,
            dtype=dtype,
            non_blocking=checkpoint.residual.is_pinned(),
        )
        return hidden, residual


@dataclass(frozen=True)
class ActivationSegment:
    key: str
    prefix_offset: int
    token_count: int
    prompt_length: int
    flat_offset: int
    active: bool


class HybridActivationContext:
    """Per-forward capture/restore plan consumed by hybrid model layers."""

    def __init__(
        self,
        store: HybridActivationStore,
        segments: list[ActivationSegment],
        *,
        expected_tokens: int,
        capture_layer_type: str | None = None,
        restore_layer_type: str | None = None,
    ) -> None:
        self.store = store
        self.segments = segments
        self.expected_tokens = expected_tokens
        self.capture_layer_type = capture_layer_type
        self.restore_layer_type = restore_layer_type

    def capture(
        self,
        layer_idx: int,
        layer_type: str,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        if layer_type != self.capture_layer_type:
            return
        for segment in self.segments:
            self.store.capture(
                segment.key,
                layer_idx,
                hidden_states[segment.flat_offset : segment.flat_offset + segment.token_count],
                residual[segment.flat_offset : segment.flat_offset + segment.token_count],
                start=segment.prefix_offset,
                count=segment.token_count,
                total_tokens=segment.prompt_length,
            )

    def restore(
        self,
        layer_idx: int,
        layer_type: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if layer_type != self.restore_layer_type or not self.segments:
            return None
        # A mixed batch cannot safely skip a layer without a separate masked
        # execution kernel.  Fall back to the normal forward in that case.
        if not all(segment.active for segment in self.segments):
            return None
        if sum(segment.token_count for segment in self.segments) != self.expected_tokens:
            return None
        hidden_parts: list[torch.Tensor] = []
        residual_parts: list[torch.Tensor] = []
        for segment in self.segments:
            restored = self.store.restore(
                segment.key,
                layer_idx,
                segment.prefix_offset,
                segment.token_count,
                device,
                dtype,
            )
            if restored is None:
                return None
            hidden, residual = restored
            hidden_parts.append(hidden)
            residual_parts.append(residual)
        return torch.cat(hidden_parts, dim=0), torch.cat(residual_parts, dim=0)
