# SPDX-License-Identifier: Apache-2.0
"""Unit tests for group selection in native CPU/H2D hybrid recovery."""

from types import SimpleNamespace

import torch

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


def _scheduler_with_full_and_linear_groups() -> OffloadingConnectorScheduler:
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.config = SimpleNamespace(kv_group_configs=(object(), object()))
    scheduler._group_specs = (
        object(),
        MambaSpec(
            block_size=16,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
        ),
    )
    return scheduler


def test_native_offload_loads_all_groups_by_default(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    monkeypatch.setenv("VLLM_MOONCAKE_HYBRID_POLICY", "all_load")
    assert scheduler._get_loaded_group_indices() == (0, 1)


def test_native_offload_p4_loads_only_linear_group(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    monkeypatch.setenv(
        "VLLM_MOONCAKE_HYBRID_POLICY", "full_replay_linear_load"
    )
    assert scheduler._get_loaded_group_indices() == (1,)


def test_native_offload_p3_loads_only_full_group(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
        ),
        MambaSpec(
            block_size=16,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
        ),
    )
    monkeypatch.setenv(
        "VLLM_MOONCAKE_HYBRID_POLICY", "full_load_linear_replay"
    )
    assert scheduler._get_loaded_group_indices() == (0,)
