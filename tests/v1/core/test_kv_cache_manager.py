# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import _is_mooncake_hybrid_replay_policy

pytestmark = pytest.mark.cpu_test


def test_reset_external_cache_for_replay_preserves_attached_blocks():
    """Mixed replay keeps the async-load allocation attached to the request."""
    request = SimpleNamespace(request_id="req-0")
    loaded_blocks = [object(), object()]
    manager_with_blocks = SimpleNamespace(
        req_to_blocks={request.request_id: loaded_blocks},
        num_cached_block={},
    )
    manager_without_blocks = SimpleNamespace(
        req_to_blocks={request.request_id: []},
        num_cached_block={},
    )
    cache_manager = object.__new__(KVCacheManager)
    cache_manager.coordinator = SimpleNamespace(
        single_type_managers=(manager_with_blocks, manager_without_blocks)
    )

    cache_manager.reset_external_cache_for_replay(request)

    assert manager_with_blocks.num_cached_block == {request.request_id: 2}
    assert manager_without_blocks.num_cached_block == {}


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("all_load", False),
        ("full_load_linear_load", False),
        ("full_load_linear_replay", True),
        ("full_replay_linear_load", True),
    ],
)
def test_mooncake_hybrid_replay_policy_only_matches_remaining_replay_path(
    policy, expected
):
    assert _is_mooncake_hybrid_replay_policy(policy) is expected
