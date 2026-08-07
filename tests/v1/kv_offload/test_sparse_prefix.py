# SPDX-License-Identifier: Apache-2.0

from vllm.v1.kv_offload.sparse_prefix import (
    SparsePrefixConfig,
    SparsePrefixPlanner,
)


def test_sparse_prefix_budget_and_weighted_placement():
    planner = SparsePrefixPlanner(
        SparsePrefixConfig(
            checkpoint_interval=100,
            state_bytes=10,
            memory_budget_bytes=30,
        )
    )
    plan = planner.select_checkpoints(
        500, [100, 200, 300, 400, 500], [0.05, 0.1, 0.2, 0.25, 0.4]
    )
    assert plan.positions[0] == 0
    assert len(plan.positions) <= 3
    assert plan.positions[-1] == 400
    assert plan.expected_recompute_tokens < 250


def test_sparse_prefix_resume_uses_deepest_checkpoint():
    plan = SparsePrefixPlanner.resume((0, 128, 256), 220)
    assert plan.checkpoint_tokens == 128
    assert plan.recompute_tokens == 92
