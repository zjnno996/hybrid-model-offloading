# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.marconi_policy import MarconiPolicyConfig


def test_marconi_reorders_real_eviction_queue_by_policy_score():
    pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
        hash_block_size=16,
        marconi_config=MarconiPolicyConfig(efficiency_weight=2.0),
    )
    blocks = pool.get_new_blocks(4)
    assert all(
        pool.register_marconi_entry(
            block,
            state_bytes=100,
            compute_savings=10 if block is blocks[0] else 10_000,
        )
        for block in blocks
    )
    # The low-efficiency entry is recently reused, but the high expected
    # compute savings of the remaining entries still dominates at weight=2.
    pool.touch([blocks[0]])
    pool.free_blocks([blocks[0]])
    pool.free_blocks(blocks)
    eviction_order = pool.free_block_queue.get_all_free_blocks()
    assert eviction_order[0] is blocks[0]


def test_marconi_admission_threshold_can_skip_cache_entry():
    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=True,
        hash_block_size=16,
        marconi_config=MarconiPolicyConfig(min_admission_efficiency=1.0),
    )
    block = pool.get_new_blocks(1)[0]
    assert not pool.register_marconi_entry(
        block, state_bytes=100, compute_savings=10
    )
