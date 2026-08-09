# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec


def _spec() -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
    )


def test_mamba_sparse_retention_keeps_segment_and_replay_boundaries():
    mask = MambaManager.reachable_block_mask(
        start_block=0,
        end_block=10,
        alignment_tokens=16,
        kv_cache_spec=_spec(),
        use_eagle=False,
        retention_interval=64,
        num_prompt_tokens=144,
    )
    # Ends at 64, 128 and the latest full prompt boundary 144.
    assert mask == [False, False, False, True, False, False, False, True, True, False]


def test_mamba_dense_retention_preserves_default_behavior():
    assert MambaManager.reachable_block_mask(
        0, 2, 16, _spec(), False, retention_interval=None
    ) is None


def test_mamba_lookup_resumes_from_deepest_sparse_checkpoint():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(2)
    hashes = [BlockHash(bytes([index]) * 32) for index in range(5)]
    for block, index in zip(blocks, (1, 3)):
        key = make_block_hash_with_group_id(hashes[index], 0)
        block.block_hash = key
        pool.cached_block_hash_to_block.insert(key, block)

    (hit_blocks,) = MambaManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=80,
        kv_cache_group_ids=[0],
        block_pool=pool,
        kv_cache_spec=_spec(),
        drop_eagle_block=False,
        alignment_tokens=16,
    )
    assert len(hit_blocks) == 4
    assert hit_blocks[:3] == [pool.null_block] * 3
    assert hit_blocks[3] is blocks[1]
