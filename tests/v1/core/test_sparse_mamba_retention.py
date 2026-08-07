# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
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
