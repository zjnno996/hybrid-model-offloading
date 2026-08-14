# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.kv_offload.mooncake_linear_recovery import (
    RecoveryCosts,
    choose_recovery_path,
    qwen_full_attention_kv_layout,
    qwen_gated_delta_net_state_layout,
)


QWEN35_TEXT_CONFIG = {
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "layer_types": ["linear_attention", "full_attention", "linear_attention"],
}


def test_qwen_gated_delta_net_layout_matches_vllm_shape_calculation():
    layout = qwen_gated_delta_net_state_layout({"text_config": QWEN35_TEXT_CONFIG})

    assert layout.num_linear_layers == 2
    assert layout.num_value_heads == 48
    assert layout.conv_state_elements == 30_720
    assert layout.temporal_state_elements == 786_432
    assert layout.state_bytes_per_layer == 1_634_304
    assert layout.state_bytes_per_request == 3_268_608


def test_qwen_gated_delta_net_layout_honors_tensor_parallel_size():
    layout = qwen_gated_delta_net_state_layout(
        QWEN35_TEXT_CONFIG, tensor_parallel_size=2, dtype_bytes=4
    )

    assert layout.num_value_heads == 24
    assert layout.conv_state_elements == 15_360
    assert layout.temporal_state_elements == 393_216
    assert layout.state_bytes_per_layer == 1_634_304


def test_qwen_gated_delta_net_layout_supports_fp32_temporal_state():
    layout = qwen_gated_delta_net_state_layout(
        QWEN35_TEXT_CONFIG, temporal_dtype_bytes=4
    )

    assert layout.effective_temporal_dtype_bytes == 4
    assert layout.state_bytes_per_layer == 3_207_168


def test_qwen_full_attention_kv_layout_matches_qwen_config():
    layout = qwen_full_attention_kv_layout({"text_config": QWEN35_TEXT_CONFIG})

    assert layout.num_full_attention_layers == 1
    assert layout.kv_bytes_per_token == 4_096
    assert layout.bytes_for_tokens(784) == 3_211_264


def test_recovery_path_chooses_lower_latency():
    transfer = choose_recovery_path(2.0, 5.0)
    recompute = choose_recovery_path(5.0, 2.0)

    assert (transfer.path, transfer.speedup) == ("transfer", 2.5)
    assert (recompute.path, recompute.speedup) == ("recompute", 2.5)


def test_recovery_costs_include_mooncake_queue_and_lookup():
    costs = RecoveryCosts(
        queue_ms=100.0,
        lookup_ms=1.0,
        transfer_ms=25.0,
        materialize_ms=2.0,
        recompute_ms=40.0,
    )

    assert costs.load_ms == 128.0
    assert costs.choose().path == "recompute"


def test_qwen_gated_delta_net_layout_rejects_invalid_config():
    with pytest.raises(ValueError, match="no linear_attention"):
        qwen_gated_delta_net_state_layout(
            {"text_config": {**QWEN35_TEXT_CONFIG, "layer_types": []}}
        )
