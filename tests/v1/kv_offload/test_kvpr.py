# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.kv_offload.kvpr import KVPRConfig, KVPRPlanner


def test_kvpr_plan_has_partial_recompute_for_pcie_bound_transfer():
    planner = KVPRPlanner(
        KVPRConfig(
            cpu_gpu_bandwidth_bytes_per_second=16e9,
            gpu_compute_flops_per_second=312e12,
            token_step=8,
        )
    )

    plan = planner.plan(
        prompt_tokens=4096,
        generation_tokens=256,
        batch_size=8,
        hidden_size=4096,
    )

    assert plan.is_partial
    assert 0 < plan.recompute_tokens < 4096
    assert plan.transfer_tokens + plan.recompute_tokens == plan.total_tokens


def test_kvpr_estimator_prefers_full_prompt_recompute_when_compute_is_cheap():
    planner = KVPRPlanner(
        KVPRConfig(
            cpu_gpu_bandwidth_bytes_per_second=1e9,
            gpu_compute_flops_per_second=1e18,
        )
    )

    plan = planner.plan(128, 1, batch_size=1, hidden_size=128)

    assert plan.recompute_tokens == 128


def test_kvpr_estimator_prefers_recompute_when_pcie_is_slow():
    planner = KVPRPlanner(
        KVPRConfig(
            cpu_gpu_bandwidth_bytes_per_second=1e6,
            gpu_compute_flops_per_second=1e12,
        )
    )

    plan = planner.plan(128, 1, batch_size=1, hidden_size=128)

    assert plan.recompute_tokens > 0


def test_kvpr_timeline_matches_critical_path_model():
    planner = KVPRPlanner(
        KVPRConfig(cpu_gpu_bandwidth_bytes_per_second=16e9,
                   gpu_compute_flops_per_second=312e12)
    )
    plan = planner.plan(1024, 128, batch_size=4, hidden_size=2048)
    stages = plan.timeline(planner, batch_size=4, hidden_size=2048)
    assert stages[0][0] == "activation_transfer"
    assert stages[1][1] == stages[2][1] == stages[0][2]
    critical_path = max(end for _, _, end in stages)
    assert abs(critical_path - plan.estimated_seconds) < 1e-12
