# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Event

from vllm.v1.kv_offload.kvpr import (
    KVPRConfig,
    KVPRExecutionCallbacks,
    KVPRExecutor,
    KVPRPlanner,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
)


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


def test_kvpr_executor_starts_recompute_and_suffix_after_activation():
    plan = KVPRPlanner(
        KVPRConfig(cpu_gpu_bandwidth_bytes_per_second=16e9,
                   gpu_compute_flops_per_second=312e12)
    ).plan(1024, 128, batch_size=4, hidden_size=2048)
    activation_finished = Event()
    suffix_started = Event()

    def load_activations(tokens: int) -> str:
        assert tokens == plan.recompute_tokens
        activation_finished.set()
        return "activation"

    def recompute_prefix(tokens: int) -> str:
        assert activation_finished.is_set()
        assert suffix_started.wait(timeout=1)
        assert tokens == plan.recompute_tokens
        return "recompute"

    def load_kv_suffix(tokens: int) -> str:
        assert activation_finished.is_set()
        suffix_started.set()
        assert tokens == plan.transfer_tokens
        return "suffix"

    result = KVPRExecutor.execute(
        plan,
        KVPRExecutionCallbacks(load_activations, recompute_prefix, load_kv_suffix),
    )
    assert result.activation_result == "activation"
    assert result.recompute_result == "recompute"
    assert result.suffix_result == "suffix"


def test_kvpr_plan_can_be_carried_in_offloading_metadata():
    plan = KVPRPlanner().plan(64, 8, batch_size=1, hidden_size=32)
    metadata = OffloadingConnectorMetadata(load_jobs={}, store_jobs={},
                                           kvpr_plans={"request-1": plan})
    assert metadata.kvpr_plans["request-1"] is plan


def test_kvpr_metadata_is_exposed_to_forward_context(monkeypatch):
    plan = KVPRPlanner().plan(64, 8, batch_size=1, hidden_size=32)
    metadata = OffloadingConnectorMetadata(
        load_jobs={}, store_jobs={}, kvpr_plans={"request-1": plan}
    )
    context = type("Context", (), {"additional_kwargs": {}})()
    monkeypatch.setattr(
        "vllm.v1.worker.kv_connector_model_runner_mixin.get_forward_context",
        lambda: context,
    )
    KVConnectorModelRunnerMixin._bind_experimental_forward_metadata(metadata)
    assert context.additional_kwargs["kvpr_plans"]["request-1"] is plan


def test_kvpr_remote_hit_plan_is_scheduler_safe():
    planner = KVPRPlanner(
        KVPRConfig(
            cpu_gpu_bandwidth_bytes_per_second=32e9,
            gpu_compute_flops_per_second=312e12,
            token_step=4,
        )
    )
    plan = planner.plan_for_remote_hit(hit_tokens=32, batch_size=1, hidden_size=768)
    assert plan.loads_contiguous_prefix
    assert plan.transfer_tokens + plan.recompute_tokens == 32
    assert plan.transfer_tokens % 4 == 0
    stages = plan.timeline(planner, batch_size=1, hidden_size=768)
    assert [stage[0] for stage in stages] == [
        "prefix_kv_transfer",
        "suffix_recompute",
    ]
    assert stages[1][1] == stages[0][2]
    assert abs(
        stages[-1][2]
        - planner.estimate_contiguous_prefix_seconds(
            32, plan.recompute_tokens, batch_size=1, hidden_size=768
        )
    ) < 1e-12
