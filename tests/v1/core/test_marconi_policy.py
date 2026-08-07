# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.marconi_policy import MarconiEntry, MarconiPolicy, MarconiPolicyConfig


def test_admission_accounts_for_reuse_and_state_size():
    policy = MarconiPolicy(MarconiPolicyConfig(min_admission_efficiency=2.0))
    expensive_state = MarconiEntry(
        key="ssm",
        state_bytes=100,
        compute_savings=1_000,
        last_access=1,
        reuse_probability=0.5,
    )
    large_kv = MarconiEntry(
        key="kv",
        state_bytes=1_000,
        compute_savings=1_000,
        last_access=1,
        reuse_probability=0.5,
    )

    assert policy.should_admit(expensive_state)
    assert not policy.should_admit(large_kv)


def test_eviction_keeps_recent_compute_efficient_state():
    policy = MarconiPolicy(MarconiPolicyConfig(efficiency_weight=2.0))
    recent_but_cheap = MarconiEntry(
        key="recent",
        state_bytes=100,
        compute_savings=10,
        last_access=10,
        reuse_probability=0.1,
    )
    old_but_expensive = MarconiEntry(
        key="expensive",
        state_bytes=100,
        compute_savings=10_000,
        last_access=1,
        reuse_probability=1.0,
    )

    victims = policy.choose_victims(
        [recent_but_cheap, old_but_expensive], bytes_to_free=100, now=10
    )

    assert [entry.key for entry in victims] == ["recent"]
