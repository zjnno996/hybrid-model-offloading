"""Synthetic HYPIC segment-composition benchmark."""

import torch

from vllm.v1.kv_offload.hypic_state import HypicPlanner, HypicTransition


def main() -> None:
    planner = HypicPlanner(segment_tokens=128)
    transitions = tuple(
        HypicTransition(torch.eye(4) * (1.0 + i / 100), torch.ones(4) * i)
        for i in range(8)
    )
    combined = planner.compose_segments(transitions)
    assert combined is not None
    print("segments=8", "state_norm=", float(combined.apply(torch.zeros(4)).norm()))
    for prefix_tokens in (256, 300, 900):
        print(planner.recovery_plan(prefix_tokens))


if __name__ == "__main__":
    main()
