"""Synthetic sparse-prefix checkpoint placement benchmark."""

from vllm.v1.kv_offload.sparse_prefix import SparsePrefixConfig, SparsePrefixPlanner


def main() -> None:
    planner = SparsePrefixPlanner(
        SparsePrefixConfig(checkpoint_interval=256, state_bytes=2 * 1024**2,
                           memory_budget_bytes=16 * 1024**2)
    )
    plan = planner.select_checkpoints(
        4096, [512, 1024, 2048, 3072, 4096], [0.1, 0.2, 0.3, 0.25, 0.15]
    )
    print(plan)
    for request_tokens in (700, 2300, 4096):
        print(request_tokens, planner.resume(plan.positions, request_tokens))


if __name__ == "__main__":
    main()
