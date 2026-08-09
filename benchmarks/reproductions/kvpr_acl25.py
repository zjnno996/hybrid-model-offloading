"""Analytical KVPR split benchmark (no model weights required)."""

import argparse

from vllm.v1.kv_offload.kvpr import KVPRConfig, KVPRPlanner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--generation-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--remote-hit",
        type=int,
        default=None,
        help="Use the executable contiguous-prefix vLLM split for this hit",
    )
    args = parser.parse_args()
    planner = KVPRPlanner(KVPRConfig(token_step=8))
    if args.remote_hit is None:
        plan = planner.plan(
            prompt_tokens=args.prompt_tokens,
            generation_tokens=args.generation_tokens,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
        )
    else:
        plan = planner.plan_for_remote_hit(
            hit_tokens=args.remote_hit,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
        )
    print(plan)
    for name, start, end in plan.timeline(planner, args.batch_size, args.hidden_size):
        print(f"{name}: start={start:.6f}s end={end:.6f}s")


if __name__ == "__main__":
    main()
