"""CPU microbenchmark for KVBuffer materialization and deferred updates."""

import argparse
import time

import torch

from vllm.v1.attention.backends.kvbuffer import KVBuffer, KVBufferConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    keys = torch.randn(args.tokens, args.dim)
    values = torch.randn(args.tokens, args.dim)
    queries = torch.randn(32, args.dim)
    buffer = KVBuffer(args.dim, args.dim, KVBufferConfig(buffer_tokens=128,
                                                        update_chunk_tokens=64))
    start = time.perf_counter()
    buffer.append(keys, values)
    output = buffer.materialize(queries)
    elapsed = time.perf_counter() - start
    print(f"tokens={args.tokens} buffered={buffer.keys.shape[0]} "
          f"state_tokens={args.tokens - buffer.keys.shape[0]} "
          f"elapsed={elapsed:.6f}s output_norm={float(output.norm()):.4f}")


if __name__ == "__main__":
    main()
