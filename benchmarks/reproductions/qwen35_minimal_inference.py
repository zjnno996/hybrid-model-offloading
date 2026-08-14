# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a minimal native-vLLM Qwen3.5 inference smoke test.

This is intentionally a small, eager-mode test. It checks that the actual
Hybrid Attention model, including Qwen Gated DeltaNet layers, loads and emits
tokens before using it in a larger Mooncake experiment.
"""

import argparse
import os
import sys
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/models/Qwen3.5-27B")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="请只回答一个词：正常")
    parser.add_argument(
        "--log-file",
        type=Path,
        help="write parent and spawned EngineCore stdout/stderr to this file",
    )
    args = parser.parse_args()

    if args.log_file is not None:
        log_file = args.log_file.open("w", buffering=1)
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    result = llm.generate(
        [args.prompt], SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    )
    print("PROMPT_TOKENS", len(result[0].prompt_token_ids))
    print("OUTPUT", repr(result[0].outputs[0].text))


if __name__ == "__main__":
    main()
