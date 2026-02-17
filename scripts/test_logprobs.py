#!/usr/bin/env python3
"""Quick smoke test: can we extract logprobs without overflow?

Run on a GPU node to verify vLLM logprob extraction works before
submitting a full sweep. Tests multiple batches to increase the
chance of hitting problematic token IDs.

Usage (interactive node):
    python scripts/test_logprobs.py
    python scripts/test_logprobs.py --model meta-llama/Llama-3.1-8B
    python scripts/test_logprobs.py --top-logprobs 100
    python scripts/test_logprobs.py --top-logprobs 10000  # test if 10k works
"""

import argparse
import json
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--top-logprobs", type=int, default=100)
    p.add_argument("--n-prompts", type=int, default=200,
                   help="Number of prompts to test (more = higher chance of catching overflow)")
    p.add_argument("--tp", type=int, default=None)
    args = p.parse_args()

    # Load a sample of real prompts
    data = json.load(open("datasets/test_gsm8k.json"))[:args.n_prompts]
    prompts = [
        f"Question: {item['question']}\nClaim: {item['cot']}\nI think this claim is"
        for item in data
    ]

    import torch
    from vllm import LLM, SamplingParams

    tp = args.tp or torch.cuda.device_count() or 1
    print(f"Model: {args.model}")
    print(f"Top logprobs: {args.top_logprobs}")
    print(f"Prompts: {len(prompts)}")
    print(f"TP: {tp}")
    print(f"Loading model...")

    model = LLM(
        model=args.model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.9,
        max_logprobs=args.top_logprobs,
    )

    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.top_logprobs)

    print(f"Scoring {len(prompts)} prompts in batches of 50...")
    t0 = time.time()
    n_with_true = 0
    n_with_false = 0
    batch_size = 50

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        outputs = model.generate(chunk, params)
        for output in outputs:
            logprob_dict = output.outputs[0].logprobs[0]
            tokens = [
                getattr(lp, "decoded_token", "") or ""
                for lp in logprob_dict.values()
            ]
            token_text = " ".join(tokens)
            if "true" in token_text.lower() or "True" in token_text:
                n_with_true += 1
            if "false" in token_text.lower() or "False" in token_text:
                n_with_false += 1
        print(f"  Batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size} OK")

    elapsed = time.time() - t0
    print(f"\nAll {len(prompts)} prompts scored successfully in {elapsed:.1f}s")
    print(f"Prompts with 'true' variant in top-{args.top_logprobs}: {n_with_true}/{len(prompts)}")
    print(f"Prompts with 'false' variant in top-{args.top_logprobs}: {n_with_false}/{len(prompts)}")
    miss = len(prompts) - min(n_with_true, n_with_false)
    if miss > 0:
        print(f"WARNING: {miss} prompts missing true or false token in top-{args.top_logprobs}")
    else:
        print("All prompts have both true and false tokens covered.")


if __name__ == "__main__":
    main()
