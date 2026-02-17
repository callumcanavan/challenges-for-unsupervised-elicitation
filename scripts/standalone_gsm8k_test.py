#!/usr/bin/env python3
"""Standalone script: test how well Llama-3.1-70B classifies GSM8K claims.

Zero-shot binary classification: given a math question and a chain-of-thought
claim, predict True (correct) or False (incorrect) via logprob comparison.

Usage:
    python scripts/standalone_gsm8k_test.py
    python scripts/standalone_gsm8k_test.py --model meta-llama/Llama-3.1-8B
    python scripts/standalone_gsm8k_test.py --max-items 200
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-70B")
    p.add_argument("--data", default="datasets/test_gsm8k.json")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--top-logprobs", type=int, default=10000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output", default=None, help="Output JSON path")
    p.add_argument("--tp", type=int, default=None, help="Tensor parallel size")
    return p.parse_args()


TEMPLATE = "Question: {question}\nClaim: {cot}\nI think this claim is"
TRUE_ID = "true"
FALSE_ID = "false"


def build_prompts(items):
    return [TEMPLATE.format(**item) for item in items]


def aggregate_label_probs(logprob_dict, true_id="true", false_id="false"):
    """Sum probabilities across all tokens matching true/false identifiers."""
    eps = 1e-5
    true_prob, false_prob = eps, eps
    for logprob_obj in logprob_dict.values():
        decoded = getattr(logprob_obj, "decoded_token", None)
        if decoded is None:
            continue
        parts = re.split(r"[^a-z]", decoded.lower())
        has_true = true_id in parts
        has_false = false_id in parts
        if has_true != has_false:
            prob = math.exp(logprob_obj.logprob)
            if has_true:
                true_prob += prob
            else:
                false_prob += prob
    return math.log(true_prob), math.log(false_prob)


def main():
    args = parse_args()

    # Load data
    data = json.load(open(args.data))
    if args.max_items:
        data = data[: args.max_items]

    labels = np.array([bool(item["label"]) if isinstance(item["label"], bool)
                       else str(item["label"]).lower() == "true"
                       for item in data])
    prompts = build_prompts(data)

    print(f"Model: {args.model}")
    print(f"Items: {len(data)} (true={labels.sum()}, false={(~labels).sum()})")
    print(f"Temperature: {args.temperature}")
    print(f"Top logprobs: {args.top_logprobs}")
    print(f"\nExample prompt:\n{prompts[0]}\n")

    # Load vLLM
    import torch
    from vllm import LLM, SamplingParams

    tp = args.tp or torch.cuda.device_count() or 1
    print(f"Loading model with tp={tp}...")
    model = LLM(
        model=args.model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.9,
        max_logprobs=args.top_logprobs,
    )

    params = SamplingParams(
        max_tokens=1,
        temperature=args.temperature,
        logprobs=args.top_logprobs,
    )

    # Score
    print("Scoring...")
    t0 = time.time()

    true_lps = np.zeros(len(prompts))
    false_lps = np.zeros(len(prompts))
    batch_size = 500

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        outputs = model.generate(chunk, params)
        for j, output in enumerate(outputs):
            token_logprobs = output.outputs[0].logprobs
            if token_logprobs and len(token_logprobs) > 0:
                t_lp, f_lp = aggregate_label_probs(token_logprobs[0])
                true_lps[i + j] = t_lp
                false_lps[i + j] = f_lp
        print(f"  Scored {min(i + batch_size, len(prompts))}/{len(prompts)}")

    elapsed = time.time() - t0
    scores = true_lps - false_lps
    preds = scores > 0

    # Metrics
    from sklearn.metrics import roc_auc_score

    acc = float(np.mean(preds == labels))
    auroc = float(roc_auc_score(labels.astype(int), scores))

    # Per-class breakdown
    true_acc = float(np.mean(preds[labels] == labels[labels]))
    false_acc = float(np.mean(preds[~labels] == labels[~labels]))

    print(f"\n{'='*50}")
    print(f"Results ({elapsed:.1f}s)")
    print(f"{'='*50}")
    print(f"AUROC:     {auroc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"  True  items acc: {true_acc:.4f} ({labels.sum()} items)")
    print(f"  False items acc: {false_acc:.4f} ({(~labels).sum()} items)")
    print(f"\nScore stats:")
    print(f"  True  items: mean={scores[labels].mean():.3f}, std={scores[labels].std():.3f}")
    print(f"  False items: mean={scores[~labels].mean():.3f}, std={scores[~labels].std():.3f}")

    # Show worst misclassifications
    print(f"\nTop 5 False items misclassified as True (highest scores):")
    false_indices = np.where(~labels)[0]
    false_scores = scores[false_indices]
    worst = false_indices[np.argsort(false_scores)[-5:][::-1]]
    for idx in worst:
        item = data[idx]
        print(f"  score={scores[idx]:+.3f} | answer={item.get('answer','?')} | "
              f"ref_answer={item.get('ref_answer','?')} | "
              f"cot={item['cot'][:80]}...")

    # Save
    out_path = args.output or f"results/standalone_gsm8k_{args.model.split('/')[-1]}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "n_items": len(data),
        "auroc": auroc,
        "accuracy": acc,
        "true_acc": true_acc,
        "false_acc": false_acc,
        "temperature": args.temperature,
        "top_logprobs": args.top_logprobs,
        "elapsed_seconds": elapsed,
    }
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
