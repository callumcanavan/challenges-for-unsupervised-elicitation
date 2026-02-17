"""Prompting methods: zero-shot, few-shot, bootstrap, and golden few-shot.

Each method scores test items using vLLM logprob differences.
"""

from __future__ import annotations

import logging
import random as rng_module
from dataclasses import dataclass, field

import numpy as np

from src.config import DEFAULT_BOOTSTRAP_POOL_SIZE, PromptFormat
from src.data import sample_few_shot_examples

logger = logging.getLogger(__name__)

# Number of random example prompts to log per scoring call
_NUM_EXAMPLE_PROMPTS = 2

# Conservative character limit for prompts. Llama-3.1's context is 131072 tokens;
# ctrl-z code averages ~3.7 chars/token, so 131072 * 3.0 ≈ 393K chars gives
# ample headroom. Prompts exceeding this are resampled with new few-shot examples.
_MAX_PROMPT_CHARS = 390_000
_MAX_RESAMPLE_ATTEMPTS = 10


@dataclass
class PromptingResult:
    """Result from a prompting method."""

    scores: np.ndarray
    """Per-item scores (true_logprob - false_logprob)."""

    true_logprobs: np.ndarray
    """Per-item true-token log-probabilities."""

    false_logprobs: np.ndarray
    """Per-item false-token log-probabilities."""

    labels: np.ndarray
    """Predicted labels (score > 0)."""

    iteration_metrics: list[dict] | None = None
    """Per-iteration metrics for bootstrap methods."""


def _log_example_prompts(
    prompts: list[str],
    prompt_format: PromptFormat,
    n: int = _NUM_EXAMPLE_PROMPTS,
) -> None:
    """Log a few randomly selected example prompts for debugging."""
    if not prompts:
        return
    indices = rng_module.sample(range(len(prompts)), min(n, len(prompts)))
    for idx in indices:
        logger.info(
            "=== Example prompt (index %d of %d) ===\n%s\n[next token: %s / %s]\n"
            "=== End example ===",
            idx, len(prompts), prompts[idx],
            prompt_format.true_token, prompt_format.false_token,
        )


def _score_prompts(
    model_name: str,
    prompts: list[str],
    prompt_format: PromptFormat,
    debug: bool = False,
    **model_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a list of prompts via vLLM (or return random scores in debug mode)."""
    _log_example_prompts(prompts, prompt_format)
    if debug:
        n = len(prompts)
        return np.random.randn(n), np.random.randn(n), np.random.randn(n)
    from src.vllm_inference import compute_scores_vllm
    return compute_scores_vllm(
        model_name, prompts,
        prompt_format.true_token, prompt_format.false_token,
        **model_kwargs,
    )


# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------


def run_zero_shot(
    items: list[dict],
    prompt_format: PromptFormat,
    model_name: str,
    debug: bool = False,
    **model_kwargs,
) -> PromptingResult:
    """Score items with zero-shot prompting (no few-shot examples).

    Each item's prompt is scored directly for true/false token logprobs.
    """
    # Build prompts (with prefix if any, but no few-shot examples)
    prompts = []
    for item in items:
        p = prompt_format.format_prompt(item)
        if prompt_format.prefix:
            p = prompt_format.prefix + p
        prompts.append(p)

    scores, true_lps, false_lps = _score_prompts(
        model_name, prompts, prompt_format, debug=debug, **model_kwargs
    )

    return PromptingResult(
        scores=scores,
        true_logprobs=true_lps,
        false_logprobs=false_lps,
        labels=scores > 0,
    )


# ---------------------------------------------------------------------------
# Few-shot (random or golden labels)
# ---------------------------------------------------------------------------


def run_few_shot(
    items: list[dict],
    train_pool: list[dict],
    n_shots: int,
    prompt_format: PromptFormat,
    model_name: str,
    labels_source: str = "random",
    seed: int = 0,
    debug: bool = False,
    **model_kwargs,
) -> PromptingResult:
    """Score items using few-shot prompting.

    For ``labels_source="random"``, labels are assigned to pool items
    **before** selection to avoid ground-truth leakage.

    Args:
        items: Test items to score.
        train_pool: Pool of labeled items for few-shot examples.
        n_shots: Number of few-shot examples per prompt.
        prompt_format: Prompt format config.
        model_name: Model identifier.
        labels_source: ``"random"`` for balanced random labels,
            ``"golden"`` for ground-truth labels.
        seed: Random seed.

    Returns:
        PromptingResult with scores and predictions.
    """

    if labels_source == "random":
        # Assign random labels to pool BEFORE selection (avoids GT leakage)
        pool = [dict(item) for item in train_pool]  # shallow copy
        n_true_pool = len(pool) // 2
        random_pool_labels = [True] * n_true_pool + [False] * (len(pool) - n_true_pool)
        rng_module.shuffle(random_pool_labels)
        for item, lbl in zip(pool, random_pool_labels):
            item["predicted_label"] = lbl
        label_key = "predicted_label"
    elif labels_source == "golden":
        pool = train_pool
        label_key = "label"
    else:
        raise ValueError(f"Unknown labels_source: {labels_source!r}")

    prompts: list[str] = []
    for i, item in enumerate(items):
        for _attempt in range(_MAX_RESAMPLE_ATTEMPTS):
            examples, ex_labels = sample_few_shot_examples(
                pool, n_shots, label_key=label_key,
                exclude_item=item,
            )

            if labels_source == "golden":
                few_shot_labels: list[bool | None] = [
                    _get_golden_label(ex) for ex in examples
                ]
            else:
                few_shot_labels = ex_labels

            prompt = prompt_format.format_few_shot_prompt(examples, few_shot_labels, item)
            if len(prompt) <= _MAX_PROMPT_CHARS:
                break
            logger.warning(
                "Prompt %d too long (%d chars), resampling (attempt %d/%d)",
                i, len(prompt), _attempt + 1, _MAX_RESAMPLE_ATTEMPTS,
            )
        prompts.append(prompt)

    # Log cross-class borrowing if it occurred
    n_borrowed = sum(
        1 for item in pool if item.get("_cross_class_borrowed", False)
    )
    if n_borrowed > 0:
        logger.warning("Cross-class borrowing occurred: %d pool items borrowed", n_borrowed)

    scores, true_lps, false_lps = _score_prompts(
        model_name, prompts, prompt_format, debug=debug, **model_kwargs
    )

    return PromptingResult(
        scores=scores,
        true_logprobs=true_lps,
        false_logprobs=false_lps,
        labels=scores > 0,
    )


def _get_golden_label(item: dict) -> bool | None:
    """Get the golden label for a few-shot example.

    For items from normative datasets (with ``dataset_source == "political_normative"``),
    returns None (indeterminate) if the prompt format supports it.
    """
    if item.get("dataset_source") == "political_normative":
        return None
    return item["label"]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def run_bootstrap(
    items: list[dict],
    train_pool: list[dict],
    shot_sequence: list[int],
    prompt_format: PromptFormat,
    model_name: str,
    bootstrap_pool_size: int = DEFAULT_BOOTSTRAP_POOL_SIZE,
    seed: int = 0,
    debug: bool = False,
    **model_kwargs,
) -> PromptingResult:
    """Score items using bootstrapped few-shot prompting.

    Iteratively refines labels on a training pool subset. At each iteration,
    a pool of ``bootstrap_pool_size`` items is labeled using the previous
    iteration's predictions, then used as few-shot examples for the next
    iteration.

    Args:
        items: Test items to score.
        train_pool: Full training pool.
        shot_sequence: List of shot counts per iteration. The last value is
            used for final test scoring.
        prompt_format: Prompt format config.
        model_name: Model identifier.
        bootstrap_pool_size: Number of items to label per iteration.
        seed: Random seed.

    Returns:
        PromptingResult with scores and per-iteration metrics.
    """
    # Initialize: sample the pool subset
    pool = list(train_pool)
    rng_module.shuffle(pool)
    pool = pool[:bootstrap_pool_size]

    # Assign initial random balanced labels
    n_true = len(pool) // 2
    initial_labels = [True] * n_true + [False] * (len(pool) - n_true)
    rng_module.shuffle(initial_labels)
    for item, label in zip(pool, initial_labels):
        item["predicted_label"] = label

    iteration_metrics: list[dict] = []

    # Iterate through shot sequence (all but last are pool-labeling iterations)
    for iter_idx, n_shots in enumerate(shot_sequence[:-1]):
        logger.info("Bootstrap iteration %d: %d shots", iter_idx, n_shots)

        # Build prompts for pool items using current labels
        prompts: list[str] = []
        for i, item in enumerate(pool):
            for _attempt in range(_MAX_RESAMPLE_ATTEMPTS):
                examples, _ = sample_few_shot_examples(
                    pool, n_shots, label_key="predicted_label",
                    exclude_item=item,
                )
                ex_labels: list[bool | None] = [ex["predicted_label"] for ex in examples]
                prompt = prompt_format.format_few_shot_prompt(examples, ex_labels, item)
                if len(prompt) <= _MAX_PROMPT_CHARS:
                    break
                logger.warning(
                    "Bootstrap pool prompt %d too long (%d chars), resampling (attempt %d/%d)",
                    i, len(prompt), _attempt + 1, _MAX_RESAMPLE_ATTEMPTS,
                )
            prompts.append(prompt)

        # Score pool
        pool_scores, _, _ = _score_prompts(
            model_name, prompts, prompt_format, debug=debug, **model_kwargs
        )

        # Update pool labels
        for item, score in zip(pool, pool_scores):
            item["predicted_label"] = bool(score > 0)

        # Record metrics
        gt_labels = np.array([item["label"] for item in pool])
        pred_labels = np.array([item["predicted_label"] for item in pool])
        accuracy = float(np.mean(pred_labels == gt_labels))
        n_pred_true = int(pred_labels.sum())
        iteration_metrics.append({
            "iteration": iter_idx,
            "n_shots": n_shots,
            "pool_size": len(pool),
            "accuracy": accuracy,
            "num_pred_true": n_pred_true,
            "num_gt_true": int(gt_labels.sum()),
            "cross_class_borrowed": sum(
                1 for item in pool if item.get("_cross_class_borrowed", False)
            ),
        })
        logger.info("  Pool accuracy: %.4f, pred true: %d/%d", accuracy, n_pred_true, len(pool))

    # Final scoring on test items
    final_shots = shot_sequence[-1]
    logger.info("Final test scoring with %d shots", final_shots)

    test_prompts: list[str] = []
    for i, item in enumerate(items):
        for _attempt in range(_MAX_RESAMPLE_ATTEMPTS):
            examples, _ = sample_few_shot_examples(
                pool, final_shots, label_key="predicted_label",
                exclude_item=item,
            )
            ex_labels_final: list[bool | None] = [ex["predicted_label"] for ex in examples]
            prompt = prompt_format.format_few_shot_prompt(examples, ex_labels_final, item)
            if len(prompt) <= _MAX_PROMPT_CHARS:
                break
            logger.warning(
                "Bootstrap test prompt %d too long (%d chars), resampling (attempt %d/%d)",
                i, len(prompt), _attempt + 1, _MAX_RESAMPLE_ATTEMPTS,
            )
        test_prompts.append(prompt)

    scores, true_lps, false_lps = _score_prompts(
        model_name, test_prompts, prompt_format, debug=debug, **model_kwargs
    )

    return PromptingResult(
        scores=scores,
        true_logprobs=true_lps,
        false_logprobs=false_lps,
        labels=scores > 0,
        iteration_metrics=iteration_metrics,
    )
