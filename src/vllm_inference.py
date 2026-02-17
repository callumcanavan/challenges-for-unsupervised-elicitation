"""vLLM inference backend for prompting experiments.

Scores prompts by generating a single token and aggregating logprobs
across all tokens matching the true/false label identifiers.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_VLLM_MODEL: Any = None
_VLLM_MODEL_NAME: str | None = None

# Maximum prompts per vLLM batch call
_BATCH_CHUNK_SIZE = 500

# Top-K logprobs to request. 100 is sufficient — true/false token variants
# are always in the top 100 (verified by test_logprobs.py). Using larger
# values (e.g. 10k) risks hitting rare high-ID tokens that cause
# OverflowError in vLLM v1's tokenizer.decode().
_NUM_LOGPROBS = 100


def get_vllm_model(
    model_name: str,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    **kwargs,
):
    """Get or create a singleton vLLM model.

    Reuses the same model if called with the same model name.

    Args:
        model_name: HuggingFace model name.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
            Auto-detected if None.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_model_len: Maximum sequence length.

    Returns:
        vLLM LLM instance.
    """
    global _VLLM_MODEL, _VLLM_MODEL_NAME

    if _VLLM_MODEL is not None and _VLLM_MODEL_NAME == model_name:
        return _VLLM_MODEL

    from vllm import LLM

    if tensor_parallel_size is None:
        import torch
        tensor_parallel_size = torch.cuda.device_count() or 1

    vllm_kwargs: dict = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_logprobs": _NUM_LOGPROBS,
    }
    # enable_chunked_prefill improves throughput but was added in vLLM ≥0.6
    import inspect
    from vllm import EngineArgs
    if "enable_chunked_prefill" in inspect.signature(EngineArgs).parameters:
        vllm_kwargs["enable_chunked_prefill"] = True
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len
    vllm_kwargs.update(kwargs)

    logger.info("Loading vLLM model: %s (tp=%d)", model_name, tensor_parallel_size)
    _VLLM_MODEL = LLM(**vllm_kwargs)
    _VLLM_MODEL_NAME = model_name
    return _VLLM_MODEL


def cleanup_vllm() -> None:
    """Free vLLM model and GPU memory."""
    global _VLLM_MODEL, _VLLM_MODEL_NAME
    if _VLLM_MODEL is not None:
        del _VLLM_MODEL
        _VLLM_MODEL = None
        _VLLM_MODEL_NAME = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _aggregate_label_probs(
    logprob_dict: dict,
    true_identifier: str,
    false_identifier: str,
) -> tuple[float, float]:
    """Aggregate probabilities across all tokens matching true/false identifiers.

    Iterates over returned logprobs, splits each decoded token into
    alphabetic parts, and sums probabilities for tokens containing
    the true or false identifier. Returns log of summed probabilities.

    Args:
        logprob_dict: Mapping of token_id -> Logprob from vLLM output.
        true_identifier: Lowercase alphabetic identifier for true (e.g. "true").
        false_identifier: Lowercase alphabetic identifier for false (e.g. "false").

    Returns:
        (log_true_prob, log_false_prob) with eps=1e-5 base.
    """
    eps = 1e-5
    true_prob = eps
    false_prob = eps

    for logprob_obj in logprob_dict.values():
        decoded = getattr(logprob_obj, "decoded_token", None)
        if decoded is None:
            continue
        parts = re.split(r"[^a-z]", decoded.lower())
        has_true = true_identifier in parts
        has_false = false_identifier in parts
        if has_true != has_false:
            prob = math.exp(logprob_obj.logprob)
            if has_true:
                true_prob += prob
            else:
                false_prob += prob

    return math.log(true_prob), math.log(false_prob)


def compute_scores_vllm(
    model_name: str,
    prompts: list[str],
    true_token: str,
    false_token: str,
    **model_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score prompts by comparing aggregated P(true) vs P(false).

    Generates a single token per prompt and aggregates probabilities
    across all returned tokens that contain the true/false identifier
    (e.g. "True", " true", "TRUE" all contribute to the true bucket).
    Score = log(sum_true_probs) - log(sum_false_probs).

    Args:
        model_name: HuggingFace model name.
        prompts: Formatted prompt strings (no label token appended).
        true_token: True label token string (e.g. ``" True"``).
        false_token: False label token string (e.g. ``" False"``).

    Returns:
        Tuple of (scores, true_logprobs, false_logprobs).
    """
    from vllm import SamplingParams

    model = get_vllm_model(model_name, **model_kwargs)

    # Extract alphabetic identifiers for fuzzy matching
    true_id_str = re.sub(r"[^a-z]", "", true_token.lower())
    false_id_str = re.sub(r"[^a-z]", "", false_token.lower())

    logger.info(
        "Scoring %d prompts (true_id=%r, false_id=%r, top_logprobs=%d)",
        len(prompts), true_id_str, false_id_str, _NUM_LOGPROBS,
    )

    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=_NUM_LOGPROBS,
    )

    true_lps = np.zeros(len(prompts))
    false_lps = np.zeros(len(prompts))

    for i in range(0, len(prompts), _BATCH_CHUNK_SIZE):
        chunk = prompts[i : i + _BATCH_CHUNK_SIZE]
        outputs = model.generate(chunk, params)

        for j, output in enumerate(outputs):
            token_logprobs = output.outputs[0].logprobs
            if token_logprobs and len(token_logprobs) > 0:
                logprob_dict = token_logprobs[0]
                t_lp, f_lp = _aggregate_label_probs(
                    logprob_dict, true_id_str, false_id_str,
                )
                true_lps[i + j] = t_lp
                false_lps[i + j] = f_lp

    scores = true_lps - false_lps
    return scores, true_lps, false_lps


def get_token_id(model_name: str, token_str: str) -> int:
    """Get the token ID for a label string using the model's tokenizer."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) != 1:
        stripped = token_str.lstrip()
        ids = tokenizer.encode(stripped, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Token {token_str!r} encodes to {len(ids)} tokens, expected 1."
        )
    return ids[0]
