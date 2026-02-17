"""Activation extraction from HuggingFace models.

Extracts last-token hidden states and logprobs for contrastive probing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

EPS = 1e-12


@dataclass
class ActivationResult:
    """Container for extracted activations and logprobs."""

    activations: dict[int, tuple[torch.Tensor, torch.Tensor]]
    """Per-layer (X_pos, X_neg) tensors, each of shape (N, D)."""

    true_logprobs: torch.Tensor
    """Log-probabilities of the true token, shape (N,)."""

    false_logprobs: torch.Tensor
    """Log-probabilities of the false token, shape (N,)."""

    logprob_scores: torch.Tensor
    """true_logprobs - false_logprobs, shape (N,)."""


@dataclass
class NormalizationStats:
    """Statistics for activation normalization (mean-centering + scaling)."""

    mean: torch.Tensor
    scale_factor: torch.Tensor


def load_model(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load a HuggingFace model and tokenizer.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
    )
    model.eval()
    return model, tokenizer


def _get_token_id(tokenizer, token_str: str) -> int:
    """Get the token ID for a label token string."""
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) != 1:
        # Try without leading space
        stripped = token_str.lstrip()
        ids = tokenizer.encode(stripped, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Token {token_str!r} encodes to {len(ids)} tokens, expected 1. "
            f"IDs: {ids}"
        )
    return ids[0]


def _extract_batch_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Extract last-token hidden states and pre-token logits for a batch.

    Hidden states are extracted at the last token (the appended True/False
    token) for contrastive probing.  Logits are extracted one position
    *before* the appended token so that log P(True) and log P(False) reflect
    the model's prediction from the base prompt.

    Returns:
        (layer_hidden_states, pre_token_logits)
        - layer_hidden_states: {layer: (batch, D)}
        - pre_token_logits: (batch, vocab_size)
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Detect if truncation removed tokens (e.g. the appended True/False token).
    for i, prompt in enumerate(prompts):
        untokenized_len = len(tokenizer.encode(prompt, add_special_tokens=True))
        actual_len = inputs["attention_mask"][i].sum().item()
        if actual_len < untokenized_len:
            raise ValueError(
                f"Prompt {i} was truncated from {untokenized_len} to "
                f"{actual_len} tokens. The appended label token may have been "
                f"dropped. Shorten the prompt or increase max_length."
            )

    with torch.no_grad():
        outputs = model(**inputs)

    # Find the last non-padding token position for each sequence
    attention_mask = inputs["attention_mask"]
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
    batch_idx = torch.arange(len(prompts))

    # Hidden states: at the last token (the appended True/False token)
    hidden_states = outputs.hidden_states  # tuple of (batch, seq, D)
    layer_results: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        hs = hidden_states[layer_idx]  # (batch, seq, D)
        last_token_hs = hs[batch_idx, seq_lengths]  # (batch, D)
        layer_results[layer_idx] = last_token_hs.cpu()

    # Logits: one position before the appended token for logprob computation
    logits = outputs.logits  # (batch, seq, vocab)
    pre_token_logits = logits[batch_idx, seq_lengths - 1]  # (batch, vocab)

    return layer_results, pre_token_logits.cpu()


def extract_contrastive_activations(
    model,
    tokenizer,
    items: list[dict],
    prompt_format,
    layers: list[int],
    batch_size: int = 1,
) -> ActivationResult:
    """Extract contrastive activations and logprobs for a list of items.

    For each item, creates positive (true-token appended) and negative
    (false-token appended) prompts, then extracts last-token hidden states.

    Args:
        model: HuggingFace model with ``output_hidden_states=True``.
        tokenizer: Corresponding tokenizer.
        items: Prepared dataset items.
        prompt_format: PromptFormat instance.
        layers: Layer indices to extract.
        batch_size: Batch size for inference.

    Returns:
        ActivationResult with per-layer (X_pos, X_neg) tensors and logprobs.
    """
    device = next(model.parameters()).device

    true_id = _get_token_id(tokenizer, prompt_format.true_token)
    false_id = _get_token_id(tokenizer, prompt_format.false_token)

    # Build all prompts
    pos_prompts: list[str] = []
    neg_prompts: list[str] = []
    for item in items:
        p_pos, p_neg = prompt_format.format_contrastive_prompts(item)
        pos_prompts.append(p_pos)
        neg_prompts.append(p_neg)

    # Process in batches
    all_pos_layers: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    all_neg_layers: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    all_pos_logits: list[torch.Tensor] = []
    all_neg_logits: list[torch.Tensor] = []

    n = len(items)
    for start in tqdm(range(0, n, batch_size), desc="Extracting activations"):
        end = min(start + batch_size, n)

        # Positive batch
        pos_batch = pos_prompts[start:end]
        pos_hs, pos_logit = _extract_batch_hidden_states(
            model, tokenizer, pos_batch, layers, device
        )
        for layer_idx in layers:
            all_pos_layers[layer_idx].append(pos_hs[layer_idx])
        all_pos_logits.append(pos_logit)

        # Negative batch
        neg_batch = neg_prompts[start:end]
        neg_hs, neg_logit = _extract_batch_hidden_states(
            model, tokenizer, neg_batch, layers, device
        )
        for layer_idx in layers:
            all_neg_layers[layer_idx].append(neg_hs[layer_idx])
        all_neg_logits.append(neg_logit)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate results
    activations: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in layers:
        X_pos = torch.cat(all_pos_layers[layer_idx], dim=0)
        X_neg = torch.cat(all_neg_layers[layer_idx], dim=0)
        activations[layer_idx] = (X_pos, X_neg)

    # Compute logprobs from logits
    pos_logits = torch.cat(all_pos_logits, dim=0)  # (N, vocab)
    neg_logits = torch.cat(all_neg_logits, dim=0)  # (N, vocab)

    # Logprobs are extracted one position before the appended token,
    # giving P(True) and P(False) from the base prompt.
    pos_log_probs = torch.log_softmax(pos_logits, dim=-1)
    true_logprobs = pos_log_probs[:, true_id]
    false_logprobs = pos_log_probs[:, false_id]
    logprob_scores = true_logprobs - false_logprobs

    return ActivationResult(
        activations=activations,
        true_logprobs=true_logprobs,
        false_logprobs=false_logprobs,
        logprob_scores=logprob_scores,
    )


def normalize_activations(
    X: torch.Tensor,
    stats: NormalizationStats | None = None,
) -> tuple[torch.Tensor, NormalizationStats]:
    """Mean-center and scale activations.

    Subtracts the mean and divides by ``row_norm_mean * sqrt(dim)`` to
    match the normalization used in the reference implementation.

    If ``stats`` is provided, uses those statistics (for applying train
    normalization to test data). Otherwise computes from the data.

    Args:
        X: Activations, shape (N, D).
        stats: Pre-computed normalization statistics, or None.

    Returns:
        Tuple of (normalized_X, stats).
    """
    if stats is None:
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean
        row_norm_mean = X_centered.norm(dim=1, keepdim=True).mean(dim=0, keepdim=True)
        scale_factor = (row_norm_mean * torch.sqrt(torch.tensor(X.shape[1], dtype=X.dtype))).clamp_min(EPS)
        stats = NormalizationStats(mean=mean.squeeze(0), scale_factor=scale_factor.squeeze(0))

    X_norm = (X - stats.mean) / stats.scale_factor
    return X_norm, stats
