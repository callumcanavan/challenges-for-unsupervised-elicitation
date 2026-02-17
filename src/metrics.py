"""Evaluation metrics: AUROC, accuracy, and contribution to variance."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under ROC Curve.

    Args:
        scores: Per-item scores (higher = more likely true).
        labels: Ground-truth boolean labels.

    Returns:
        AUROC value in [0, 1]. Returns 0.5 if labels are all the same class.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Compute classification accuracy.

    Args:
        scores: Per-item scores.
        labels: Ground-truth boolean labels.
        threshold: Score threshold for predicting True.

    Returns:
        Fraction of correct predictions.
    """
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(scores) > threshold
    return float(np.mean(predictions == labels))


def compute_contribution_to_variance(
    scores: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Compute per-group contribution to total score variance.

    Used for the impossible-task experiment to measure how much confidence
    variance comes from each dataset source (e.g., gsm8k vs political).

    Args:
        scores: Per-item scores.
        groups: Per-item group labels (e.g. dataset source strings).

    Returns:
        Dict with keys:
        - ``"total_variance"``: Variance of all scores.
        - ``"group_stats"``: Dict mapping group -> {mean, variance, count,
          contribution_to_total_variance}.
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    total_var = float(np.var(scores))
    total_mean = float(np.mean(scores))
    n = len(scores)

    unique_groups = np.unique(groups)
    group_stats: dict = {}

    for g in unique_groups:
        mask = groups == g
        g_scores = scores[mask]
        g_mean = float(np.mean(g_scores))
        g_var = float(np.var(g_scores))
        g_count = int(mask.sum())
        g_weight = g_count / n

        # Contribution = weighted within-group variance + weighted squared
        # deviation of group mean from total mean
        contribution = g_weight * g_var + g_weight * (g_mean - total_mean) ** 2

        group_stats[str(g)] = {
            "mean": g_mean,
            "variance": g_var,
            "count": g_count,
            "weight": g_weight,
            "contribution_to_total_variance": contribution,
        }

    return {
        "total_variance": total_var,
        "group_stats": group_stats,
    }
