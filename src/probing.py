"""Probing methods: CCS, PCA, supervised, random, EH, UEEH, and ensembles.

All training functions take activation tensors and return trained probes.
Scoring functions take probes and activations and return per-item scores.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from src.config import (
    DEFAULT_CCS_EPOCHS,
    DEFAULT_CCS_LR,
    DEFAULT_CCS_RETRIES,
    DEFAULT_CCS_WEIGHT_DECAY,
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_ENSEMBLE_TEMPERATURE,
    DEFAULT_SUPERVISED_MAX_EPOCHS,
    DEFAULT_SUPERVISED_PATIENCE,
    DEFAULT_UEEH_ALPHA,
)

logger = logging.getLogger(__name__)

EPS = 1e-12

# ---------------------------------------------------------------------------
# LinearProbe
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Single linear layer probe."""

    def __init__(self, in_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def init_random_probe(dim: int) -> LinearProbe:
    """Create a probe with random unit-norm weights and zero bias."""
    probe = LinearProbe(dim)
    with torch.no_grad():
        w = torch.randn_like(probe.linear.weight)
        probe.linear.weight.copy_(w / (w.norm() + EPS))
        if probe.linear.bias is not None:
            probe.linear.bias.zero_()
    return probe


def _flip_probe(probe: LinearProbe) -> None:
    """Negate probe weights in-place."""
    with torch.no_grad():
        probe.linear.weight.mul_(-1)
        if probe.linear.bias is not None:
            probe.linear.bias.mul_(-1)


# ---------------------------------------------------------------------------
# CCS (Contrast-Consistent Search)
# ---------------------------------------------------------------------------


def _ccs_loss(
    probe: LinearProbe,
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
) -> torch.Tensor:
    """Compute CCS loss: consistency + confidence.

    Consistency: (p_pos - (1 - p_neg))^2 — makes predictions on pos/neg
    contrastive pairs opposite.
    Confidence: min(p_pos, p_neg)^2 — pushes predictions away from 0.5.
    """
    logit_pos = probe(X_pos).squeeze(-1)
    logit_neg = probe(X_neg).squeeze(-1)
    p_pos = torch.sigmoid(logit_pos)
    p_neg = torch.sigmoid(logit_neg)

    consistency = ((p_pos - (1 - p_neg)) ** 2).mean()
    confidence = (torch.min(p_pos, p_neg) ** 2).mean()
    return consistency + confidence


def train_ccs(
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
    *,
    lr: float = DEFAULT_CCS_LR,
    num_epochs: int = DEFAULT_CCS_EPOCHS,
    num_retries: int = DEFAULT_CCS_RETRIES,
    weight_decay: float = DEFAULT_CCS_WEIGHT_DECAY,
    batch_size: int = -1,
    device: str | torch.device | None = None,
) -> LinearProbe:
    """Train a CCS probe on contrastive activations.

    Runs multiple random initializations and keeps the probe with the
    lowest final loss.

    Args:
        X_pos: Positive activations, shape (N, D).
        X_neg: Negative activations, shape (N, D).
        lr: Learning rate.
        num_epochs: Training epochs per retry.
        num_retries: Number of random restarts.
        weight_decay: L2 regularization.
        batch_size: Batch size (-1 for full batch).
        device: Device for training.

    Returns:
        Trained LinearProbe.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_pos = X_pos.to(device)
    X_neg = X_neg.to(device)
    dim = X_pos.shape[-1]
    bsz = len(X_pos) if batch_size == -1 else batch_size

    best_probe: LinearProbe | None = None
    best_loss = float("inf")

    for retry in range(num_retries):
        probe = init_random_probe(dim).to(device)
        optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(num_epochs):
            perm = torch.randperm(len(X_pos), device=device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_pos), bsz):
                idx = perm[i : i + bsz]
                optimizer.zero_grad()
                loss = _ccs_loss(probe, X_pos[idx], X_neg[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)
        if final_loss < best_loss:
            best_loss = final_loss
            best_probe = LinearProbe(dim)
            best_probe.load_state_dict(
                {k: v.cpu() for k, v in probe.state_dict().items()}
            )

    logger.info("CCS training done. Best loss: %.6f", best_loss)
    return best_probe


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def train_pca(X_diff: torch.Tensor, n_components: int = 1) -> PCA:
    """Fit PCA on activation differences (X_pos - X_neg).

    Args:
        X_diff: Shape (N, D).
        n_components: Number of principal components.

    Returns:
        Fitted sklearn PCA.
    """
    X = X_diff.cpu().numpy() if isinstance(X_diff, torch.Tensor) else X_diff
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


# ---------------------------------------------------------------------------
# Supervised logistic regression
# ---------------------------------------------------------------------------


def train_supervised(
    X: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    X_val: torch.Tensor | np.ndarray | None = None,
    y_val: torch.Tensor | np.ndarray | None = None,
    *,
    lr: float = 0.03,
    weight_decay: float = 0.01,
    patience: int = DEFAULT_SUPERVISED_PATIENCE,
    max_epochs: int = DEFAULT_SUPERVISED_MAX_EPOCHS,
    batch_size: int = -1,
    device: str | torch.device | None = None,
) -> LinearProbe:
    """Train a supervised linear probe with optional early stopping on val loss.

    Uses AdamW optimizer with BCEWithLogitsLoss, matching the reference
    implementation. When validation data is provided, monitors validation
    loss for early stopping.

    Args:
        X: Training features, shape (N, D).
        y: Training labels, shape (N,).
        X_val: Validation features (enables early stopping).
        y_val: Validation labels.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        patience: Early stopping patience (epochs without improvement).
        max_epochs: Maximum training epochs.
        batch_size: Batch size (-1 for full batch).
        device: Training device.

    Returns:
        Trained LinearProbe.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.as_tensor(X, dtype=torch.float32).to(device)
    y_t = torch.as_tensor(y, dtype=torch.float32).to(device)

    dim = X_t.shape[-1]
    probe = LinearProbe(dim).to(device)
    with torch.no_grad():
        w = torch.randn_like(probe.linear.weight)
        probe.linear.weight.copy_(w / (w.norm() + EPS))
        if probe.linear.bias is not None:
            probe.linear.bias.zero_()

    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    bsz = len(X_t) if batch_size == -1 else batch_size

    has_val = X_val is not None and y_val is not None
    if has_val:
        X_val_t = torch.as_tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_left = patience

    for epoch in range(max_epochs):
        probe.train()
        perm = torch.randperm(len(X_t), device=device)
        X_shuffled = X_t[perm]
        y_shuffled = y_t[perm]

        for i in range(0, len(X_t), bsz):
            batch_X = X_shuffled[i : i + bsz]
            batch_y = y_shuffled[i : i + bsz]
            optimizer.zero_grad()
            logits = probe(batch_X).squeeze(-1)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

        if has_val:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(X_val_t).squeeze(-1)
                val_loss = loss_fn(val_logits, y_val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    logger.info(
                        "Early stopping at epoch %d (best val loss: %.6f)",
                        epoch, best_val_loss,
                    )
                    break

    if best_state is not None:
        result = LinearProbe(dim)
        result.load_state_dict(best_state)
        return result

    # Move to CPU
    result = LinearProbe(dim)
    result.load_state_dict({k: v.cpu() for k, v in probe.state_dict().items()})
    return result


def train_eh(
    X_easy: torch.Tensor | np.ndarray,
    y_easy: torch.Tensor | np.ndarray,
    X_val: torch.Tensor | np.ndarray | None = None,
    y_val: torch.Tensor | np.ndarray | None = None,
    **kwargs,
) -> LinearProbe:
    """Train a supervised probe on an easy dataset (easy-to-hard transfer).

    This is simply supervised training on labeled easy data. The probe is then
    evaluated on a harder target dataset.
    """
    return train_supervised(X_easy, y_easy, X_val, y_val, **kwargs)


# ---------------------------------------------------------------------------
# UEEH (joint CCS + supervised)
# ---------------------------------------------------------------------------


def train_ueeh(
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
    X_easy: torch.Tensor,
    y_easy: torch.Tensor,
    *,
    alpha: float = DEFAULT_UEEH_ALPHA,
    mode: str = "paired",
    lr: float = DEFAULT_CCS_LR,
    num_epochs: int = DEFAULT_CCS_EPOCHS,
    weight_decay: float = DEFAULT_CCS_WEIGHT_DECAY,
    batch_size: int = -1,
    device: str | torch.device | None = None,
) -> LinearProbe:
    """Train a joint UEEH probe combining CCS and supervised losses.

    In "paired" mode, the loss is ``alpha * L_supervised + (1-alpha) * L_ccs``.
    In "alternate" mode, each batch randomly picks one loss with probability
    ``alpha`` for supervised.

    Args:
        X_pos: Positive contrastive activations from hard dataset, (N, D).
        X_neg: Negative contrastive activations from hard dataset, (N, D).
        X_easy: Concatenated [X_pos_easy, X_neg_easy] features, (M, D).
        y_easy: Labels for easy data, (M,).
        alpha: Weighting between supervised and CCS (0.5 = equal).
        mode: ``"paired"`` or ``"alternate"``.
        lr: Learning rate.
        num_epochs: Training epochs.
        weight_decay: L2 regularization.
        batch_size: Batch size (-1 for full batch).
        device: Training device.

    Returns:
        Trained LinearProbe.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_pos = X_pos.to(device)
    X_neg = X_neg.to(device)
    X_easy = X_easy.to(device)
    y_easy = y_easy.to(device).float()

    dim = X_pos.shape[-1]
    probe = init_random_probe(dim).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    sup_loss_fn = nn.BCEWithLogitsLoss()

    bsz = max(len(X_pos), len(X_easy)) if batch_size == -1 else batch_size

    def _make_indices(n: int, length: int) -> torch.Tensor:
        """Create index tensor of given length from n items (with wrapping)."""
        if n >= length:
            return torch.randperm(n, device=device)[:length]
        reps = (length + n - 1) // n
        return torch.randperm(n, device=device).repeat(reps)[:length]

    for _ in range(num_epochs):
        L = max(len(X_pos), len(X_easy))
        idx_pos = _make_indices(len(X_pos), L)
        idx_neg = _make_indices(len(X_neg), L)
        idx_easy = _make_indices(len(X_easy), L)

        for i in range(0, L, bsz):
            j = min(i + bsz, L)
            xp = X_pos[idx_pos[i:j]]
            xn = X_neg[idx_neg[i:j]]
            xe = X_easy[idx_easy[i:j]]
            ye = y_easy[idx_easy[i:j]]

            optimizer.zero_grad()

            if mode == "paired":
                loss_sup = sup_loss_fn(probe(xe).squeeze(-1), ye)
                loss_ccs = _ccs_loss(probe, xp, xn)
                loss = alpha * loss_sup + (1 - alpha) * loss_ccs
            elif mode == "alternate":
                if np.random.rand() < alpha:
                    loss = sup_loss_fn(probe(xe).squeeze(-1), ye)
                else:
                    loss = _ccs_loss(probe, xp, xn)
            else:
                raise ValueError(f"Unknown mode: {mode!r}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

    # Move to CPU
    result = LinearProbe(dim)
    result.load_state_dict({k: v.cpu() for k, v in probe.state_dict().items()})
    return result


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------


def train_random_ensemble(
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
    *,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    easy_dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    temperature: float = DEFAULT_ENSEMBLE_TEMPERATURE,
    scale_by_variance: bool = False,
) -> list[LinearProbe]:
    """Train an ensemble of random linear probes.

    Without ``easy_dataset``: uses consensus weighting (each probe's weight
    is +1 if it agrees with the running ensemble on >50% of items, -1 otherwise).

    With ``easy_dataset``: uses EH weighting (weight = softmax of AUROC on
    easy data minus 0.5).

    Args:
        X_pos: Positive activations, (N, D).
        X_neg: Negative activations, (N, D).
        ensemble_size: Number of probes.
        easy_dataset: Tuple of (X_pos_easy, X_neg_easy, y_easy) for EH weighting.
        temperature: Softmax temperature for EH weighting.
        scale_by_variance: If True, scale each probe by 1/std(scores).

    Returns:
        List of weighted LinearProbe instances.
    """
    dim = X_pos.shape[-1]
    probes = [init_random_probe(dim) for _ in range(ensemble_size)]

    with torch.no_grad():
        if scale_by_variance:
            for probe in probes:
                scores = probe(X_pos).squeeze(-1) - probe(X_neg).squeeze(-1)
                std = scores.std().clamp_min(EPS)
                probe.linear.weight.div_(std)

        if easy_dataset is None:
            # Consensus weighting
            scores_0 = probes[0](X_pos).squeeze(-1) - probes[0](X_neg).squeeze(-1)
            ensemble_scores = scores_0.clone()

            for i in range(1, len(probes)):
                ensemble_preds = ensemble_scores > 0
                probe_scores = probes[i](X_pos).squeeze(-1) - probes[i](X_neg).squeeze(-1)
                probe_preds = probe_scores > 0
                agreement = (ensemble_preds == probe_preds).float().mean()

                if agreement < 0.5:
                    _flip_probe(probes[i])
                    probe_scores = -probe_scores

                # Running weighted average
                ensemble_scores = (i * ensemble_scores + probe_scores) / (i + 1)
        else:
            # EH weighting
            X_pos_easy, X_neg_easy, y_easy = easy_dataset
            y_np = y_easy.cpu().numpy() if isinstance(y_easy, torch.Tensor) else np.asarray(y_easy)

            aurocs = []
            for probe in probes:
                scores_easy = probe(X_pos_easy).squeeze(-1) - probe(X_neg_easy).squeeze(-1)
                auroc = roc_auc_score(y_np, scores_easy.cpu().numpy()) - 0.5
                aurocs.append(auroc)

            weights = torch.softmax(
                torch.tensor(aurocs, dtype=X_pos.dtype) / temperature,
                dim=0,
            )
            for probe, w in zip(probes, weights):
                probe.linear.weight.data.mul_(w.item())

    return probes


def train_pca_ensemble(
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
    *,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    easy_dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    temperature: float = DEFAULT_ENSEMBLE_TEMPERATURE,
    scale_by_variance: bool = False,
) -> PCA:
    """Train a PCA ensemble on contrastive differences.

    Similar weighting strategies as :func:`train_random_ensemble` but
    using PCA components instead of random probes.

    Args:
        X_pos: Positive activations, (N, D).
        X_neg: Negative activations, (N, D).
        ensemble_size: Number of PCA components.
        easy_dataset: Tuple for EH weighting.
        temperature: Softmax temperature.
        scale_by_variance: Scale components by 1/sqrt(explained_variance).

    Returns:
        Fitted PCA with (possibly weighted/flipped) components.
    """
    X_diff = (X_pos - X_neg).cpu().numpy()
    pca = PCA(n_components=ensemble_size)
    pca.fit(X_diff)

    if scale_by_variance:
        pca.components_ /= np.maximum(np.sqrt(pca.explained_variance_)[:, None], EPS)

    if easy_dataset is None:
        # Consensus weighting
        all_components = pca.transform(X_diff)  # (N, ensemble_size)
        ensemble_scores = all_components[:, 0].copy()

        for i in range(1, ensemble_size):
            ensemble_preds = ensemble_scores > 0
            comp_scores = all_components[:, i]
            comp_preds = comp_scores > 0
            agreement = np.mean(ensemble_preds == comp_preds)

            if agreement < 0.5:
                pca.components_[i] *= -1
                comp_scores = -comp_scores

            ensemble_scores = (i * ensemble_scores + comp_scores) / (i + 1)
    else:
        # EH weighting
        X_pos_easy, X_neg_easy, y_easy = easy_dataset
        easy_diff = (X_pos_easy - X_neg_easy).cpu().numpy()
        y_np = y_easy.cpu().numpy() if isinstance(y_easy, torch.Tensor) else np.asarray(y_easy)

        easy_components = pca.transform(easy_diff)
        aurocs = []
        for i in range(ensemble_size):
            auroc = roc_auc_score(y_np, easy_components[:, i]) - 0.5
            aurocs.append(auroc)

        weights = torch.softmax(
            torch.tensor(aurocs) / temperature, dim=0
        ).numpy()
        for i in range(ensemble_size):
            pca.components_[i] *= weights[i]

    return pca


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def score_with_probe(
    probe: LinearProbe,
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
) -> np.ndarray:
    """Score items using a linear probe on contrastive activations.

    Returns:
        Per-item scores (positive = more likely true).
    """
    with torch.no_grad():
        scores = probe(X_pos).squeeze(-1) - probe(X_neg).squeeze(-1)
    return scores.cpu().numpy()


def score_with_pca(
    pca: PCA,
    X_diff: torch.Tensor | np.ndarray,
    component: int = 0,
) -> np.ndarray:
    """Score items by projecting differences onto a PCA component."""
    X = X_diff.cpu().numpy() if isinstance(X_diff, torch.Tensor) else X_diff
    projections = pca.transform(X)
    return projections[:, component]


def score_with_ensemble(
    probes: list[LinearProbe],
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
) -> np.ndarray:
    """Score items using an ensemble of probes (mean of individual scores)."""
    with torch.no_grad():
        all_scores = torch.stack(
            [p(X_pos).squeeze(-1) - p(X_neg).squeeze(-1) for p in probes],
            dim=1,
        )
    return all_scores.mean(dim=1).cpu().numpy()


def score_with_pca_ensemble(
    pca: PCA,
    X_diff: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Score items using all PCA components (mean of projections)."""
    X = X_diff.cpu().numpy() if isinstance(X_diff, torch.Tensor) else X_diff
    projections = pca.transform(X)
    return projections.mean(axis=1)


def compute_flip_direction(
    scores: np.ndarray,
    labels: np.ndarray,
) -> bool:
    """Determine whether scores should be negated (AUROC < 0.5).

    Should be called with **train** scores and labels. The returned flag
    is then applied to both train and test scores via :func:`apply_flip`.
    """
    labels = np.asarray(labels, dtype=bool)
    if len(np.unique(labels)) < 2:
        return False
    auroc = roc_auc_score(labels, scores)
    return bool(auroc < 0.5)


def apply_flip(scores: np.ndarray, flip: bool) -> np.ndarray:
    """Negate *scores* if *flip* is True."""
    return -scores if flip else scores
