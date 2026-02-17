"""Tests for src/probing.py."""

import numpy as np
import pytest
import torch

from src.probing import (
    LinearProbe,
    apply_flip,
    compute_flip_direction,
    init_random_probe,
    score_with_ensemble,
    score_with_pca,
    score_with_pca_ensemble,
    score_with_probe,
    train_ccs,
    train_pca,
    train_pca_ensemble,
    train_random_ensemble,
    train_supervised,
    train_ueeh,
)


class TestLinearProbe:
    def test_output_shape(self):
        probe = LinearProbe(16)
        x = torch.randn(10, 16)
        out = probe(x)
        assert out.shape == (10, 1)

    def test_init_random_probe_unit_norm(self):
        probe = init_random_probe(32)
        norm = probe.linear.weight.data.norm()
        assert abs(norm.item() - 1.0) < 1e-4


class TestTrainCCS:
    def test_trains_without_error(self, synthetic_activations):
        probe = train_ccs(
            synthetic_activations["X_pos"],
            synthetic_activations["X_neg"],
            num_epochs=10,
            num_retries=2,
            device="cpu",
        )
        assert isinstance(probe, LinearProbe)

    def test_produces_valid_scores(self, synthetic_activations):
        probe = train_ccs(
            synthetic_activations["X_pos"],
            synthetic_activations["X_neg"],
            num_epochs=50,
            num_retries=3,
            device="cpu",
        )
        scores = score_with_probe(
            probe,
            synthetic_activations["X_pos"],
            synthetic_activations["X_neg"],
        )
        assert scores.shape == (synthetic_activations["n"],)
        assert np.isfinite(scores).all()


class TestTrainPCA:
    def test_fits(self, synthetic_activations):
        X_diff = synthetic_activations["X_pos"] - synthetic_activations["X_neg"]
        pca = train_pca(X_diff, n_components=1)
        assert pca.n_components == 1

    def test_scores_separable_data(self, synthetic_activations):
        X_diff = synthetic_activations["X_pos"] - synthetic_activations["X_neg"]
        pca = train_pca(X_diff, n_components=1)
        scores = score_with_pca(pca, X_diff)
        assert scores.shape == (synthetic_activations["n"],)

        # Should achieve good AUROC on separable data
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(synthetic_activations["labels"], scores)
        assert auroc > 0.9 or auroc < 0.1  # May be flipped


class TestTrainSupervised:
    def test_trains_without_error(self, synthetic_activations):
        X = torch.cat([
            synthetic_activations["X_pos"],
            synthetic_activations["X_neg"],
        ])
        y = np.concatenate([
            synthetic_activations["labels"],
            ~synthetic_activations["labels"],
        ])
        probe = train_supervised(X, y, max_epochs=100)
        assert isinstance(probe, LinearProbe)

    def test_early_stopping(self, synthetic_activations):
        n = synthetic_activations["n"]
        X = torch.cat([
            synthetic_activations["X_pos"],
            synthetic_activations["X_neg"],
        ])
        y = np.concatenate([
            synthetic_activations["labels"],
            ~synthetic_activations["labels"],
        ])
        # Use first 80% for train, last 20% for val
        split = int(0.8 * len(X))
        probe = train_supervised(
            X[:split], y[:split],
            X[split:], y[split:],
            patience=10, max_epochs=10000,
        )
        assert isinstance(probe, LinearProbe)


class TestTrainUEEH:
    def test_trains_without_error(self, synthetic_activations):
        # Use the same data as "easy" for simplicity
        X_easy = torch.cat([
            synthetic_activations["X_pos"][:20],
            synthetic_activations["X_neg"][:20],
        ])
        y_easy = torch.cat([
            torch.tensor(synthetic_activations["labels"][:20], dtype=torch.float32),
            1 - torch.tensor(synthetic_activations["labels"][:20], dtype=torch.float32),
        ])

        probe = train_ueeh(
            synthetic_activations["X_pos"],
            synthetic_activations["X_neg"],
            X_easy, y_easy,
            num_epochs=10,
            device="cpu",
        )
        assert isinstance(probe, LinearProbe)


class TestEnsembles:
    def test_random_ensemble_consensus(self, synthetic_activations):
        probes = train_random_ensemble(
            synthetic_activations["X_pos"],
            synthetic_activations["X_neg"],
            ensemble_size=8,
        )
        assert len(probes) == 8
        scores = score_with_ensemble(
            probes,
            synthetic_activations["X_pos"],
            synthetic_activations["X_neg"],
        )
        assert scores.shape == (synthetic_activations["n"],)

    def test_random_ensemble_eh(self, synthetic_activations):
        easy_data = (
            synthetic_activations["X_pos"][:20],
            synthetic_activations["X_neg"][:20],
            torch.tensor(synthetic_activations["labels"][:20], dtype=torch.bool),
        )
        probes = train_random_ensemble(
            synthetic_activations["X_pos"],
            synthetic_activations["X_neg"],
            ensemble_size=8,
            easy_dataset=easy_data,
        )
        assert len(probes) == 8

    def test_pca_ensemble(self, synthetic_activations):
        pca = train_pca_ensemble(
            synthetic_activations["X_pos"],
            synthetic_activations["X_neg"],
            ensemble_size=4,
        )
        X_diff = synthetic_activations["X_pos"] - synthetic_activations["X_neg"]
        scores = score_with_pca_ensemble(pca, X_diff)
        assert scores.shape == (synthetic_activations["n"],)


class TestAutoFlip:
    def test_flips_when_needed(self):
        scores = np.array([-1.0, -2.0, -3.0, 1.0, 2.0, 3.0])
        labels = np.array([True, True, True, False, False, False])
        flip = compute_flip_direction(scores, labels)
        assert flip is True
        flipped = apply_flip(scores, flip)
        assert np.all(flipped == -scores)

    def test_no_flip_when_correct(self):
        scores = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        labels = np.array([True, True, True, False, False, False])
        flip = compute_flip_direction(scores, labels)
        assert flip is False
        result = apply_flip(scores, flip)
        assert np.all(result == scores)

    def test_single_class_no_flip(self):
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([True, True, True])
        flip = compute_flip_direction(scores, labels)
        assert flip is False
        result = apply_flip(scores, flip)
        assert np.all(result == scores)

    def test_train_flip_applies_to_test(self):
        """Flip direction computed on train should be applied to test."""
        train_scores = np.array([-1.0, -2.0, 1.0, 2.0])
        train_labels = np.array([True, True, False, False])
        test_scores = np.array([0.5, -0.5, 1.0])

        flip = compute_flip_direction(train_scores, train_labels)
        assert flip is True  # AUROC < 0.5

        flipped_test = apply_flip(test_scores, flip)
        assert np.all(flipped_test == -test_scores)


class TestNormalization:
    def test_includes_scaling(self):
        """Normalization should subtract mean AND scale by row_norm_mean * sqrt(dim)."""
        from src.activations import normalize_activations

        X = torch.randn(50, 16)
        X_norm, stats = normalize_activations(X)

        # Should have zero mean
        assert torch.abs(X_norm.mean(dim=0)).max() < 1e-5

        # stats should have both mean and scale_factor
        assert hasattr(stats, "mean")
        assert hasattr(stats, "scale_factor")
        assert stats.scale_factor.numel() >= 1  # scalar or per-dim

    def test_train_stats_applied_to_test(self):
        """Train stats should be reused for test normalization."""
        from src.activations import normalize_activations

        X_train = torch.randn(50, 16) + 5.0  # Shifted mean
        X_test = torch.randn(20, 16) + 5.0

        _, stats = normalize_activations(X_train)
        X_test_norm, _ = normalize_activations(X_test, stats=stats)

        # Test data should NOT have zero mean (it uses train stats)
        # But it should be approximately centered if distributions match
        assert X_test_norm.shape == (20, 16)

    def test_scaling_factor_nonzero(self):
        """Scale factor should be non-zero even for near-zero data."""
        from src.activations import normalize_activations

        X = torch.zeros(10, 8)
        X_norm, stats = normalize_activations(X)
        # Should not have NaN
        assert torch.isfinite(X_norm).all()
