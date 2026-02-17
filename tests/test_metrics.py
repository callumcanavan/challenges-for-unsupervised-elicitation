"""Tests for src/metrics.py."""

import numpy as np
import pytest

from src.metrics import compute_accuracy, compute_auroc, compute_contribution_to_variance


class TestComputeAuroc:
    def test_perfect_separation(self):
        scores = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        labels = np.array([True, True, True, False, False, False])
        assert compute_auroc(scores, labels) == 1.0

    def test_inverse_separation(self):
        scores = np.array([-1.0, -2.0, -3.0, 1.0, 2.0, 3.0])
        labels = np.array([True, True, True, False, False, False])
        assert compute_auroc(scores, labels) == 0.0

    def test_random_chance(self):
        rng = np.random.RandomState(42)
        scores = rng.randn(1000)
        labels = rng.rand(1000) > 0.5
        auroc = compute_auroc(scores, labels)
        assert abs(auroc - 0.5) < 0.1

    def test_single_class(self):
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([True, True, True])
        assert compute_auroc(scores, labels) == 0.5


class TestComputeAccuracy:
    def test_perfect(self):
        scores = np.array([1.0, 2.0, -1.0, -2.0])
        labels = np.array([True, True, False, False])
        assert compute_accuracy(scores, labels) == 1.0

    def test_half(self):
        scores = np.array([1.0, -1.0, 1.0, -1.0])
        labels = np.array([True, True, False, False])
        assert compute_accuracy(scores, labels) == 0.5

    def test_custom_threshold(self):
        scores = np.array([0.5, 0.6, 0.3, 0.1])
        labels = np.array([True, True, False, False])
        assert compute_accuracy(scores, labels, threshold=0.4) == 1.0


class TestContributionToVariance:
    def test_single_group(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        groups = np.array(["a", "a", "a", "a"])
        result = compute_contribution_to_variance(scores, groups)
        assert abs(result["total_variance"] - np.var(scores)) < 1e-6
        assert "a" in result["group_stats"]

    def test_two_groups(self):
        scores = np.array([1.0, 1.0, 10.0, 10.0])
        groups = np.array(["a", "a", "b", "b"])
        result = compute_contribution_to_variance(scores, groups)
        total = sum(
            g["contribution_to_total_variance"]
            for g in result["group_stats"].values()
        )
        assert abs(total - result["total_variance"]) < 1e-6

    def test_group_stats_complete(self):
        scores = np.array([1.0, 2.0, 3.0])
        groups = np.array(["x", "x", "y"])
        result = compute_contribution_to_variance(scores, groups)
        for stats in result["group_stats"].values():
            assert "mean" in stats
            assert "variance" in stats
            assert "count" in stats
            assert "weight" in stats
