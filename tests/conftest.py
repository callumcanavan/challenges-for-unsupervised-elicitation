"""Shared test fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Sample items
# ---------------------------------------------------------------------------


@pytest.fixture
def icm_items() -> list[dict]:
    """Sample items for the ICM binary format."""
    return [
        {"question": "What is 2+2?", "choice": "4", "label": True},
        {"question": "What is 2+2?", "choice": "5", "label": False},
        {"question": "What is 3+3?", "choice": "6", "label": True},
        {"question": "What is 3+3?", "choice": "7", "label": False},
        {"question": "What is 1+1?", "choice": "2", "label": True},
        {"question": "What is 1+1?", "choice": "3", "label": False},
    ]


@pytest.fixture
def ctrl_z_items() -> list[dict]:
    """Sample items for the ctrl_z format."""
    items = []
    for i, label in enumerate([True, False]):
        item = {
            "task_description": f"Task {i}",
            "label": label,
        }
        for j in range(10):
            item[f"command_{j}"] = f"cmd_{i}_{j}"
        items.append(item)
    return items


@pytest.fixture
def gsm8k_test_items() -> list[dict]:
    """Sample items that need field mapping (cot -> choice)."""
    return [
        {"question": "What is 2+2?", "cot": "2+2=4", "answer": "4", "label": True},
        {"question": "What is 3+3?", "cot": "3+3=6", "answer": "6", "label": True},
    ]


# ---------------------------------------------------------------------------
# Synthetic activations
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_activations() -> dict:
    """Synthetic activations with known separable structure.

    True items have positive direction, false items have negative direction
    in the first dimension, with noise in other dimensions.
    """
    rng = np.random.RandomState(42)
    n = 100
    dim = 16
    labels = np.array([True] * (n // 2) + [False] * (n // 2))

    # Create separable data: true items positive in dim 0, false negative
    signal = np.zeros((n, dim))
    signal[:n // 2, 0] = 2.0
    signal[n // 2:, 0] = -2.0
    noise = rng.randn(n, dim) * 0.1

    X_pos = torch.tensor(signal + noise, dtype=torch.float32)
    X_neg = torch.tensor(-signal + noise, dtype=torch.float32)

    return {
        "X_pos": X_pos,
        "X_neg": X_neg,
        "labels": labels,
        "n": n,
        "dim": dim,
    }


# ---------------------------------------------------------------------------
# Temporary dataset files
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dataset_dir(tmp_path, monkeypatch):
    """Create temporary dataset files and point DATA_DIR to them."""
    import src.data as data_mod
    import src.storage_utils as su

    # Create training data
    train_data = [
        {"question": f"Q{i}", "choice": f"C{i}", "label": i % 2 == 0}
        for i in range(20)
    ]
    test_data = [
        {"question": f"TQ{i}", "choice": f"TC{i}", "label": i % 2 == 0}
        for i in range(10)
    ]

    (tmp_path / "train_gsm8k.json").write_text(json.dumps(train_data))
    (tmp_path / "test_gsm8k.json").write_text(json.dumps(test_data))

    # Patch DATA_DIR in both modules (data.py imports it by value)
    monkeypatch.setattr(su, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)

    return tmp_path
