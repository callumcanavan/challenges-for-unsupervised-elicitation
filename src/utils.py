"""General utilities: seeding, CSV helpers, label normalization."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Set seeds for random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


TRUTHY_VALUES = {True, 1, "True", "true", "1", "yes", "Yes"}
FALSY_VALUES = {False, 0, "False", "false", "0", "no", "No"}


def normalize_label(label: bool | int | str) -> bool:
    """Convert various label representations to bool."""
    if label in TRUTHY_VALUES:
        return True
    if label in FALSY_VALUES:
        return False
    raise ValueError(f"Cannot normalize label: {label!r}")


def write_csv_header(path: Path, fieldnames: list[str]) -> None:
    """Create a CSV file with the given header row."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def write_csv_row(path: Path, row: dict) -> None:
    """Append a single row to an existing CSV file."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    """Append multiple rows to an existing CSV file."""
    if not rows:
        return
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writerows(rows)
