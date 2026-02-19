"""Dataset loading, item preparation, few-shot sampling, and mixed datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import DatasetConfig, get_dataset_config
from src.storage_utils import DATA_DIR
from src.utils import normalize_label


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dataset(dataset_name: str, split: str) -> list[dict]:
    """Load a dataset split from the ``datasets/`` directory.

    Args:
        dataset_name: Dataset identifier (e.g. ``"gsm8k_preference"``).
        split: Either ``"train"`` or ``"test"``.

    Returns:
        List of item dicts as stored in the JSON file.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    config = get_dataset_config(dataset_name)
    filename = config.train_file if split == "train" else config.test_file
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def prepare_items(
    data: list[dict],
    config: DatasetConfig,
) -> list[dict]:
    """Add computed fields to raw dataset items.

    For each item:
    - Applies ``config.field_mapping`` so template placeholders resolve.
    - Normalizes ``label`` to ``bool``.
    - Adds ``prompt`` by formatting through the prompt template.

    Args:
        data: Raw items from :func:`load_dataset`.
        config: Dataset configuration.

    Returns:
        New list of dicts with added/modified fields (originals are not mutated).
    """
    fmt = config.prompt_format
    prepared: list[dict] = []

    for raw in data:
        item = dict(raw)

        # Apply field mapping: copy source -> dest if dest is missing
        for src, dst in config.field_mapping.items():
            if dst not in item and src in item:
                item[dst] = item[src]

        # Normalize label
        if "label" in item:
            item["label"] = normalize_label(item["label"])

        # Build prompt
        item["prompt"] = fmt.format_prompt(item)

        prepared.append(item)

    return prepared


# ---------------------------------------------------------------------------
# Subsampling by true proportion
# ---------------------------------------------------------------------------


def subsample_by_true_proportion(
    data: list[dict],
    true_proportion: float,
    max_size: int | None = None,
) -> list[dict]:
    """Subsample data to achieve a target proportion of true-labeled items.

    The total size is determined by ``min(n_true, n_false)`` (the minority
    class count in the unaltered data) so that different proportions
    produce identically-sized datasets.  An explicit ``max_size`` further
    caps the result.

    Uses module-level random state (set via ``seed_all`` at script start).

    Args:
        data: Prepared items with boolean ``label`` field.
        true_proportion: Desired fraction of True labels (0.0 to 1.0).
        max_size: Maximum number of items to return.

    Returns:
        Subsampled list of items.

    Raises:
        ValueError: If the dataset is too small to satisfy the request.
    """

    true_items = [x for x in data if x["label"]]
    false_items = [x for x in data if not x["label"]]

    random.shuffle(true_items)
    random.shuffle(false_items)

    if true_proportion <= 0.0:
        total = min(len(true_items), len(false_items))
        if max_size is not None:
            total = min(total, max_size)
        result = false_items[:total]
    elif true_proportion >= 1.0:
        total = min(len(true_items), len(false_items))
        if max_size is not None:
            total = min(total, max_size)
        result = true_items[:total]
    else:
        # Use min(n_true, n_false) as the base so all TTPs get same total
        base_size = min(len(true_items), len(false_items))
        if max_size is not None:
            base_size = min(base_size, max_size)

        n_true = int(round(true_proportion * base_size))
        n_false = base_size - n_true

        if n_true > len(true_items):
            raise ValueError(
                f"Need {n_true} true items but only {len(true_items)} available"
            )
        if n_false > len(false_items):
            raise ValueError(
                f"Need {n_false} false items but only {len(false_items)} available"
            )

        result = true_items[:n_true] + false_items[:n_false]

    random.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Few-shot example sampling
# ---------------------------------------------------------------------------


def sample_few_shot_examples(
    pool: list[dict],
    n: int,
    label_key: str = "label",
    exclude_item: dict | None = None,
) -> tuple[list[dict], list[bool]]:
    """Sample balanced few-shot examples from a pool.

    Selects ``n // 2`` items whose *label_key* is True and the rest False,
    ensuring no example shares a question with the target or with other
    selected examples.

    Use ``label_key="label"`` for ground-truth balancing (golden few-shot)
    or ``label_key="predicted_label"`` for balancing on assigned labels
    (random / bootstrap few-shot — avoids GT leakage).

    Uses module-level random state (set via ``seed_all`` at script start).

    Args:
        pool: Candidate items. Each must have a boolean field *label_key*.
        n: Number of examples to select.
        label_key: Which boolean field to balance on.
        exclude_item: The target item to avoid collisions with.

    Returns:
        Tuple of (examples, assigned_labels) where *assigned_labels* are the
        values of *label_key* for the selected examples.
    """
    if n == 0:
        return [], []

    # Determine exclusion criteria from the target item
    exclude_questions: set[str] = set()
    if exclude_item is not None:
        if "question" in exclude_item:
            exclude_questions.add(exclude_item["question"])

    def _is_eligible(item: dict, used_questions: set) -> bool:
        q = item.get("question")
        if q is not None and q in (exclude_questions | used_questions):
            return False
        return True

    true_items = [x for x in pool if x[label_key]]
    false_items = [x for x in pool if not x[label_key]]
    random.shuffle(true_items)
    random.shuffle(false_items)

    n_true = n // 2
    n_false = n - n_true

    used_questions: set[str] = set()

    def _select(candidates: list[dict], count: int) -> list[dict]:
        selected: list[dict] = []
        for item in candidates:
            if len(selected) >= count:
                break
            if _is_eligible(item, used_questions):
                selected.append(item)
                q = item.get("question")
                if q is not None:
                    used_questions.add(q)
        return selected

    true_selected = _select(true_items, n_true)
    false_selected = _select(false_items, n_false)

    # If we couldn't get enough, fill from the other class
    deficit = n - len(true_selected) - len(false_selected)
    cross_class_borrowed = 0
    if deficit > 0:
        remaining = [
            x
            for x in (true_items + false_items)
            if x not in true_selected
            and x not in false_selected
            and _is_eligible(x, used_questions)
        ]
        extra = remaining[:deficit]
        cross_class_borrowed = len(extra)
        if extra:
            for item in extra:
                if item[label_key]:
                    true_selected.append(item)
                else:
                    false_selected.append(item)

    examples = true_selected + false_selected
    labels = [x[label_key] for x in examples]

    # Record cross-class borrowing on the examples for downstream logging
    if cross_class_borrowed > 0:
        for ex in examples:
            ex["_cross_class_borrowed"] = True

    # Shuffle so true/false aren't grouped
    combined = list(zip(examples, labels))
    random.shuffle(combined)
    examples = [x[0] for x in combined]
    labels = [x[1] for x in combined]

    return examples, labels


# ---------------------------------------------------------------------------
# Mixed datasets (GSMPolitical)
# ---------------------------------------------------------------------------


def build_mixed_dataset(
    dataset_a: str,
    dataset_b: str,
    train_size: int = 4000,
) -> tuple[list[dict], list[dict]]:
    """Construct a mixed train/test dataset from two dataset configs.

    The training set takes ``train_size // 2`` items from each dataset.
    The test set is the full concatenation of both test sets, with a
    ``dataset_source`` field added to each item.

    Uses module-level random state (set via ``seed_all`` at script start).

    Args:
        dataset_a: First dataset name.
        dataset_b: Second dataset name.
        train_size: Total training set size (split equally).

    Returns:
        Tuple of (mixed_train, mixed_test).
    """
    config_a = get_dataset_config(dataset_a)
    config_b = get_dataset_config(dataset_b)

    half = train_size // 2

    # Load and prepare both datasets
    train_a = prepare_items(load_dataset(dataset_a, "train"), config_a)
    train_b = prepare_items(load_dataset(dataset_b, "train"), config_b)
    test_a = prepare_items(load_dataset(dataset_a, "test"), config_a)
    test_b = prepare_items(load_dataset(dataset_b, "test"), config_b)

    # Subsample training sets
    if len(train_a) < half:
        raise ValueError(
            f"Dataset {dataset_a} train has {len(train_a)} items, need {half}"
        )
    if len(train_b) < half:
        raise ValueError(
            f"Dataset {dataset_b} train has {len(train_b)} items, need {half}"
        )
    random.shuffle(train_a)
    random.shuffle(train_b)
    train_a = train_a[:half]
    train_b = train_b[:half]

    # Add source labels
    for item in train_a:
        item["dataset_source"] = dataset_a
    for item in train_b:
        item["dataset_source"] = dataset_b
    for item in test_a:
        item["dataset_source"] = dataset_a
    for item in test_b:
        item["dataset_source"] = dataset_b

    mixed_train = train_a + train_b
    random.shuffle(mixed_train)

    mixed_test = test_a + test_b

    return mixed_train, mixed_test
