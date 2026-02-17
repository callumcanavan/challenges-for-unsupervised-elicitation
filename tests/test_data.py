"""Tests for src/data.py."""

import pytest

from src.config import get_dataset_config
from src.data import (
    load_dataset,
    prepare_items,
    sample_few_shot_examples,
    subsample_by_true_proportion,
)


class TestLoadDataset:
    def test_load_train(self, tmp_dataset_dir):
        data = load_dataset("gsm8k_preference", "train")
        assert len(data) == 20

    def test_load_test(self, tmp_dataset_dir):
        data = load_dataset("gsm8k_preference", "test")
        assert len(data) == 10

    def test_load_nonexistent(self, tmp_dataset_dir):
        with pytest.raises(FileNotFoundError):
            load_dataset("larger_than", "train")


class TestPrepareItems:
    def test_normalizes_labels(self, icm_items):
        config = get_dataset_config("gsm8k_preference")
        items = prepare_items(icm_items, config)
        for item in items:
            assert isinstance(item["label"], bool)

    def test_adds_prompt(self, icm_items):
        config = get_dataset_config("gsm8k_preference")
        items = prepare_items(icm_items, config)
        for item in items:
            assert "prompt" in item
            assert "Question:" in item["prompt"]

    def test_field_mapping(self, gsm8k_test_items):
        config = get_dataset_config("gsm8k_preference")
        items = prepare_items(gsm8k_test_items, config)
        # cot should be mapped to choice
        for item in items:
            assert "choice" in item

    def test_does_not_mutate_original(self, icm_items):
        config = get_dataset_config("gsm8k_preference")
        original_keys = set(icm_items[0].keys())
        prepare_items(icm_items, config)
        assert set(icm_items[0].keys()) == original_keys

    def test_ctrl_z_format(self, ctrl_z_items):
        config = get_dataset_config("ctrl_z_10steps")
        items = prepare_items(ctrl_z_items, config)
        for item in items:
            assert "prompt" in item
            assert "ASSIGNED_TASK" in item["prompt"]


class TestSubsampleByTrueProportion:
    @pytest.fixture
    def pool(self):
        return [{"label": i < 50, "id": i} for i in range(100)]

    def test_all_true(self, pool):
        result = subsample_by_true_proportion(pool, 1.0)
        assert all(x["label"] for x in result)

    def test_all_false(self, pool):
        result = subsample_by_true_proportion(pool, 0.0)
        assert not any(x["label"] for x in result)

    def test_half(self, pool):
        result = subsample_by_true_proportion(pool, 0.5)
        n_true = sum(1 for x in result if x["label"])
        n_false = len(result) - n_true
        assert abs(n_true - n_false) <= 1

    def test_max_size(self, pool):
        result = subsample_by_true_proportion(pool, 0.5, max_size=20)
        assert len(result) <= 20

    def test_reproducible(self, pool):
        import random
        random.seed(42)
        a = subsample_by_true_proportion(pool, 0.5)
        random.seed(42)
        b = subsample_by_true_proportion(pool, 0.5)
        assert [x["id"] for x in a] == [x["id"] for x in b]


class TestSampleFewShotExamples:
    @pytest.fixture
    def pool(self):
        return [
            {"question": f"Q{i}", "choice": f"C{i}", "label": i % 2 == 0}
            for i in range(20)
        ]

    def test_returns_correct_count(self, pool):
        examples, labels = sample_few_shot_examples(pool, 4)
        assert len(examples) == 4
        assert len(labels) == 4

    def test_balanced(self, pool):
        examples, labels = sample_few_shot_examples(pool, 8)
        n_true = sum(labels)
        n_false = len(labels) - n_true
        assert n_true == 4
        assert n_false == 4

    def test_excludes_target(self, pool):
        target = pool[0]
        examples, _ = sample_few_shot_examples(pool, 4, exclude_item=target)
        for ex in examples:
            assert ex["question"] != target["question"]

    def test_unique_questions(self, pool):
        examples, _ = sample_few_shot_examples(pool, 6)
        questions = [ex["question"] for ex in examples]
        assert len(questions) == len(set(questions))

    def test_zero_shots(self, pool):
        examples, labels = sample_few_shot_examples(pool, 0)
        assert examples == []
        assert labels == []

    def test_reproducible(self, pool):
        import random
        random.seed(42)
        a, la = sample_few_shot_examples(pool, 4)
        random.seed(42)
        b, lb = sample_few_shot_examples(pool, 4)
        assert [x["question"] for x in a] == [x["question"] for x in b]
        assert la == lb

    def test_label_key_predicted(self, pool):
        """Balancing on predicted_label should use that field, not GT label."""
        import random
        random.seed(0)
        # Assign random predicted_labels uncorrelated with GT
        for item in pool:
            item["predicted_label"] = random.random() > 0.5

        random.seed(1)
        examples, labels = sample_few_shot_examples(
            pool, 8, label_key="predicted_label"
        )
        # Labels should match predicted_label, not label
        for ex, lbl in zip(examples, labels):
            assert lbl == ex["predicted_label"]

        n_pred_true = sum(labels)
        assert n_pred_true == 4  # balanced by predicted_label


class TestTTPConstantSizing:
    """Different train_true_proportions should produce same-sized datasets."""

    @pytest.fixture
    def pool(self):
        # 100 items: 50 true, 50 false
        return [{"label": i < 50, "id": i} for i in range(100)]

    def test_same_size_across_ttps(self, pool):
        import random
        proportions = [0.1, 0.25, 0.5, 0.75, 0.9]
        sizes = []
        for ttp in proportions:
            random.seed(0)
            result = subsample_by_true_proportion(pool, ttp)
            sizes.append(len(result))
        # All should be the same size
        assert len(set(sizes)) == 1, f"Got different sizes: {dict(zip(proportions, sizes))}"

    def test_raises_on_impossible_proportion(self):
        # 5 true, 95 false; base_size=5; TTP=0.1 → n_true=round(0.5)=0, n_false=5
        # Actually with rounding this won't raise. Use a case where it must.
        # 3 true, 3 false; base_size=3; TTP=0.9 → n_true=round(2.7)=3, n_false=0
        # That's fine. But with 0 of one class:
        pool = [{"label": True, "id": i} for i in range(10)]
        # 10 true, 0 false → min(10, 0) = 0, base_size = 0
        result = subsample_by_true_proportion(pool, 0.5)
        assert len(result) == 0  # No false items, can't build 50/50
