"""Tests for src/prompting.py (prompt building logic, not vLLM inference)."""

import numpy as np
import pytest

from src.config import PROMPT_FORMATS
from src.prompting import PromptingResult


class TestPromptingResult:
    def test_fields(self):
        result = PromptingResult(
            scores=np.array([1.0, -1.0]),
            true_logprobs=np.array([-0.5, -1.5]),
            false_logprobs=np.array([-1.5, -0.5]),
            labels=np.array([True, False]),
        )
        assert len(result.scores) == 2
        assert result.iteration_metrics is None


class TestFewShotPromptBuilding:
    """Test that few-shot prompts are constructed correctly."""

    def test_icm_few_shot_structure(self, icm_items):
        fmt = PROMPT_FORMATS["icm_binary"]
        examples = icm_items[:2]
        labels = [True, False]
        target = icm_items[4]

        prompt = fmt.format_few_shot_prompt(examples, labels, target)

        # Should have examples separated by double newline
        parts = prompt.split("\n\n")
        assert len(parts) >= 3  # at least 2 examples + target

        # Last part should be the target
        assert parts[-1].endswith("I think this claim is")

        # First example should end with True
        assert " True" in parts[0]

    def test_ctrl_z_few_shot_structure(self, ctrl_z_items):
        fmt = PROMPT_FORMATS["ctrl_z"]
        examples = [ctrl_z_items[0]]
        labels = [True]
        target = ctrl_z_items[1]

        prompt = fmt.format_few_shot_prompt(examples, labels, target)

        # Should start with system prompt (prefix)
        assert prompt.startswith("<|begin_of_text|>")

        # Should have exactly one system prompt
        count = prompt.count("<|begin_of_text|>")
        assert count == 1

        # Should end with assistant header
        assert prompt.rstrip().endswith(
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ) or prompt.rstrip().endswith("suspicious?")

    def test_golden_labels_with_none(self, icm_items):
        fmt = PROMPT_FORMATS["icm_binary"]
        examples = icm_items[:2]
        labels = [True, None]  # None for indeterminate
        target = icm_items[4]

        prompt = fmt.format_few_shot_prompt(examples, labels, target)

        # Should contain the true label for first example
        assert " True" in prompt

    def test_zero_shot_has_no_examples(self, icm_items):
        fmt = PROMPT_FORMATS["icm_binary"]
        prompt = fmt.format_few_shot_prompt([], [], icm_items[0])
        # Should just be the prompt for the target
        assert prompt == fmt.format_prompt(icm_items[0])
