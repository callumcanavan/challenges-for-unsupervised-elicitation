"""Tests for src/config.py."""

import pytest

from src.config import (
    DATASET_CONFIGS,
    DEFAULT_LAYERS,
    PROMPT_FORMATS,
    DatasetConfig,
    PromptFormat,
    get_dataset_config,
    get_default_layer,
    get_prompt_format,
)


class TestPromptFormat:
    def test_format_prompt_icm(self, icm_items):
        fmt = PROMPT_FORMATS["icm_binary"]
        prompt = fmt.format_prompt(icm_items[0])
        assert "Question: What is 2+2?" in prompt
        assert "Claim: 4" in prompt
        assert prompt.endswith("I think this claim is")

    def test_format_contrastive_prompts(self, icm_items):
        fmt = PROMPT_FORMATS["icm_binary"]
        pos, neg = fmt.format_contrastive_prompts(icm_items[0])
        assert pos.endswith(" True")
        assert neg.endswith(" False")
        # Both should share the same base prompt
        assert pos[:-5] == neg[:-6]

    def test_format_contrastive_prompts_ctrl_z(self, ctrl_z_items):
        fmt = PROMPT_FORMATS["ctrl_z"]
        pos, neg = fmt.format_contrastive_prompts(ctrl_z_items[0])
        assert pos.endswith("Yes")
        assert neg.endswith("No")
        assert "<|begin_of_text|>" in pos  # prefix included

    def test_format_few_shot_prompt_icm(self, icm_items):
        fmt = PROMPT_FORMATS["icm_binary"]
        examples = icm_items[:2]
        labels = [True, False]
        target = icm_items[2]

        prompt = fmt.format_few_shot_prompt(examples, labels, target)

        # Should contain both examples with labels
        assert " True" in prompt
        assert " False" in prompt
        # Should contain the target without a label
        assert "What is 3+3?" in prompt
        # Target should be the last part
        assert prompt.endswith("I think this claim is")

    def test_format_few_shot_prompt_with_prefix(self, ctrl_z_items):
        fmt = PROMPT_FORMATS["ctrl_z"]
        examples = ctrl_z_items[:1]
        labels = [True]
        target = ctrl_z_items[1]

        prompt = fmt.format_few_shot_prompt(examples, labels, target)
        # Prefix should appear exactly once at the start
        assert prompt.startswith("<|begin_of_text|>")
        # Separator between examples
        assert "<|eot_id|>" in prompt

    def test_format_few_shot_prompt_empty(self, icm_items):
        fmt = PROMPT_FORMATS["icm_binary"]
        prompt = fmt.format_few_shot_prompt([], [], icm_items[0])
        expected = fmt.format_prompt(icm_items[0])
        assert prompt == expected

    def test_format_few_shot_indeterminate(self, icm_items):
        fmt = PromptFormat(
            template="Q: {question}\nC: {choice}\nAnswer:",
            true_token=" True",
            false_token=" False",
            indeterminate_token=" Unknown",
        )
        prompt = fmt.format_few_shot_prompt(
            icm_items[:1], [None], icm_items[1]
        )
        assert " Unknown" in prompt

    def test_get_label_token(self):
        fmt = PROMPT_FORMATS["icm_binary"]
        assert fmt.get_label_token(True) == " True"
        assert fmt.get_label_token(False) == " False"


class TestDatasetConfig:
    def test_all_configs_have_prompt_format(self):
        for name, config in DATASET_CONFIGS.items():
            assert config.prompt_format_key in PROMPT_FORMATS, (
                f"Dataset {name} references unknown prompt format {config.prompt_format_key}"
            )

    def test_prompt_format_property(self):
        config = DATASET_CONFIGS["gsm8k_preference"]
        assert isinstance(config.prompt_format, PromptFormat)

    def test_get_dataset_config(self):
        config = get_dataset_config("gsm8k_preference")
        assert config.name == "gsm8k_preference"

    def test_get_dataset_config_unknown(self):
        with pytest.raises(KeyError):
            get_dataset_config("nonexistent")

    def test_field_mapping_gsm8k(self):
        config = DATASET_CONFIGS["gsm8k_preference"]
        assert "cot" in config.field_mapping
        assert config.field_mapping["cot"] == "choice"


class TestModelDefaults:
    def test_default_layers_exist(self):
        assert "meta-llama/Llama-3.1-8B" in DEFAULT_LAYERS
        assert "meta-llama/Llama-3.1-70B" in DEFAULT_LAYERS

    def test_get_default_layer(self):
        assert get_default_layer("meta-llama/Llama-3.1-8B") == 18
        assert get_default_layer("meta-llama/Llama-3.1-70B") == 36

    def test_get_default_layer_short_name(self):
        assert get_default_layer("Llama-3.1-8B") == 18

    def test_get_default_layer_unknown(self):
        with pytest.raises(KeyError):
            get_default_layer("unknown-model")

    def test_get_prompt_format(self):
        fmt = get_prompt_format("icm_binary")
        assert isinstance(fmt, PromptFormat)

    def test_get_prompt_format_unknown(self):
        with pytest.raises(KeyError):
            get_prompt_format("nonexistent")
