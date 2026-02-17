#!/usr/bin/env python3
"""Extract contrastive activations and logprobs for all layers of a model.

Usage:
    python scripts/get_activations.py --dataset gsm8k_preference --split both
    python scripts/get_activations.py --dataset ctrl_z_10steps --model meta-llama/Llama-3.1-70B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activations import extract_contrastive_activations, load_model
from src.config import (
    DATASET_CONFIGS,
    get_dataset_config,
    get_num_layers,
)
from src.data import load_dataset, prepare_items
from src.logging_utils import get_logger
from src.storage_utils import (
    ACTIVATIONS_DIR,
    backup_dir_to_cloud,
    cache_to_disk,
    get_hydra_filename,
    write_config_json,
)
from src.utils import seed_all

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract contrastive activations")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (default: from dataset config)")
    parser.add_argument("--prompt-format", type=str, default=None, help="Prompt format key (default: from config)")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-if-exists", action="store_true", default=True)
    parser.add_argument("--no-skip-if-exists", action="store_false", dest="skip_if_exists")
    parser.add_argument("--debug", action="store_true",
                        help="Generate random activations instead of running model. "
                             "Writes to results/debug/ by default.")
    return parser.parse_args()


def extract_and_save(
    model,
    tokenizer,
    dataset_name: str,
    split: str,
    prompt_format,
    config,
    model_name: str,
    layers: list[int],
    batch_size: int,
    output_dir: Path,
    skip_if_exists: bool,
    seed: int,
) -> None:
    """Extract activations for a single split and save to disk."""
    # Build output path
    exp_config = {
        "model": model_name.split("/")[-1],
        "dataset": dataset_name,
        "split": split,
        "format": config.prompt_format_key,
    }
    out_folder = output_dir / get_hydra_filename(exp_config)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Check if already done
    data_path = out_folder / "data.json"
    if skip_if_exists:
        all_exist = True
        for l in layers:
            layer_path = out_folder / f"layer_{l:02d}.pt"
            cache_to_disk(layer_path)
            if not layer_path.exists():
                all_exist = False
        cache_to_disk(data_path)
        if all_exist and data_path.exists():
            logger.info("All layers cached for %s/%s, skipping.", dataset_name, split)
            return

    # Load and prepare data
    data = load_dataset(dataset_name, split)
    items = prepare_items(data, config)
    logger.info("Loaded %d items for %s/%s", len(items), dataset_name, split)

    # Save data metadata
    with open(data_path, "w") as f:
        json.dump(items, f, indent=2, default=str)

    # Extract activations
    prompt_fmt = prompt_format or config.prompt_format
    result = extract_contrastive_activations(
        model, tokenizer, items, prompt_fmt, layers, batch_size
    )

    # Save per-layer activations
    for layer_idx, (X_pos, X_neg) in result.activations.items():
        layer_path = out_folder / f"layer_{layer_idx:02d}.pt"
        torch.save({"X_pos": X_pos, "X_neg": X_neg}, layer_path)

    # Save logprobs
    torch.save(
        {
            "true_logprobs": result.true_logprobs,
            "false_logprobs": result.false_logprobs,
            "logprob_scores": result.logprob_scores,
        },
        out_folder / "logprobs.pt",
    )

    # Save config
    write_config_json(out_folder, {
        **exp_config,
        "num_items": len(items),
        "layers": layers,
        "batch_size": batch_size,
        "seed": seed,
    })

    backup_dir_to_cloud(out_folder)
    logger.info("Saved activations for %d layers to %s", len(layers), out_folder)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()

    seed_all(args.seed)
    config = get_dataset_config(args.dataset)

    model_name = args.model or config.default_model
    num_layers = get_num_layers(model_name)
    layers = list(range(num_layers))

    output_dir = Path(args.output_dir) if args.output_dir else ACTIVATIONS_DIR
    if args.debug and args.output_dir is None:
        output_dir = Path("results/debug/activations")

    if args.debug:
        _generate_debug_activations(
            args.dataset, model_name, config, layers, output_dir, args.split,
        )
        return 0

    logger.info("Loading model: %s", model_name)
    model, tokenizer = load_model(model_name)

    prompt_format = None
    if args.prompt_format:
        from src.config import get_prompt_format
        prompt_format = get_prompt_format(args.prompt_format)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        extract_and_save(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            split=split,
            prompt_format=prompt_format,
            config=config,
            model_name=model_name,
            layers=layers,
            batch_size=args.batch_size,
            output_dir=output_dir,
            skip_if_exists=args.skip_if_exists,
            seed=args.seed,
        )

    return 0


def _generate_debug_activations(
    dataset_name: str,
    model_name: str,
    config,
    layers: list[int],
    output_dir: Path,
    split_arg: str,
) -> None:
    """Generate random activations for debug/testing."""
    from src.data import load_dataset, prepare_items

    dim = 256
    splits = ["train", "test"] if split_arg == "both" else [split_arg]

    for split in splits:
        data = load_dataset(dataset_name, split)
        items = prepare_items(data, config)
        n = len(items)

        exp_config = {
            "model": model_name.split("/")[-1],
            "dataset": dataset_name,
            "split": split,
            "format": config.prompt_format_key,
        }
        out_folder = output_dir / get_hydra_filename(exp_config)
        out_folder.mkdir(parents=True, exist_ok=True)

        for layer_idx in layers:
            X_pos = torch.randn(n, dim)
            X_neg = torch.randn(n, dim)
            torch.save({"X_pos": X_pos, "X_neg": X_neg},
                        out_folder / f"layer_{layer_idx:02d}.pt")

        # Save logprobs
        true_lp = torch.randn(n)
        false_lp = torch.randn(n)
        torch.save({
            "true_logprobs": true_lp,
            "false_logprobs": false_lp,
            "logprob_scores": true_lp - false_lp,
        }, out_folder / "logprobs.pt")

        write_config_json(out_folder, {
            **exp_config,
            "num_items": n,
            "layers": layers,
            "debug": True,
        })
        logger.info("Generated debug activations for %s/%s (%d items, %d layers)",
                     dataset_name, split, n, len(layers))


if __name__ == "__main__":
    sys.exit(main())
