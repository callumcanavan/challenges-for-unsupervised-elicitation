#!/usr/bin/env python3
"""Run a prompting method on a dataset using vLLM.

Usage:
    python scripts/run_prompting.py --method zero_shot --dataset gsm8k_preference
    python scripts/run_prompting.py --method random_few_shot --dataset gsm8k_preference --shots 8
    python scripts/run_prompting.py --method bootstrap --dataset ctrl_z_10steps --shots 2,4,8
    python scripts/run_prompting.py --method golden_few_shot --dataset gsm_political --shots 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATASET_CONFIGS,
    DEFAULT_BOOTSTRAP_POOL_SIZE,
    get_dataset_config,
    get_prompt_format,
)
from src.data import (
    build_mixed_dataset,
    load_dataset,
    prepare_items,
    subsample_by_true_proportion,
)
from src.logging_utils import add_file_handler, get_logger
from src.metrics import compute_accuracy, compute_auroc
from src.prompting import PromptingResult, run_bootstrap, run_few_shot, run_zero_shot
from src.storage_utils import (
    PROMPTING_RESULTS_DIR,
    backup_dir_to_cloud,
    get_hydra_filename,
    write_config_json,
)
from src.utils import seed_all, write_csv_header, write_csv_rows
from src.vllm_inference import cleanup_vllm

logger = get_logger(__name__)

PROMPTING_METHODS = [
    "zero_shot",
    "random_few_shot",
    "golden_few_shot",
    "bootstrap",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run prompting method")
    p.add_argument("--method", required=True, choices=PROMPTING_METHODS)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--prompt-format", type=str, default=None)
    p.add_argument("--shots", type=str, default="8",
                   help="Number of shots (int). For bootstrap, comma-separated "
                        "sequence, e.g. '2,4,8'.")
    p.add_argument("--train-true-proportion", type=float, default=None)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--bootstrap-pool-size", type=int,
                   default=DEFAULT_BOOTSTRAP_POOL_SIZE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--skip-if-exists", action="store_true", default=True)
    p.add_argument("--no-skip-if-exists", action="store_false",
                   dest="skip_if_exists")
    # vLLM args
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--no-backup", action="store_true",
                   help="Skip per-experiment GCS backup (use tools/backup.py after).")
    p.add_argument("--debug", action="store_true",
                   help="Generate random scores instead of running vLLM. "
                        "Writes to results/debug/ by default.")
    return p.parse_args()


def _parse_shot_sequence(shots_str: str) -> list[int]:
    """Parse a comma-separated shot sequence string."""
    return [int(s.strip()) for s in shots_str.split(",")]


def _build_experiment_config(args: argparse.Namespace) -> dict:
    """Build config dict for hydra filename (non-default values only)."""
    config: dict = {
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed,
    }
    if args.model is not None:
        config["model"] = args.model.split("/")[-1]
    if args.method != "zero_shot":
        config["shots"] = args.shots
    if args.train_true_proportion is not None:
        config["ttp"] = args.train_true_proportion
    if args.train_size is not None:
        config["train_size"] = args.train_size
    if args.method == "bootstrap":
        config["pool"] = args.bootstrap_pool_size
    return config


def _save_results(
    output_dir: Path,
    full_config: dict,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    true_logprobs: np.ndarray,
    false_logprobs: np.ndarray,
    test_sources: list[str] | None = None,
    iteration_metrics: list[dict] | None = None,
    backup: bool = True,
) -> None:
    """Save predictions CSV, results JSON, and optional iteration metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config_json(output_dir, full_config)

    # Predictions CSV
    predictions_path = output_dir / "predictions.csv"
    fieldnames = [
        "index", "score", "predicted_label", "gt_label",
        "true_logprob", "false_logprob",
    ]
    if test_sources is not None:
        fieldnames.append("dataset_source")

    write_csv_header(predictions_path, fieldnames)
    rows = []
    for i in range(len(test_scores)):
        row = {
            "index": i,
            "score": float(test_scores[i]),
            "predicted_label": bool(test_scores[i] > 0),
            "gt_label": bool(test_labels[i]),
            "true_logprob": float(true_logprobs[i]),
            "false_logprob": float(false_logprobs[i]),
        }
        if test_sources is not None:
            row["dataset_source"] = test_sources[i]
        rows.append(row)
    write_csv_rows(predictions_path, rows)

    # Summary results
    test_auroc = compute_auroc(test_scores, test_labels)
    test_acc = compute_accuracy(test_scores, test_labels)

    results = {
        "test_auroc": test_auroc,
        "test_accuracy": test_acc,
        "test_size": len(test_scores),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Bootstrap iteration metrics
    if iteration_metrics:
        iter_path = output_dir / "iteration_metrics.csv"
        iter_fields = list(iteration_metrics[0].keys())
        with open(iter_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=iter_fields)
            writer.writeheader()
            writer.writerows(iteration_metrics)

    logger.info("Test AUROC: %.4f | Test Acc: %.4f", test_auroc, test_acc)
    if backup:
        backup_dir_to_cloud(output_dir)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()

    seed_all(args.seed)

    # Resolve dataset and model
    is_mixed = args.dataset == "gsm_political"
    if is_mixed:
        config_main = get_dataset_config("gsm8k_preference")
    else:
        config_main = get_dataset_config(args.dataset)

    model_name = args.model or config_main.default_model

    # Resolve prompt format
    if args.prompt_format:
        prompt_format = get_prompt_format(args.prompt_format)
    else:
        prompt_format = config_main.prompt_format

    results_dir = Path(args.results_dir) if args.results_dir else PROMPTING_RESULTS_DIR
    if args.debug and args.results_dir is None:
        results_dir = Path("results/debug/prompting")

    # Build output path
    exp_config = _build_experiment_config(args)
    output_dir = results_dir / get_hydra_filename(exp_config)

    if args.skip_if_exists and (output_dir / "results.json").exists():
        logger.info("Results exist at %s, skipping.", output_dir)
        return 0

    # Log to output dir (add to root logger so src.* loggers are also captured)
    output_dir.mkdir(parents=True, exist_ok=True)
    import logging as _logging
    root_logger = _logging.getLogger()
    root_logger.setLevel(_logging.INFO)
    add_file_handler(root_logger, output_dir / "run.log")

    # vLLM model kwargs
    model_kwargs: dict = {}
    if args.tensor_parallel_size is not None:
        model_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization != 0.9:
        model_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        model_kwargs["max_model_len"] = args.max_model_len

    # Load data
    if is_mixed:
        from src.config import DEFAULT_MIXED_DATASET_SIZE
        train_size = args.train_size or DEFAULT_MIXED_DATASET_SIZE
        train_pool, test_items = build_mixed_dataset(
            "gsm8k_preference", "political_normative",
            train_size=train_size,
        )
        test_sources = [item["dataset_source"] for item in test_items]
    else:
        train_data = load_dataset(args.dataset, "train")
        train_pool = prepare_items(train_data, config_main)
        test_data = load_dataset(args.dataset, "test")
        test_items = prepare_items(test_data, config_main)
        test_sources = None

    # Subsample training pool by true proportion if requested
    if args.train_true_proportion is not None:
        train_pool = subsample_by_true_proportion(
            train_pool, args.train_true_proportion,
            max_size=args.train_size,
        )
    elif args.train_size is not None:
        import random as rng_module
        rng = rng_module.Random(args.seed)
        rng.shuffle(train_pool)
        train_pool = train_pool[:args.train_size]

    test_labels = np.array([item["label"] for item in test_items])

    logger.info(
        "Dataset: %s | Method: %s | Train pool: %d | Test: %d",
        args.dataset, args.method, len(train_pool), len(test_items),
    )

    # Run prompting method
    if args.method == "zero_shot":
        result = run_zero_shot(
            test_items, prompt_format, model_name,
            debug=args.debug, **model_kwargs,
        )

    elif args.method == "random_few_shot":
        n_shots = int(args.shots)
        result = run_few_shot(
            test_items, train_pool, n_shots, prompt_format, model_name,
            labels_source="random", seed=args.seed,
            debug=args.debug, **model_kwargs,
        )

    elif args.method == "golden_few_shot":
        n_shots = int(args.shots)
        result = run_few_shot(
            test_items, train_pool, n_shots, prompt_format, model_name,
            labels_source="golden", seed=args.seed,
            debug=args.debug, **model_kwargs,
        )

    elif args.method == "bootstrap":
        shot_sequence = _parse_shot_sequence(args.shots)
        result = run_bootstrap(
            test_items, train_pool, shot_sequence, prompt_format, model_name,
            bootstrap_pool_size=args.bootstrap_pool_size,
            seed=args.seed, debug=args.debug, **model_kwargs,
        )

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save results
    full_config = {
        "method": args.method,
        "dataset": args.dataset,
        "model": model_name,
        "prompt_format": args.prompt_format or config_main.prompt_format_key,
        "shots": args.shots if args.method != "zero_shot" else None,
        "seed": args.seed,
        "train_true_proportion": args.train_true_proportion,
        "train_pool_size": len(train_pool),
        "test_size": len(test_items),
        "bootstrap_pool_size": (
            args.bootstrap_pool_size if args.method == "bootstrap" else None
        ),
    }
    _save_results(
        output_dir, full_config,
        result.scores, test_labels,
        result.true_logprobs, result.false_logprobs,
        test_sources=test_sources,
        iteration_metrics=result.iteration_metrics,
        backup=not args.no_backup,
    )

    return 0


def args_from_config(config: dict, fixed: dict) -> argparse.Namespace:
    """Build an argparse.Namespace from a sweep config dict.

    Required by experiment_utils.run_sweep for --inprocess execution.
    """
    merged = {**config, **fixed}
    # Map sweep keys (with dashes) to argparse attrs (with underscores)
    args = argparse.Namespace(
        method=merged.get("method", "zero_shot"),
        dataset=merged.get("dataset"),
        model=merged.get("model"),
        prompt_format=merged.get("prompt-format"),
        shots=str(merged.get("shots", "8")),
        train_true_proportion=merged.get("train-true-proportion"),
        train_size=merged.get("train-size"),
        bootstrap_pool_size=int(merged.get("bootstrap-pool-size", DEFAULT_BOOTSTRAP_POOL_SIZE)),
        seed=int(merged.get("seed", 0)),
        results_dir=merged.get("results-dir"),
        skip_if_exists="no-skip-if-exists" not in merged,
        no_backup="no-backup" in merged,
        tensor_parallel_size=merged.get("tensor-parallel-size"),
        gpu_memory_utilization=float(merged.get("gpu-memory-utilization", 0.9)),
        max_model_len=merged.get("max-model-len"),
        debug="debug" in merged,
    )
    # Convert numeric types
    if args.train_true_proportion is not None:
        args.train_true_proportion = float(args.train_true_proportion)
    if args.train_size is not None:
        args.train_size = int(args.train_size)
    if args.tensor_parallel_size is not None:
        args.tensor_parallel_size = int(args.tensor_parallel_size)
    if args.max_model_len is not None:
        args.max_model_len = int(args.max_model_len)
    return args


if __name__ == "__main__":
    sys.exit(main())
