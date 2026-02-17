#!/usr/bin/env python3
"""Run a probing method on pre-extracted activations.

Usage:
    python scripts/run_probing.py --method ccs --dataset gsm8k_preference
    python scripts/run_probing.py --method eh --dataset gsm8k_preference --easy-dataset larger_than
    python scripts/run_probing.py --method supervised --dataset gsm8k_preference --train-true-proportion 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activations import normalize_activations
from src.config import (
    DATASET_CONFIGS,
    DEFAULT_CCS_EPOCHS,
    DEFAULT_CCS_LR,
    DEFAULT_CCS_RETRIES,
    DEFAULT_CCS_WEIGHT_DECAY,
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_ENSEMBLE_TEMPERATURE,
    DEFAULT_SUPERVISED_MAX_EPOCHS,
    DEFAULT_SUPERVISED_PATIENCE,
    DEFAULT_UEEH_ALPHA,
    get_dataset_config,
    get_default_layer,
)
from src.data import (
    build_mixed_dataset,
    load_dataset,
    prepare_items,
    subsample_by_true_proportion,
)
from src.logging_utils import add_file_handler, get_logger
from src.metrics import compute_accuracy, compute_auroc
from src.probing import (
    apply_flip,
    compute_flip_direction,
    init_random_probe,
    score_with_ensemble,
    score_with_pca,
    score_with_pca_ensemble,
    score_with_probe,
    train_ccs,
    train_eh,
    train_pca,
    train_pca_ensemble,
    train_random_ensemble,
    train_supervised,
    train_ueeh,
)
from src.storage_utils import (
    ACTIVATIONS_DIR,
    PROBING_RESULTS_DIR,
    backup_dir_to_cloud,
    cache_dir_to_disk,
    get_hydra_filename,
    write_config_json,
)
from src.utils import seed_all, write_csv_header, write_csv_rows

logger = get_logger(__name__)

PROBING_METHODS = [
    "ccs", "pca", "supervised", "random", "eh", "ueeh",
    "random_ensemble", "pca_ensemble",
    "random_ensemble_eh", "pca_ensemble_eh",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run probing method")
    p.add_argument("--method", required=True, choices=PROBING_METHODS)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--activation-dir", type=str, default=None)
    p.add_argument("--train-true-proportion", type=float, default=None)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--easy-dataset", type=str, default="larger_than")
    p.add_argument("--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE)
    p.add_argument("--ensemble-temperature", type=float, default=DEFAULT_ENSEMBLE_TEMPERATURE)
    p.add_argument("--scale-by-variance", action="store_true")
    p.add_argument("--ueeh-alpha", type=float, default=DEFAULT_UEEH_ALPHA)
    p.add_argument("--ueeh-mode", type=str, default="paired", choices=["paired", "alternate"])
    p.add_argument("--use-soft-labels", action="store_true",
                   help="Use 0.5 targets for normative items in supervised probing")
    p.add_argument("--ccs-lr", type=float, default=DEFAULT_CCS_LR)
    p.add_argument("--ccs-epochs", type=int, default=DEFAULT_CCS_EPOCHS)
    p.add_argument("--ccs-retries", type=int, default=DEFAULT_CCS_RETRIES)
    p.add_argument("--ccs-weight-decay", type=float, default=DEFAULT_CCS_WEIGHT_DECAY)
    p.add_argument("--supervised-lr", type=float, default=0.03)
    p.add_argument("--supervised-patience", type=int, default=DEFAULT_SUPERVISED_PATIENCE)
    p.add_argument("--supervised-max-epochs", type=int, default=DEFAULT_SUPERVISED_MAX_EPOCHS)
    p.add_argument("--supervised-weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--skip-if-exists", action="store_true", default=True)
    p.add_argument("--no-skip-if-exists", action="store_false", dest="skip_if_exists")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip per-experiment GCS backup (use tools/backup.py after).")
    p.add_argument("--debug", action="store_true",
                   help="Use results/debug/ directories by default.")
    return p.parse_args()


def _load_activations(
    activation_dir: Path,
    dataset_name: str,
    split: str,
    model_name: str,
    prompt_format_key: str,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load pre-extracted activations for a specific layer."""
    exp_config = {
        "model": model_name.split("/")[-1],
        "dataset": dataset_name,
        "split": split,
        "format": prompt_format_key,
    }
    folder = activation_dir / get_hydra_filename(exp_config)
    layer_path = folder / f"layer_{layer:02d}.pt"

    # Only hit GCS if the file isn't already local
    if not layer_path.exists():
        cache_dir_to_disk(folder)

    if not layer_path.exists():
        raise FileNotFoundError(
            f"Activation file not found: {layer_path}\n"
            f"Run scripts/get_activations.py first."
        )

    data = torch.load(layer_path, weights_only=True)
    return data["X_pos"].float(), data["X_neg"].float()


def _load_labels(dataset_name: str, split: str) -> np.ndarray:
    """Load ground-truth labels for a dataset split."""
    config = get_dataset_config(dataset_name)
    data = load_dataset(dataset_name, split)
    items = prepare_items(data, config)
    return np.array([item["label"] for item in items])


def _build_experiment_config(args: argparse.Namespace) -> dict:
    """Build config dict for hydra filename (non-default values only)."""
    config: dict = {
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed,
    }
    if args.model is not None:
        config["model"] = args.model.split("/")[-1]
    if args.layer is not None:
        config["layer"] = args.layer
    if args.train_true_proportion is not None:
        config["ttp"] = args.train_true_proportion
    if args.train_size is not None:
        config["train_size"] = args.train_size
    if args.method in ("eh", "ueeh", "random_ensemble_eh", "pca_ensemble_eh"):
        config["easy"] = args.easy_dataset
    if "ensemble" in args.method:
        config["es"] = args.ensemble_size
    if args.method == "ueeh":
        config["alpha"] = args.ueeh_alpha
    return config


def _save_results(
    output_dir: Path,
    full_config: dict,
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_sources: list[str] | None = None,
    backup: bool = True,
) -> None:
    """Save predictions CSV and results JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config_json(output_dir, full_config)

    # Predictions CSV
    predictions_path = output_dir / "predictions.csv"
    fieldnames = ["index", "score", "predicted_label", "gt_label"]
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
        }
        if test_sources is not None:
            row["dataset_source"] = test_sources[i]
        rows.append(row)
    write_csv_rows(predictions_path, rows)

    # Summary results
    test_auroc = compute_auroc(test_scores, test_labels)
    test_acc = compute_accuracy(test_scores, test_labels)
    train_auroc = compute_auroc(train_scores, train_labels)
    train_acc = compute_accuracy(train_scores, train_labels)

    results = {
        "test_auroc": test_auroc,
        "test_accuracy": test_acc,
        "train_auroc": train_auroc,
        "train_accuracy": train_acc,
        "test_size": len(test_scores),
        "train_size": len(train_scores),
        "train_true_proportion": float(np.mean(train_labels)),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Test AUROC: %.4f | Test Acc: %.4f", test_auroc, test_acc)
    if backup:
        backup_dir_to_cloud(output_dir)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()

    seed_all(args.seed)

    # Resolve config
    is_mixed = args.dataset == "gsm_political"
    if is_mixed:
        config_main = get_dataset_config("gsm8k_preference")
    else:
        config_main = get_dataset_config(args.dataset)

    model_name = args.model or config_main.default_model
    layer = args.layer if args.layer is not None else get_default_layer(model_name)
    activation_dir = Path(args.activation_dir) if args.activation_dir else ACTIVATIONS_DIR
    results_dir = Path(args.results_dir) if args.results_dir else PROBING_RESULTS_DIR
    if args.debug:
        if args.activation_dir is None:
            activation_dir = Path("results/debug/activations")
        if args.results_dir is None:
            results_dir = Path("results/debug/probing")
        # Reduce training epochs for fast debug runs
        args.ccs_epochs = min(args.ccs_epochs, 10)
        args.ccs_retries = min(args.ccs_retries, 2)
        args.supervised_max_epochs = min(args.supervised_max_epochs, 100)
        args.supervised_patience = min(args.supervised_patience, 10)
        args.ensemble_size = min(args.ensemble_size, 4)

    # Build output path
    exp_config = _build_experiment_config(args)
    exp_config["layer"] = layer
    output_dir = results_dir / get_hydra_filename(exp_config)

    if args.skip_if_exists and (output_dir / "results.json").exists():
        logger.info("Results exist at %s, skipping.", output_dir)
        return 0

    # Log to output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(logger, output_dir / "run.log")

    # Load activations
    if is_mixed:
        config_pol = get_dataset_config("political_normative")
        X_pos_train_g, X_neg_train_g = _load_activations(
            activation_dir, "gsm8k_preference", "train", model_name,
            config_main.prompt_format_key, layer,
        )
        X_pos_train_p, X_neg_train_p = _load_activations(
            activation_dir, "political_normative", "train", model_name,
            config_pol.prompt_format_key, layer,
        )
        X_pos_test_g, X_neg_test_g = _load_activations(
            activation_dir, "gsm8k_preference", "test", model_name,
            config_main.prompt_format_key, layer,
        )
        X_pos_test_p, X_neg_test_p = _load_activations(
            activation_dir, "political_normative", "test", model_name,
            config_pol.prompt_format_key, layer,
        )

        y_train_g = _load_labels("gsm8k_preference", "train")
        y_train_p = _load_labels("political_normative", "train")
        y_test_g = _load_labels("gsm8k_preference", "test")
        y_test_p = _load_labels("political_normative", "test")

        # Subsample training to equal sizes
        from src.config import DEFAULT_MIXED_DATASET_SIZE
        half = DEFAULT_MIXED_DATASET_SIZE // 2
        rng = np.random.RandomState(args.seed)
        idx_g = rng.permutation(len(y_train_g))[:half]
        idx_p = rng.permutation(len(y_train_p))[:half]

        X_pos_train = torch.cat([X_pos_train_g[idx_g], X_pos_train_p[idx_p]])
        X_neg_train = torch.cat([X_neg_train_g[idx_g], X_neg_train_p[idx_p]])
        # Political labels set to 0.5 (no ground truth) so supervised probe
        # learns to be uncertain on political items.
        y_train = np.concatenate([
            y_train_g[idx_g].astype(np.float64),
            np.full(len(idx_p), 0.5),
        ])

        X_pos_test = torch.cat([X_pos_test_g, X_pos_test_p])
        X_neg_test = torch.cat([X_neg_test_g, X_neg_test_p])
        y_test = np.concatenate([y_test_g, y_test_p])

        test_sources = (
            ["gsm8k_preference"] * len(y_test_g) +
            ["political_normative"] * len(y_test_p)
        )
    else:
        X_pos_train, X_neg_train = _load_activations(
            activation_dir, args.dataset, "train", model_name,
            config_main.prompt_format_key, layer,
        )
        X_pos_test, X_neg_test = _load_activations(
            activation_dir, args.dataset, "test", model_name,
            config_main.prompt_format_key, layer,
        )
        y_train = _load_labels(args.dataset, "train")
        y_test = _load_labels(args.dataset, "test")
        test_sources = None

    # Subsample by true proportion (use min class count for constant sizes)
    if args.train_true_proportion is not None:
        mask_true = y_train.astype(bool)
        true_idx = np.where(mask_true)[0]
        false_idx = np.where(~mask_true)[0]
        rng = np.random.RandomState(args.seed)
        rng.shuffle(true_idx)
        rng.shuffle(false_idx)

        tp = args.train_true_proportion
        base_size = min(len(true_idx), len(false_idx))
        if args.train_size is not None:
            base_size = min(base_size, args.train_size)

        if tp <= 0:
            idx = false_idx[:base_size]
        elif tp >= 1:
            idx = true_idx[:base_size]
        else:
            n_true = int(round(tp * base_size))
            n_false = base_size - n_true
            if n_true > len(true_idx) or n_false > len(false_idx):
                raise ValueError(
                    f"Cannot build TTP={tp} with {len(true_idx)} true "
                    f"and {len(false_idx)} false items (need {n_true}/{n_false})"
                )
            idx = np.concatenate([true_idx[:n_true], false_idx[:n_false]])

        X_pos_train = X_pos_train[idx]
        X_neg_train = X_neg_train[idx]
        y_train = y_train[idx]

    # Normalize (compute stats on train, apply to test)
    X_pos_train, pos_stats = normalize_activations(X_pos_train)
    X_neg_train, neg_stats = normalize_activations(X_neg_train)
    X_pos_test, _ = normalize_activations(X_pos_test, stats=pos_stats)
    X_neg_test, _ = normalize_activations(X_neg_test, stats=neg_stats)

    # Load easy dataset if needed
    easy_data = None
    needs_easy = args.method in ("eh", "ueeh", "random_ensemble_eh", "pca_ensemble_eh")
    if needs_easy:
        easy_config = get_dataset_config(args.easy_dataset)
        easy_model = args.model or easy_config.default_model
        X_pos_easy, X_neg_easy = _load_activations(
            activation_dir, args.easy_dataset, "train", easy_model,
            easy_config.prompt_format_key, layer,
        )
        y_easy = _load_labels(args.easy_dataset, "train")
        # Normalize easy with its own stats
        X_pos_easy, _ = normalize_activations(X_pos_easy)
        X_neg_easy, _ = normalize_activations(X_neg_easy)
        easy_data = (X_pos_easy, X_neg_easy, torch.tensor(y_easy, dtype=torch.bool))

    # Train and score
    X_diff_train = X_pos_train - X_neg_train
    X_diff_test = X_pos_test - X_neg_test

    if args.method == "ccs":
        probe = train_ccs(
            X_pos_train, X_neg_train,
            lr=args.ccs_lr, num_epochs=args.ccs_epochs,
            num_retries=args.ccs_retries, weight_decay=args.ccs_weight_decay,
        )
        train_scores = score_with_probe(probe, X_pos_train, X_neg_train)
        test_scores = score_with_probe(probe, X_pos_test, X_neg_test)

    elif args.method == "pca":
        pca = train_pca(X_diff_train, n_components=1)
        train_scores = score_with_pca(pca, X_diff_train)
        test_scores = score_with_pca(pca, X_diff_test)

    elif args.method == "supervised":
        # Split train into train/val for early stopping
        n_val = max(1, len(y_train) // 5)
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(y_train))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        X_tr = torch.cat([X_pos_train[tr_idx], X_neg_train[tr_idx]])
        y_tr = np.concatenate([y_train[tr_idx], 1.0 - y_train[tr_idx]])
        X_v = torch.cat([X_pos_train[val_idx], X_neg_train[val_idx]])
        y_v = np.concatenate([y_train[val_idx], 1.0 - y_train[val_idx]])

        probe = train_supervised(
            X_tr, y_tr, X_v, y_v,
            lr=args.supervised_lr, weight_decay=args.supervised_weight_decay,
            patience=args.supervised_patience, max_epochs=args.supervised_max_epochs,
        )
        train_scores = score_with_probe(probe, X_pos_train, X_neg_train)
        test_scores = score_with_probe(probe, X_pos_test, X_neg_test)

    elif args.method == "random":
        probe = init_random_probe(X_pos_train.shape[-1])
        train_scores = score_with_probe(probe, X_pos_train, X_neg_train)
        test_scores = score_with_probe(probe, X_pos_test, X_neg_test)

    elif args.method == "eh":
        assert easy_data is not None
        X_pos_e, X_neg_e, y_e = easy_data
        y_e_np = y_e.numpy()

        # Split easy data into train/val for early stopping
        n_val_e = max(1, len(y_e_np) // 5)
        rng_e = np.random.RandomState(args.seed)
        perm_e = rng_e.permutation(len(y_e_np))
        val_idx_e, tr_idx_e = perm_e[:n_val_e], perm_e[n_val_e:]

        X_tr_e = torch.cat([X_pos_e[tr_idx_e], X_neg_e[tr_idx_e]])
        y_tr_e = np.concatenate([y_e_np[tr_idx_e], ~y_e_np[tr_idx_e].astype(bool)])
        X_v_e = torch.cat([X_pos_e[val_idx_e], X_neg_e[val_idx_e]])
        y_v_e = np.concatenate([y_e_np[val_idx_e], ~y_e_np[val_idx_e].astype(bool)])

        probe = train_eh(
            X_tr_e, y_tr_e, X_v_e, y_v_e,
            lr=args.supervised_lr, weight_decay=args.supervised_weight_decay,
            patience=args.supervised_patience, max_epochs=args.supervised_max_epochs,
        )
        train_scores = score_with_probe(probe, X_pos_train, X_neg_train)
        test_scores = score_with_probe(probe, X_pos_test, X_neg_test)

    elif args.method == "ueeh":
        assert easy_data is not None
        X_pos_e, X_neg_e, y_e = easy_data
        X_e = torch.cat([X_pos_e, X_neg_e])
        y_e_float = torch.cat([y_e.float(), 1 - y_e.float()])
        probe = train_ueeh(
            X_pos_train, X_neg_train, X_e, y_e_float,
            alpha=args.ueeh_alpha, mode=args.ueeh_mode,
        )
        train_scores = score_with_probe(probe, X_pos_train, X_neg_train)
        test_scores = score_with_probe(probe, X_pos_test, X_neg_test)

    elif args.method == "random_ensemble":
        probes = train_random_ensemble(
            X_pos_train, X_neg_train,
            ensemble_size=args.ensemble_size,
            temperature=args.ensemble_temperature,
            scale_by_variance=args.scale_by_variance,
        )
        train_scores = score_with_ensemble(probes, X_pos_train, X_neg_train)
        test_scores = score_with_ensemble(probes, X_pos_test, X_neg_test)

    elif args.method == "random_ensemble_eh":
        assert easy_data is not None
        probes = train_random_ensemble(
            X_pos_train, X_neg_train,
            ensemble_size=args.ensemble_size,
            easy_dataset=easy_data,
            temperature=args.ensemble_temperature,
            scale_by_variance=args.scale_by_variance,
        )
        train_scores = score_with_ensemble(probes, X_pos_train, X_neg_train)
        test_scores = score_with_ensemble(probes, X_pos_test, X_neg_test)

    elif args.method == "pca_ensemble":
        pca = train_pca_ensemble(
            X_pos_train, X_neg_train,
            ensemble_size=args.ensemble_size,
            temperature=args.ensemble_temperature,
            scale_by_variance=args.scale_by_variance,
        )
        train_scores = score_with_pca_ensemble(pca, X_diff_train)
        test_scores = score_with_pca_ensemble(pca, X_diff_test)

    elif args.method == "pca_ensemble_eh":
        assert easy_data is not None
        pca = train_pca_ensemble(
            X_pos_train, X_neg_train,
            ensemble_size=args.ensemble_size,
            easy_dataset=easy_data,
            temperature=args.ensemble_temperature,
            scale_by_variance=args.scale_by_variance,
        )
        train_scores = score_with_pca_ensemble(pca, X_diff_train)
        test_scores = score_with_pca_ensemble(pca, X_diff_test)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Auto-flip for purely unsupervised methods (decide on train, apply to both)
    # For gsm_political, only use GSM8K items (political labels are 0.5/unknown).
    unsupervised = {"ccs", "pca", "random", "random_ensemble", "pca_ensemble"}
    if args.method in unsupervised:
        if is_mixed:
            n_gsm = len(idx_g)
            flip = compute_flip_direction(train_scores[:n_gsm], y_train[:n_gsm])
        else:
            flip = compute_flip_direction(train_scores, y_train)
        train_scores = apply_flip(train_scores, flip)
        test_scores = apply_flip(test_scores, flip)

    # Save results
    full_config = {
        "method": args.method,
        "dataset": args.dataset,
        "model": model_name,
        "layer": layer,
        "seed": args.seed,
        "train_true_proportion": args.train_true_proportion,
        "train_size": len(y_train),
        "easy_dataset": args.easy_dataset if needs_easy else None,
        "ensemble_size": args.ensemble_size if "ensemble" in args.method else None,
        "ensemble_temperature": args.ensemble_temperature if "ensemble" in args.method else None,
        "ueeh_alpha": args.ueeh_alpha if args.method == "ueeh" else None,
        "ccs_lr": args.ccs_lr,
        "ccs_epochs": args.ccs_epochs,
        "ccs_retries": args.ccs_retries,
        "ccs_weight_decay": args.ccs_weight_decay,
        "supervised_lr": args.supervised_lr,
        "supervised_patience": args.supervised_patience,
        "supervised_max_epochs": args.supervised_max_epochs,
        "supervised_weight_decay": args.supervised_weight_decay,
    }
    _save_results(
        output_dir, full_config,
        train_scores, y_train, test_scores, y_test,
        test_sources=test_sources,
        backup=not args.no_backup,
    )

    return 0


def args_from_config(config: dict, fixed: dict) -> argparse.Namespace:
    """Build an argparse.Namespace from a sweep config dict.

    Required by experiment_utils.run_sweep for --inprocess execution.
    """
    merged = {**config, **fixed}
    args = argparse.Namespace(
        method=merged.get("method"),
        dataset=merged.get("dataset"),
        model=merged.get("model"),
        layer=merged.get("layer"),
        activation_dir=merged.get("activation-dir"),
        train_true_proportion=merged.get("train-true-proportion"),
        train_size=merged.get("train-size"),
        easy_dataset=merged.get("easy-dataset", "larger_than"),
        ensemble_size=int(merged.get("ensemble-size", DEFAULT_ENSEMBLE_SIZE)),
        ensemble_temperature=float(merged.get("ensemble-temperature", DEFAULT_ENSEMBLE_TEMPERATURE)),
        scale_by_variance="scale-by-variance" in merged,
        ueeh_alpha=float(merged.get("ueeh-alpha", DEFAULT_UEEH_ALPHA)),
        ueeh_mode=merged.get("ueeh-mode", "paired"),
        use_soft_labels="use-soft-labels" in merged,
        ccs_lr=float(merged.get("ccs-lr", DEFAULT_CCS_LR)),
        ccs_epochs=int(merged.get("ccs-epochs", DEFAULT_CCS_EPOCHS)),
        ccs_retries=int(merged.get("ccs-retries", DEFAULT_CCS_RETRIES)),
        ccs_weight_decay=float(merged.get("ccs-weight-decay", DEFAULT_CCS_WEIGHT_DECAY)),
        supervised_lr=float(merged.get("supervised-lr", 0.03)),
        supervised_patience=int(merged.get("supervised-patience", DEFAULT_SUPERVISED_PATIENCE)),
        supervised_max_epochs=int(merged.get("supervised-max-epochs", DEFAULT_SUPERVISED_MAX_EPOCHS)),
        supervised_weight_decay=float(merged.get("supervised-weight-decay", 0.01)),
        seed=int(merged.get("seed", 0)),
        results_dir=merged.get("results-dir"),
        skip_if_exists="no-skip-if-exists" not in merged,
        no_backup="no-backup" in merged,
        debug="debug" in merged,
    )
    if args.train_true_proportion is not None:
        args.train_true_proportion = float(args.train_true_proportion)
    if args.train_size is not None:
        args.train_size = int(args.train_size)
    if args.layer is not None:
        args.layer = int(args.layer)
    return args


if __name__ == "__main__":
    sys.exit(main())
