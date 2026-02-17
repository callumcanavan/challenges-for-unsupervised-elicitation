#!/usr/bin/env python3
"""Probing experiment sweep definitions for the UE challenges paper.

Sweeps:
- Imbalanced GSM8K: all methods x true_proportions x seeds
- Imbalanced CtrlZ: all methods x proportions x seeds
- Impossible task (GSMPolitical): all methods x seeds
- Ensemble sweep: ensemble_size x seeds

Run with:
    python experiments/probing_sweep.py
    python experiments/probing_sweep.py --dry-run
    python experiments/probing_sweep.py -e seed 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_utils import run_sweep

# ============================================================================
# Sweep definitions
# ============================================================================

UNSUPERVISED_METHODS = ["ccs", "pca", "random", "random_ensemble", "pca_ensemble"]
SUPERVISED_METHODS = ["supervised", "eh"]
JOINT_METHODS = ["ueeh"]
EH_ENSEMBLE_METHODS = ["random_ensemble_eh", "pca_ensemble_eh"]
ALL_METHODS = UNSUPERVISED_METHODS + SUPERVISED_METHODS + JOINT_METHODS + EH_ENSEMBLE_METHODS

GSM8K_TRUE_PROPORTIONS = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]
CTRL_Z_TRUE_PROPORTIONS = [0.0, 0.01, 0.5]
SEEDS = [0, 1, 2]

# --- Imbalanced GSM8K ---
IMBALANCED_GSM8K = (
    ("dataset", "gsm8k_preference"),
    ("train-true-proportion", GSM8K_TRUE_PROPORTIONS),
    [
        ("method", UNSUPERVISED_METHODS + SUPERVISED_METHODS),
        (
            ("method", JOINT_METHODS + EH_ENSEMBLE_METHODS),
            ("easy-dataset", "larger_than"),
        ),
    ],
    ("seed", SEEDS),
)

# --- Imbalanced CtrlZ (8B) ---
IMBALANCED_CTRL_Z = (
    ("dataset", "ctrl_z_10steps"),
    ("train-true-proportion", CTRL_Z_TRUE_PROPORTIONS),
    [
        ("method", UNSUPERVISED_METHODS + SUPERVISED_METHODS),
        (
            ("method", JOINT_METHODS + EH_ENSEMBLE_METHODS),
            ("easy-dataset", "larger_than"),
        ),
    ],
    ("seed", SEEDS),
)

# --- Imbalanced CtrlZ (70B) ---
IMBALANCED_CTRL_Z_70B = (
    ("dataset", "ctrl_z_10steps"),
    ("model", "meta-llama/Llama-3.1-70B"),
    ("train-true-proportion", CTRL_Z_TRUE_PROPORTIONS),
    [
        ("method", UNSUPERVISED_METHODS + SUPERVISED_METHODS),
        (
            ("method", JOINT_METHODS + EH_ENSEMBLE_METHODS),
            ("easy-dataset", "larger_than"),
        ),
    ],
    ("seed", SEEDS),
)

# --- Impossible task (GSMPolitical) ---
# Uses 70B model for both gsm8k and political_normative activations
IMPOSSIBLE_TASK = (
    ("dataset", "gsm_political"),
    ("model", "meta-llama/Llama-3.1-70B"),
    [
        ("method", UNSUPERVISED_METHODS + SUPERVISED_METHODS),
        (
            ("method", JOINT_METHODS + EH_ENSEMBLE_METHODS),
            ("easy-dataset", "larger_than"),
        ),
    ],
    ("seed", SEEDS),
)

# --- Full paper sweep ---
SWEEPS = [IMBALANCED_GSM8K, IMBALANCED_CTRL_Z, IMBALANCED_CTRL_Z_70B, IMPOSSIBLE_TASK]

FIXED: dict = {"no-backup": True}
ENV: dict = {}


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inprocess", action="store_true",
                        help="Run in-process to avoid subprocess startup overhead")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Output directory, e.g. results/debug/probing")
    parser.add_argument("--no-skip-if-exists", action="store_true",
                        help="Re-run experiments even if results exist")
    parser.add_argument(
        "--extra", "-e", nargs=2, action="append",
        metavar=("KEY", "VALUE"),
        help="Extra sweep args, e.g. -e seed 0",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Use results/debug/ directories by default")
    args = parser.parse_args()

    fixed = dict(FIXED)
    if args.debug:
        fixed["debug"] = True
        if not args.results_dir:
            fixed["results-dir"] = "results/debug/probing"
            fixed["activation-dir"] = "results/debug/activations"
    if args.results_dir:
        fixed["results-dir"] = args.results_dir
    if args.no_skip_if_exists:
        fixed["no-skip-if-exists"] = True

    sweeps = SWEEPS
    if args.extra:
        extra = tuple(
            (k, eval(v) if v in ("True", "False") else v)
            for k, v in args.extra
        )
        sweeps = (extra, sweeps)

    run_sweep(
        script="scripts/run_probing.py",
        sweeps=sweeps,
        fixed=fixed,
        dry_run=args.dry_run,
        env=ENV,
        inprocess=args.inprocess,
    )
