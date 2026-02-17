#!/usr/bin/env python3
"""Prompting experiment sweep definitions for the UE challenges paper.

Sweeps:
- Imbalanced GSM8K: all prompting methods x true_proportions x seeds
- Imbalanced CtrlZ: all prompting methods x proportions x seeds
- Impossible task (GSMPolitical): all prompting methods x seeds

Run with:
    python experiments/prompting_sweep.py
    python experiments/prompting_sweep.py --dry-run
    python experiments/prompting_sweep.py --inprocess
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_utils import run_sweep

# ============================================================================
# Sweep definitions
# ============================================================================

GSM8K_TRUE_PROPORTIONS = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]
CTRL_Z_TRUE_PROPORTIONS = [0.0, 0.01, 0.5]
SEEDS = [0, 1, 2]

RANDOM_SHOTS = "2"
GOLDEN_SHOTS = "32"
BOOTSTRAP_SHOTS = "2,32"

# --- Imbalanced GSM8K ---
IMBALANCED_GSM8K = (
    ("dataset", "gsm8k_preference"),
    ("train-true-proportion", GSM8K_TRUE_PROPORTIONS),
    [
        ("method", "zero_shot"),
        (("method", "random_few_shot"), ("shots", RANDOM_SHOTS)),
        (("method", "golden_few_shot"), ("shots", GOLDEN_SHOTS)),
        (("method", "bootstrap"), ("shots", BOOTSTRAP_SHOTS)),
    ],
    ("seed", SEEDS),
)

# --- Imbalanced CtrlZ (8B) ---
# Single seed for prompting — ctrl_z prompts are long and slow
IMBALANCED_CTRL_Z = (
    ("dataset", "ctrl_z_10steps"),
    ("train-true-proportion", CTRL_Z_TRUE_PROPORTIONS),
    [
        ("method", "zero_shot"),
        (("method", "random_few_shot"), ("shots", RANDOM_SHOTS)),
        (("method", "golden_few_shot"), ("shots", GOLDEN_SHOTS)),
        (("method", "bootstrap"), ("shots", BOOTSTRAP_SHOTS)),
    ],
    ("seed", 0),
)

# --- Imbalanced CtrlZ (70B) ---
IMBALANCED_CTRL_Z_70B = (
    ("dataset", "ctrl_z_10steps"),
    ("model", "meta-llama/Llama-3.1-70B"),
    ("train-true-proportion", CTRL_Z_TRUE_PROPORTIONS),
    [
        ("method", "zero_shot"),
        (("method", "random_few_shot"), ("shots", RANDOM_SHOTS)),
        (("method", "golden_few_shot"), ("shots", GOLDEN_SHOTS)),
        (("method", "bootstrap"), ("shots", BOOTSTRAP_SHOTS)),
    ],
    ("seed", 0),
)

# --- Impossible task (GSMPolitical, 70B) ---
IMPOSSIBLE_TASK = (
    ("dataset", "gsm_political"),
    ("model", "meta-llama/Llama-3.1-70B"),
    [
        ("method", "zero_shot"),
        (("method", "random_few_shot"), ("shots", RANDOM_SHOTS)),
        (("method", "golden_few_shot"), ("shots", GOLDEN_SHOTS)),
        (("method", "bootstrap"), ("shots", BOOTSTRAP_SHOTS)),
    ],
    ("seed", SEEDS),
)

# --- Full paper sweep ---
SWEEPS_8B = [IMBALANCED_GSM8K, IMBALANCED_CTRL_Z]
SWEEPS_70B = [IMBALANCED_CTRL_Z_70B, IMPOSSIBLE_TASK]
SWEEPS = SWEEPS_8B + SWEEPS_70B

FIXED: dict = {"no-backup": True}
ENV: dict = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inprocess", action="store_true",
                        help="Run in-process to reuse vLLM model")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Output directory, e.g. results/debug/prompting")
    parser.add_argument("--no-skip-if-exists", action="store_true",
                        help="Re-run experiments even if results exist")
    parser.add_argument(
        "--extra", "-e", nargs=2, action="append",
        metavar=("KEY", "VALUE"),
        help="Extra sweep args, e.g. -e seed 0",
    )
    parser.add_argument("--model-size", choices=["8b", "70b", "all"],
                        default="all",
                        help="Filter sweep by model size (for separate SLURM jobs)")
    parser.add_argument("--debug", action="store_true",
                        help="Generate random scores instead of running vLLM")
    args = parser.parse_args()

    fixed = dict(FIXED)
    if args.debug:
        fixed["debug"] = True
        if not args.results_dir:
            fixed["results-dir"] = "results/debug/prompting"
    if args.results_dir:
        fixed["results-dir"] = args.results_dir
    if args.no_skip_if_exists:
        fixed["no-skip-if-exists"] = True

    if args.model_size == "8b":
        sweeps = SWEEPS_8B
    elif args.model_size == "70b":
        sweeps = SWEEPS_70B
    else:
        sweeps = SWEEPS
    if args.extra:
        extra = tuple(
            (k, eval(v) if v in ("True", "False") else v)
            for k, v in args.extra
        )
        sweeps = (extra, sweeps)

    run_sweep(
        script="scripts/run_prompting.py",
        sweeps=sweeps,
        fixed=fixed,
        dry_run=args.dry_run,
        env=ENV,
        inprocess=args.inprocess,
    )
