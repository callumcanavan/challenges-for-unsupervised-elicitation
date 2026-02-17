#!/usr/bin/env python3
"""Utilities for running experiment sweeps.

Sweep specification uses implicit typing:
- ``(str, value_or_list)`` → range of values for a single parameter
- ``(...)`` → outer product of items (cartesian product)
- ``[...]`` → concatenate resolved sub-sweeps (alternatives)

Example::

    SWEEPS = [
        (
            ("dataset", ["gsm8k_preference", "ctrl_z_10steps"]),
            ("method", ["ccs", "pca", "supervised"]),
            ("seed", [0, 1, 2]),
        ),
    ]
    run_sweep(script="scripts/run_probing.py", sweeps=SWEEPS)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from pprint import pformat

from src.storage_utils import ROOT_DIR

LOGS_DIR = ROOT_DIR / "logs"


def setup_sweep_logging(script_name: str) -> logging.Logger:
    """Set up logging to both terminal and a timestamped file."""
    LOGS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_stem = Path(script_name).stem
    log_file = LOGS_DIR / f"{script_stem}_{timestamp}.log"

    logger = logging.getLogger(f"sweep.{script_stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("Logging to: %s", log_file)
    return logger


def pow2_shots(max_shots: int) -> list[int]:
    """Generate shot counts: 0, 1, 2, 4, 8, ... up to *max_shots*."""
    shots = [0]
    n = 1
    while n <= max_shots:
        shots.append(n)
        n *= 2
    return shots


def resolve_sweep(sweep) -> list[dict]:
    """Resolve a sweep specification into a list of config dicts.

    See module docstring for specification format.
    """
    # Range: (key, value_or_values)
    if isinstance(sweep, tuple) and len(sweep) == 2 and isinstance(sweep[0], str):
        key, values = sweep
        if not isinstance(values, list):
            values = [values]
        return [{key: v} for v in values]

    # Outer product: (item1, item2, ...)
    if isinstance(sweep, tuple):
        result = [{}]
        for item in sweep:
            item_configs = resolve_sweep(item)
            result = [
                {**existing, **new}
                for existing in result
                for new in item_configs
            ]
        return result

    # Concatenate: [sub1, sub2, ...]
    if isinstance(sweep, list):
        result = []
        for sub in sweep:
            if isinstance(sub, tuple) and len(sub) == 2 and isinstance(sub[0], str):
                sub = (sub,)
            result.extend(resolve_sweep(sub))
        return result

    raise ValueError(f"Unknown sweep type: {type(sweep)}")


def _add_arg(cmd: list[str], key: str, value) -> None:
    """Append a CLI argument to *cmd*.

    True → ``--key`` (flag), False → omit, None → ``--key none``.
    """
    if value is True:
        cmd.append(f"--{key}")
    elif value is False:
        pass
    elif value is None:
        cmd.append(f"--{key}")
        cmd.append("none")
    else:
        cmd.append(f"--{key}")
        cmd.append(str(value))


def build_commands(
    script: str,
    sweeps,
    fixed: dict | None = None,
) -> list[list[str]]:
    """Build subprocess command lists from a sweep specification."""
    fixed = fixed or {}
    configs = resolve_sweep(sweeps)
    commands = []
    for config in configs:
        cmd = ["python", script]
        for key, value in config.items():
            _add_arg(cmd, key, value)
        for key, value in fixed.items():
            _add_arg(cmd, key, value)
        commands.append(cmd)
    return commands


def run_sweep(
    script: str,
    sweeps,
    fixed: dict | None = None,
    dry_run: bool = False,
    env: dict | None = None,
    inprocess: bool = False,
) -> None:
    """Run an experiment sweep.

    Args:
        script: Path to the Python script.
        sweeps: Sweep specification.
        fixed: Fixed args applied to every command.
        dry_run: If True, only print commands.
        env: Extra environment variables.
        inprocess: If True, import the script and call ``main()`` directly
            to avoid reloading the model between experiments.
    """
    logger = setup_sweep_logging(script)

    logger.info("=" * 50)
    logger.info("Sweep specification:")
    logger.info(pformat(sweeps, indent=2, width=100))
    if fixed:
        logger.info("Fixed args: %s", pformat(fixed, indent=2))

    configs = resolve_sweep(sweeps)
    commands = build_commands(script, sweeps, fixed)

    logger.info("=" * 50)
    logger.info("Experiment Sweep: %s", script)
    logger.info("Total experiments: %d", len(commands))
    if dry_run:
        logger.info("(DRY RUN)")
    logger.info("=" * 50)

    for i, config in enumerate(configs):
        args_str = ", ".join(f"{k}={v}" for k, v in config.items())
        logger.info("  [%d] %s", i + 1, args_str)
    logger.info("")

    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    os.chdir(ROOT_DIR)

    # In-process execution
    script_main = None
    script_args_from_config = None
    if inprocess and not dry_run:
        import importlib.util

        script_path = ROOT_DIR / script
        spec = importlib.util.spec_from_file_location("sweep_script", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["sweep_script"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "main") or not hasattr(module, "args_from_config"):
            raise AttributeError(
                f"Script {script} must export main(args) and args_from_config(config, fixed) "
                "for in-process execution."
            )
        script_main = module.main
        script_args_from_config = module.args_from_config
        logger.info("Using in-process execution")

    for i, (cmd, config) in enumerate(zip(commands, configs)):
        logger.info("=" * 50)
        logger.info("[%d/%d] %s", i + 1, len(commands), " ".join(cmd))
        logger.info("=" * 50)

        if not dry_run:
            if inprocess and script_main is not None:
                args = script_args_from_config(config, fixed)
                rc = script_main(args)
                if rc != 0:
                    logger.error("Failed with return code %d", rc)
                    sys.exit(rc)
            else:
                result = subprocess.run(cmd, env=run_env)
                if result.returncode != 0:
                    logger.error("Failed with return code %d", result.returncode)
                    sys.exit(result.returncode)

    logger.info("=" * 50)
    logger.info("All experiments complete!")
    logger.info("=" * 50)
