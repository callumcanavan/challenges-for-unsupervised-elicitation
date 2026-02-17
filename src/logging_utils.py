"""Centralized logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Create a logger with console output and optional file handler.

    Args:
        name: Logger name (typically __name__).
        log_file: Optional path for file logging.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console)

    if log_file is not None:
        add_file_handler(logger, log_file)

    return logger


def add_file_handler(logger: logging.Logger, log_file: str | Path) -> None:
    """Add a file handler to an existing logger."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
