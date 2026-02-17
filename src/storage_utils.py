"""Path management, hydra-style filenames, and GCS backup/cache operations."""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets"

CACHE_DIR = Path(os.getenv("CACHE_DIR", str(ROOT_DIR)))
RESULTS_DIR = CACHE_DIR / "results"

ACTIVATIONS_DIR = RESULTS_DIR / "activations"
PROBING_RESULTS_DIR = RESULTS_DIR / "probing"
PROMPTING_RESULTS_DIR = RESULTS_DIR / "prompting"

# ---------------------------------------------------------------------------
# Hydra-style filenames
# ---------------------------------------------------------------------------

_MAX_FILENAME_LEN = 255

# Keys to abbreviate when filenames are too long
_ABBREVIATIONS: dict[str, str] = {
    "dataset": "d",
    "method": "m",
    "model": "mod",
    "layer": "l",
    "seed": "s",
    "train_true_proportion": "ttp",
    "ensemble_size": "es",
    "ensemble_temperature": "et",
    "bootstrap_pool_size": "bps",
    "scale_by_variance": "sbv",
    "prompt_format": "pf",
    "shots": "sh",
}


def get_hydra_filename(config: dict, extension: str = "") -> str:
    """Build a readable filename from a config dict.

    Format: ``key1=val1,key2=val2``. If the result exceeds filesystem limits
    the keys are abbreviated.

    Args:
        config: Key-value pairs to encode.
        extension: Optional file extension (e.g. ``".json"``).

    Returns:
        A filesystem-safe string.
    """
    if not config:
        return "default" + extension

    def _encode(key: str, val: object) -> str:
        s = str(val)
        s = urllib.parse.quote(s, safe=",-./")
        return f"{key}={s}"

    name = ",".join(_encode(k, v) for k, v in sorted(config.items()))

    if len(name) + len(extension) > _MAX_FILENAME_LEN:
        short_config = {_ABBREVIATIONS.get(k, k): v for k, v in config.items()}
        name = ",".join(_encode(k, v) for k, v in sorted(short_config.items()))

    return name + extension


def get_config_from_hydra_filename(name: str) -> dict[str, str]:
    """Parse a hydra-style filename back into a dict."""
    name = name.removesuffix(".json")
    if name == "default":
        return {}
    result: dict[str, str] = {}
    for part in name.split(","):
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        result[key] = urllib.parse.unquote(val)
    return result


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def write_config_json(output_dir: Path, config: dict) -> Path:
    """Write a config dict to ``config.json`` in the given directory.

    Args:
        output_dir: Directory to write to (created if needed).
        config: Full experiment configuration.

    Returns:
        Path to the written config file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return path


def load_config_json(config_path: Path) -> dict:
    """Load a config dict from a JSON file."""
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GCS operations
# ---------------------------------------------------------------------------

_GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
_BACKUP_BUCKET = os.getenv("BACKUP_BUCKET")

_bucket_cache: dict = {}  # cached result of _get_bucket()


def _get_bucket():
    """Return the GCS bucket, or None if not configured."""
    if "result" in _bucket_cache:
        return _bucket_cache["result"]
    if not _GCP_PROJECT_ID or not _BACKUP_BUCKET:
        _bucket_cache["result"] = None
        return None
    try:
        from google.cloud import storage

        client = storage.Client(project=_GCP_PROJECT_ID)
        _bucket_cache["result"] = client.bucket(_BACKUP_BUCKET)
        return _bucket_cache["result"]
    except Exception as e:
        logger.warning("Could not connect to GCS: %s", e)
        _bucket_cache["result"] = None
        return None


def get_path_relative_to_cache_dir(path: Path) -> str:
    """Get a path relative to CACHE_DIR for use as a GCS blob name."""
    try:
        return str(path.resolve().relative_to(CACHE_DIR.resolve()))
    except ValueError:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))


def is_on_cloud(path: Path) -> bool:
    """Check if a file exists in the GCS bucket."""
    bucket = _get_bucket()
    if bucket is None:
        return False
    blob_name = get_path_relative_to_cache_dir(path)
    return bucket.blob(blob_name).exists()


def backup_to_cloud(path: Path, skip_if_exists: bool = True) -> bool:
    """Upload a local file to GCS.

    Returns:
        True if uploaded, False if skipped or GCS not configured.
    """
    bucket = _get_bucket()
    if bucket is None:
        return False

    blob_name = get_path_relative_to_cache_dir(path)
    blob = bucket.blob(blob_name)

    if skip_if_exists and blob.exists():
        return False

    blob.upload_from_filename(str(path))
    return True


def backup_dir_to_cloud(
    dir_path: Path, skip_if_exists: bool = True
) -> list[bool]:
    """Upload all files in a directory to GCS."""
    if not dir_path.is_dir():
        return []

    files = [f for f in dir_path.rglob("*") if f.is_file()]
    results = []
    for f in files:
        results.append(backup_to_cloud(f, skip_if_exists=skip_if_exists))
    return results


def cache_to_disk(path: Path, force: bool = False) -> bool:
    """Download a file from GCS to local disk if not already present.

    Returns:
        True if downloaded, False if skipped or GCS not configured.
    """
    if path.exists() and not force:
        return False

    bucket = _get_bucket()
    if bucket is None:
        return False

    blob_name = get_path_relative_to_cache_dir(path)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(path))
    return True


def cache_dir_to_disk(
    dir_path: Path,
    force: bool = False,
    quiet: bool = True,
) -> list[bool]:
    """Download all files under a GCS prefix to local disk."""
    bucket = _get_bucket()
    if bucket is None:
        return []

    prefix = get_path_relative_to_cache_dir(dir_path)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        return []

    def _download(blob) -> bool:
        local = CACHE_DIR / blob.name
        if local.exists() and not force:
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local))
        return True

    with ThreadPoolExecutor() as pool:
        results = list(pool.map(_download, blobs))

    if not quiet:
        downloaded = sum(results)
        logger.info("Cached %d/%d files to %s", downloaded, len(results), dir_path)

    return results
