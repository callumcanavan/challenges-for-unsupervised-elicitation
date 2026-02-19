#!/usr/bin/env python3
"""Download datasets from HuggingFace and convert to the JSON format expected by the codebase.

Usage:
    python scripts/download_datasets.py

If the HuggingFace repo is private, authenticate first with:
    huggingface-cli login
"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO = "callum-canavan/challenges-for-unsupervised-elicitation"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# Maps HF subset name -> (local filename stem, field transform function)
SUBSETS = {
    "gsm8k": ("gsm8k", None),
    "ctrl_z": ("ctrl_z_10steps", None),
    "normative_political": (
        "political_normative",
        lambda item: {
            "question": item["question"],
            "choice": item["choice"],
            "is_liberal": item["political_leaning"] == "liberal",
        },
    ),
    "larger_than": ("larger_than", None),
}


def download_subset(subset: str, local_stem: str, transform=None) -> None:
    for split in ("train", "test"):
        hf_path = f"{subset}/{split}.jsonl"
        local_file = hf_hub_download(
            HF_REPO, hf_path, repo_type="dataset",
        )
        with open(local_file) as f:
            items = [json.loads(line) for line in f if line.strip()]
        if transform is not None:
            items = [transform(item) for item in items]
        out_path = DATASETS_DIR / f"{split}_{local_stem}.json"
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  {out_path}: {len(items)} items")


def main():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for subset, (local_stem, transform) in SUBSETS.items():
        print(f"Downloading {subset}...")
        download_subset(subset, local_stem, transform)
    print(f"\nAll datasets saved to {DATASETS_DIR}")


if __name__ == "__main__":
    main()
