#!/usr/bin/env python3
"""Upload checkpoints and dataset to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py --username your_hf_username --token hf_xxx
    HF_TOKEN=hf_xxx python scripts/upload_to_hf.py --username your_hf_username

    # Only dataset:
    python scripts/upload_to_hf.py --username your_hf_username --skip-checkpoints

    # Only checkpoints:
    python scripts/upload_to_hf.py --username your_hf_username --skip-dataset
"""

import argparse
import os
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent.parent   # experiments/
ALL_CODE    = EXPERIMENTS.parent                        # all_code_and_stuff/

CHECKPOINTS = {
    "mapper-llm-perceiver-ctx16-full-ft":      ALL_CODE / "checkpoints/perceiver_ctx16_full_ft/last.ckpt",
    "mapper-llm-perceiver-ctx16-mapper-only":  ALL_CODE / "checkpoints/perceiver_ctx16_mapper_only/last.ckpt",
}

DATASET_PATH = EXPERIMENTS / "data/unified_dataset"


def parse_args():
    p = argparse.ArgumentParser(description="Upload to Hugging Face Hub")
    p.add_argument("--username", required=True, help="Your HF username (e.g. john-doe)")
    p.add_argument("--token",    default=os.environ.get("HF_TOKEN"),
                   help="HF write token; or set HF_TOKEN env var")
    p.add_argument("--skip-checkpoints", action="store_true", help="Skip uploading .ckpt files")
    p.add_argument("--skip-dataset",     action="store_true", help="Skip uploading dataset")
    return p.parse_args()


def upload_checkpoints(api, username, token):
    from huggingface_hub import create_repo

    for repo_name, ckpt_path in CHECKPOINTS.items():
        repo_id = f"{username}/{repo_name}"
        size_gb = ckpt_path.stat().st_size / 1e9
        print(f"\n{'='*60}")
        print(f"Checkpoint: {repo_id}")
        print(f"File:       {ckpt_path.name}  ({size_gb:.1f} GB)")

        create_repo(repo_id, repo_type="model", private=False,
                    token=token, exist_ok=True)

        print("Uploading... (large file — may take 10-30 min on home internet)")
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo="last.ckpt",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Done -> https://huggingface.co/{repo_id}")


def upload_dataset(username, token):
    from datasets import load_from_disk
    from huggingface_hub import create_repo

    repo_id = f"{username}/mapper-llm-dataset"
    print(f"\n{'='*60}")
    print(f"Dataset: {repo_id}")
    print(f"Path:    {DATASET_PATH}")

    create_repo(repo_id, repo_type="dataset", private=False,
                token=token, exist_ok=True)

    print("Loading dataset from disk...")
    ds = load_from_disk(str(DATASET_PATH))
    print(ds)

    print("Uploading... (3.3 GB — may take 5-15 min)")
    ds.push_to_hub(repo_id, token=token, private=False)
    print(f"Done -> https://huggingface.co/datasets/{repo_id}")


def main():
    args = parse_args()

    if not args.token:
        raise SystemExit(
            "Error: no HF token found.\n"
            "Pass --token hf_xxx  or  export HF_TOKEN=hf_xxx"
        )

    if args.skip_checkpoints and args.skip_dataset:
        raise SystemExit("Nothing to do: both --skip-checkpoints and --skip-dataset set.")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    print(f"Logged in as: {api.whoami()['name']}")

    if not args.skip_checkpoints:
        upload_checkpoints(api, args.username, args.token)

    if not args.skip_dataset:
        upload_dataset(args.username, args.token)

    print(f"\nAll uploads complete. Profile: https://huggingface.co/{args.username}")


if __name__ == "__main__":
    main()
