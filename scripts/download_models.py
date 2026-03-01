"""Download all models referenced in a config into the local cache_dir.

Usage:
    python scripts/download_models.py configs/qwen_mlp_config.yaml

Each model is saved to <cache_dir>/<hub-name with '/' replaced by '--'>, e.g.:
    model_cache/Qwen--Qwen3-Embedding-0.6B/
    model_cache/Qwen--Qwen3-0.6B/

train.py's resolve_model() will pick these up automatically, making training
fully offline. To force offline mode on a machine without internet access, also
set the environment variable:
    export HF_HUB_OFFLINE=1
"""

import os
import sys
import yaml
from huggingface_hub import snapshot_download


def download(repo_id: str, cache_dir: str) -> None:
    local_dir = os.path.join(cache_dir, repo_id.replace("/", "--"))
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"[skip] {repo_id}  →  already at {local_dir}")
        return
    print(f"[download] {repo_id}  →  {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"[done]  {repo_id}")


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["models"].get("cache_dir") or "model_cache"
    os.makedirs(cache_dir, exist_ok=True)

    models = [
        cfg["models"]["embedder"]["name"],
        cfg["models"]["llm"]["name"],
    ]

    for name in models:
        download(name, cache_dir)

    print(f"\nAll models cached in: {os.path.abspath(cache_dir)}")
    print("You can now transfer this directory to the H100 machine and run training offline.")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen_mlp_config.yaml"
    main(config)
