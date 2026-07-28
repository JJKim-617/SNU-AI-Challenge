#!/usr/bin/env python3
"""Download the base model (Qwen/Qwen3.6-27B, Apache-2.0) into weights/base_model.

~52 GB; run once on a machine with network access. Base weights are used unmodified.

Usage: python scripts/download_base_model.py [--dest weights/base_model]
"""
import argparse
from pathlib import Path

REPO = "Qwen/Qwen3.6-27B"


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO, help="Hugging Face repo id")
    ap.add_argument("--dest", default=str(here / "weights" / "base_model"),
                    help="local directory to download into")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub is required: pip install -r requirements.txt")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} -> {dest}  (~52 GB, 15 shards)")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(dest),
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*",
                        "*preprocessor*", "generation_config*"],
    )

    # integrity check: every shard named in the index must be present
    import json
    idx = json.load(open(dest / "model.safetensors.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
    missing = [s for s in shards if not (dest / s).exists()]
    print(f"{len(shards)} shards expected, missing: {missing or 'none'}")
    if missing:
        raise SystemExit("download incomplete — re-run this script")
    print(f"OK. Point the pipeline at it:\n  ln -sfn {dest} weights/base_model  "
          "(or pass --model-dir to score_frames.py)")


if __name__ == "__main__":
    main()
