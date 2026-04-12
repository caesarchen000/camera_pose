#!/usr/bin/env python3
"""
Download official DynamiCrafter weights from Hugging Face into checkpoints/.

For generative frame interpolation (two keyframes), the upstream project only
ships one pretrained checkpoint:

  Doubiiu/DynamiCrafter_512_Interp  ->  checkpoints/dynamicrafter_512_interp_v1/model.ckpt

Higher-resolution DynamiCrafter_1024 is image-to-video (single image), not this
interpolation checkpoint; it uses different configs and is not a drop-in swap.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


PRESETS = {
    "interp": {
        "repo_id": "Doubiiu/DynamiCrafter_512_Interp",
        "local_dir_suffix": "dynamicrafter_512_interp_v1",
        "note": "Only official interp release (320x512, two-frame).",
    },
    "i2v512": {
        "repo_id": "Doubiiu/DynamiCrafter_512",
        "local_dir_suffix": "dynamicrafter_512_v1",
        "note": "Single-image I2V at 320x512 (not interpolation).",
    },
    "i2v1024": {
        "repo_id": "Doubiiu/DynamiCrafter_1024",
        "local_dir_suffix": "dynamicrafter_1024_v1",
        "note": "Single-image I2V at 576x1024 (paper: strong VBench I2V; not interp).",
    },
    "i2v256": {
        "repo_id": "Doubiiu/DynamiCrafter",
        "local_dir_suffix": "dynamicrafter_256_v1",
        "note": "Single-image I2V at 256x256.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "preset",
        nargs="?",
        default="interp",
        choices=sorted(PRESETS.keys()),
        help="Which weight bundle to fetch (default: interp).",
    )
    parser.add_argument(
        "--dyn_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="DynamiCrafter repo root (contains checkpoints/).",
    )
    args = parser.parse_args()
    dyn_root = args.dyn_root.resolve()
    ckpt_root = dyn_root / "checkpoints"
    spec = PRESETS[args.preset]
    local_dir = ckpt_root / spec["local_dir_suffix"]
    local_dir.mkdir(parents=True, exist_ok=True)

    print(spec["note"])
    print(f"Repo: {spec['repo_id']}")
    print(f"Destination: {local_dir / 'model.ckpt'}")

    path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename="model.ckpt",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    size = os.path.getsize(path)
    print(f"Done. File: {path} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
