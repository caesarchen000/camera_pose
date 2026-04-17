#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tvio
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate InterPose frame subsets per generated video: "
            "m=11 subsets (10 random + 1 uniform), each with 2 original + 3 generated frames."
        )
    )
    p.add_argument(
        "--video_root",
        type=Path,
        required=True,
        help="Root containing pair folders and DynamiCrafter outputs.",
    )
    p.add_argument(
        "--manifest_path",
        type=Path,
        default=None,
        help="Optional manifest output path (default: <video_root>/subsets_manifest.jsonl).",
    )
    p.add_argument("--m", type=int, default=11, help="Total subsets per video.")
    p.add_argument(
        "--num_random",
        type=int,
        default=10,
        help="Number of random subsets (usually 10; plus one uniform subset).",
    )
    p.add_argument(
        "--k_generated",
        type=int,
        default=3,
        help="Generated frames per subset (k=5 total means k_generated=3).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def discover_videos(video_root: Path) -> list[Path]:
    return sorted(video_root.glob("**/samples_separate/00_start_sample0.mp4"))


def pick_uniform_indices(inner_indices: np.ndarray, k_generated: int) -> list[int]:
    if len(inner_indices) == 0:
        return []
    if len(inner_indices) >= k_generated:
        picks = np.linspace(0, len(inner_indices) - 1, num=k_generated, dtype=int)
        return [int(inner_indices[i]) for i in picks]
    # Fallback: repeat last index if too few.
    out = inner_indices.tolist()
    while len(out) < k_generated:
        out.append(out[-1])
    return [int(x) for x in out]


def pick_random_indices(
    rng: np.random.Generator,
    inner_indices: np.ndarray,
    k_generated: int,
) -> list[int]:
    if len(inner_indices) == 0:
        return []
    replace = len(inner_indices) < k_generated
    picks = rng.choice(inner_indices, size=k_generated, replace=replace)
    picks = sorted(int(x) for x in picks.tolist())
    return picks


def save_png(frame_hwc: np.ndarray, path: Path) -> None:
    Image.fromarray(frame_hwc).save(path)


def find_pair_dir(path: Path) -> Path | None:
    for p in path.parents:
        if p.name.startswith("pair_"):
            return p
    return None


def run_for_video(
    video_path: Path,
    rng: np.random.Generator,
    m: int,
    num_random: int,
    k_generated: int,
) -> list[dict]:
    parent = video_path.parents[1]  # .../<variant_or_pair>
    pair_dir = find_pair_dir(video_path)
    if pair_dir is None:
        return []

    prompt_pack = parent / "prompt_pack"
    if not prompt_pack.is_dir():
        prompt_pack = pair_dir / "prompt_pack"
    start_img = prompt_pack / "00_start.png"
    end_img = prompt_pack / "01_end.png"
    if not start_img.is_file() or not end_img.is_file():
        return []

    frames, _, _ = tvio.read_video(str(video_path), pts_unit="sec")
    # [T,H,W,C], uint8
    if frames.numel() == 0:
        return []
    frames_np = frames.numpy()
    total = frames_np.shape[0]
    inner_indices = np.arange(1, max(1, total - 1), dtype=int)

    records = []
    variant_name = parent.name if parent != pair_dir else "base"
    base_dir = pair_dir / "subset" / variant_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Random subsets.
    for i in range(num_random):
        idxs = pick_random_indices(rng, inner_indices, k_generated)
        subset_dir = base_dir / f"subset_{i:02d}_random"
        subset_dir.mkdir(parents=True, exist_ok=True)
        # originals
        Image.open(start_img).save(subset_dir / "00_input_start.png")
        Image.open(end_img).save(subset_dir / "01_input_end.png")
        for j, fidx in enumerate(idxs):
            save_png(frames_np[fidx], subset_dir / f"{j+2:02d}_gen_f{fidx:03d}.png")
        records.append(
            {
                "video_path": str(video_path),
                "pair_dir": str(pair_dir),
                "variant": variant_name,
                "subset_type": "random",
                "subset_idx": i,
                "generated_frame_indices": idxs,
                "subset_dir": str(subset_dir),
            }
        )

    # Uniform subset (one).
    if m > num_random:
        idxs = pick_uniform_indices(inner_indices, k_generated)
        subset_dir = base_dir / f"subset_{num_random:02d}_uniform"
        subset_dir.mkdir(parents=True, exist_ok=True)
        Image.open(start_img).save(subset_dir / "00_input_start.png")
        Image.open(end_img).save(subset_dir / "01_input_end.png")
        for j, fidx in enumerate(idxs):
            save_png(frames_np[fidx], subset_dir / f"{j+2:02d}_gen_f{fidx:03d}.png")
        records.append(
            {
                "video_path": str(video_path),
                "pair_dir": str(pair_dir),
                "variant": variant_name,
                "subset_type": "uniform",
                "subset_idx": num_random,
                "generated_frame_indices": idxs,
                "subset_dir": str(subset_dir),
            }
        )

    return records


def main() -> None:
    args = parse_args()
    video_root = args.video_root.resolve()

    videos = discover_videos(video_root)
    if not videos:
        raise FileNotFoundError(f"No videos found under {video_root}/**/samples_separate/00_start_sample0.mp4")

    rng = np.random.default_rng(args.seed)
    all_records: list[dict] = []
    for vp in videos:
        recs = run_for_video(
            video_path=vp,
            rng=rng,
            m=args.m,
            num_random=args.num_random,
            k_generated=args.k_generated,
        )
        all_records.extend(recs)
        print(f"[ok] {vp} -> {len(recs)} subsets")

    manifest = args.manifest_path.resolve() if args.manifest_path else (video_root / "subsets_manifest.jsonl")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nDone. Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
