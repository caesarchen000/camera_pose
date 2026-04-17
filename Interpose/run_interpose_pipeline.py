#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run full InterPose pipeline from a pairs CSV: "
            "generate 4 variants per pair -> generate subsets -> run DUSt3R on subsets "
            "-> select consistent video poses."
        )
    )
    p.add_argument("--pairs_csv", type=Path, required=True, help="CSV with rel_a/rel_b columns.")
    p.add_argument("--data_root", type=Path, required=True, help="DL3DV benchmark root.")
    p.add_argument("--out_root", type=Path, required=True, help="Pipeline output root directory.")

    p.add_argument("--prompt1", type=str, required=True)
    p.add_argument("--prompt2", type=str, required=True)
    p.add_argument("--seed1", type=int, default=123)
    p.add_argument("--seed2", type=int, default=456)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--cfg_scale", type=float, default=7.5)

    p.add_argument("--checkpoint", type=Path, required=True, help="DUSt3R checkpoint path.")
    p.add_argument("--dust3r_root", type=Path, default=Path("dust3r"))
    p.add_argument("--image_size", type=int, default=512, choices=(224, 512))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--align_niter", type=int, default=300)
    p.add_argument("--align_lr", type=float, default=0.01)
    p.add_argument("--align_schedule", type=str, default="cosine")

    p.add_argument("--start_index", type=int, default=0, help="Start pair index in CSV.")
    p.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="How many pairs to process from start_index (default: all remaining).",
    )

    p.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip 4-video generation stage and reuse existing outputs in out_root.",
    )
    p.add_argument(
        "--skip_existing_videos",
        action="store_true",
        help="During generation, skip variants that already have 00_start_sample0.mp4.",
    )
    p.add_argument(
        "--skip_subsets",
        action="store_true",
        help="Skip subset generation stage and reuse existing subset folders.",
    )
    p.add_argument(
        "--skip_dust3r_subsets",
        action="store_true",
        help="Skip DUSt3R subset stage and reuse existing dust3r_subset_poses.jsonl.",
    )
    p.add_argument(
        "--skip_selection",
        action="store_true",
        help="Skip consistency selection stage.",
    )
    return p.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    pairs_csv = args.pairs_csv.resolve()
    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    ckpt = args.checkpoint.resolve()
    dust3r_root = args.dust3r_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    total_pairs = count_csv_rows(pairs_csv)
    if total_pairs == 0:
        raise ValueError(f"No rows found in CSV: {pairs_csv}")
    if args.start_index < 0 or args.start_index >= total_pairs:
        raise ValueError(f"--start_index {args.start_index} out of range [0, {total_pairs - 1}]")

    end_index = total_pairs
    if args.max_pairs is not None:
        if args.max_pairs <= 0:
            raise ValueError("--max_pairs must be positive")
        end_index = min(total_pairs, args.start_index + args.max_pairs)

    # Stage 1: generate 4 variants for each target pair.
    if not args.skip_generation:
        for pair_idx in range(args.start_index, end_index):
            cmd = [
                sys.executable,
                "Interpose/generate_four_variants.py",
                "--pairs_csv",
                str(pairs_csv),
                "--data_root",
                str(data_root),
                "--pair_index",
                str(pair_idx),
                "--prompt1",
                args.prompt1,
                "--prompt2",
                args.prompt2,
                "--seed1",
                str(args.seed1),
                "--seed2",
                str(args.seed2),
                "--ddim_steps",
                str(args.ddim_steps),
                "--frame_stride",
                str(args.frame_stride),
                "--cfg_scale",
                str(args.cfg_scale),
                "--out_root",
                str(out_root),
            ]
            if args.skip_existing_videos:
                cmd.append("--skip_existing")
            run_cmd(cmd, cwd=repo_root)

    # Stage 2: generate subsets.
    if not args.skip_subsets:
        cmd = [
            sys.executable,
            "Interpose/generate_subsets.py",
            "--video_root",
            str(out_root),
        ]
        run_cmd(cmd, cwd=repo_root)

    # Stage 3: run DUSt3R on subsets.
    subset_pose_manifest = out_root / "dust3r_subset_poses.jsonl"
    if not args.skip_dust3r_subsets:
        cmd = [
            sys.executable,
            "Interpose/run_dust3r_subsets.py",
            "--subset_root",
            str(out_root),
            "--checkpoint",
            str(ckpt),
            "--dust3r_root",
            str(dust3r_root),
            "--image_size",
            str(args.image_size),
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
            "--align_niter",
            str(args.align_niter),
            "--align_lr",
            str(args.align_lr),
            "--align_schedule",
            args.align_schedule,
        ]
        run_cmd(cmd, cwd=repo_root)

    # Stage 4: consistency selection.
    selection_dir = out_root / "selection"
    if not args.skip_selection:
        cmd = [
            sys.executable,
            "Interpose/select_consistent_video.py",
            "--subset_pose_manifest",
            str(subset_pose_manifest),
            "--out_dir",
            str(selection_dir),
            "--use_bias",
            "--checkpoint",
            str(ckpt),
            "--dust3r_root",
            str(dust3r_root),
            "--image_size",
            str(args.image_size),
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
        ]
        run_cmd(cmd, cwd=repo_root)

    print("\nPipeline complete.")
    print(f"Outputs root: {out_root}")
    print(f"Subset pose manifest: {subset_pose_manifest}")
    print(f"Selection file: {selection_dir / 'selected_per_pair.jsonl'}")


if __name__ == "__main__":
    main()
