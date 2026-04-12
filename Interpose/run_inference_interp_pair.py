#!/usr/bin/env python3
"""
Run DynamiCrafter scripts/evaluation/inference.py (interp) for one CSV row.

inference.py expects prompt_dir to contain:
  - One *.txt file with one prompt line per sample (here: a single line).
  - Images *.png/*.jpg sorted by name: for interp, sample i uses file_list[2*i] and file_list[2*i+1].

We copy the row's rel_a / rel_b into that folder with deterministic names so glob order is correct.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage prompt_dir and run DynamiCrafter inference.py --interp.")
    p.add_argument("--pairs_csv", type=Path, required=True)
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--pair_index", type=int, default=0, help="0-based data row after header.")
    p.add_argument("--prompt", type=str, required=True, help="Single-line text prompt for inference.py.")
    p.add_argument(
        "--savedir",
        type=Path,
        default=None,
        help="Passed to inference.py --savedir (default: <repo>/Interpose/out_inference/pair_NNNNN).",
    )
    p.add_argument("--seed", type=int, default=12306)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--cfg", type=float, default=7.5, dest="unconditional_guidance_scale")
    p.add_argument("--dry_run", action="store_true", help="Only create prompt_dir; do not run inference.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dyn = root / "DynamiCrafter"
    if not dyn.is_dir():
        print(f"[error] missing {dyn}", file=sys.stderr)
        sys.exit(1)

    args.pairs_csv = args.pairs_csv.resolve()
    args.data_root = args.data_root.resolve()

    row_data = None
    with args.pairs_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not {"rel_a", "rel_b"}.issubset(reader.fieldnames or []):
            print("[error] CSV needs rel_a, rel_b columns.", file=sys.stderr)
            sys.exit(1)
        for i, row in enumerate(reader):
            if i == args.pair_index:
                row_data = row
                break
    if row_data is None:
        print(f"[error] no row at pair_index={args.pair_index}", file=sys.stderr)
        sys.exit(1)

    rel_a = row_data["rel_a"]
    rel_b = row_data["rel_b"]
    scene = row_data.get("scene", "scene")
    img_a = args.data_root / rel_a
    img_b = args.data_root / rel_b
    if not img_a.is_file() or not img_b.is_file():
        print(f"[error] missing images:\n  {img_a}\n  {img_b}", file=sys.stderr)
        sys.exit(1)

    if args.savedir is None:
        savedir = root / "Interpose" / "out_inference" / f"pair_{args.pair_index:05d}_{scene.replace('/', '_')}"
    else:
        savedir = args.savedir.resolve()
    prompt_dir = savedir / "prompt_pack"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Sorted glob: 00_... then 01_...
    dst0 = prompt_dir / "00_start.png"
    dst1 = prompt_dir / "01_end.png"
    shutil.copy2(img_a, dst0)
    shutil.copy2(img_b, dst1)
    (prompt_dir / "prompt.txt").write_text(args.prompt.strip() + "\n", encoding="utf-8")

    print(f"prompt_dir: {prompt_dir}")
    print(f"savedir:    {savedir}")

    if args.dry_run:
        return

    cmd = [
        sys.executable,
        "scripts/evaluation/inference.py",
        "--seed",
        str(args.seed),
        "--ckpt_path",
        "checkpoints/dynamicrafter_512_interp_v1/model.ckpt",
        "--config",
        "configs/inference_512_v1.0.yaml",
        "--savedir",
        str(savedir),
        "--n_samples",
        "1",
        "--bs",
        "1",
        "--height",
        "320",
        "--width",
        "512",
        "--unconditional_guidance_scale",
        str(args.unconditional_guidance_scale),
        "--ddim_steps",
        str(args.ddim_steps),
        "--ddim_eta",
        "1.0",
        "--prompt_dir",
        str(prompt_dir),
        "--text_input",
        "--video_length",
        "16",
        "--frame_stride",
        str(args.frame_stride),
        "--timestep_spacing",
        "uniform_trailing",
        "--guidance_rescale",
        "0.7",
        "--perframe_ae",
        "--interp",
    ]
    print("Running:", " ".join(cmd))
    print(
        "\nTip: manual runs must pass the full flag set; use --timestep_spacing (not --interpep_spacing) "
        "and an absolute --prompt_dir that exists.",
        file=sys.stderr,
    )
    r = subprocess.run(cmd, cwd=str(dyn))
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"\nDone. Videos under: {savedir / 'samples_separate'} (per inference.py)")


if __name__ == "__main__":
    main()
