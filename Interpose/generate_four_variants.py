#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_generator import (
    resolve_inputs,
    resolve_inputs_from_paths,
    run_dynamiCrafter_interp_inference,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate 4 DynamiCrafter interpolation videos per pair: "
            "A->B (prompt1,seed1), A->B (prompt2,seed2), "
            "B->A (prompt1,seed1), B->A (prompt2,seed2)."
        )
    )
    p.add_argument("--pairs_csv", type=Path, default=None)
    p.add_argument("--data_root", type=Path, default=None)
    p.add_argument("--pair_index", type=int, default=0)
    p.add_argument("--start_image", type=Path, default=None)
    p.add_argument("--end_image", type=Path, default=None)

    p.add_argument("--prompt1", type=str, required=True)
    p.add_argument("--prompt2", type=str, required=True)
    p.add_argument("--seed1", type=int, default=123)
    p.add_argument("--seed2", type=int, default=456)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--cfg_scale", type=float, default=7.5)

    p.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Output root (script will create a pair-specific subfolder).",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional custom pair tag folder name.",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "If a variant already has samples_separate/00_start_sample0.mp4, skip that run. "
            "Useful when resuming a long batch after disconnect."
        ),
    )
    return p.parse_args()


def make_pair_tag(bundle: dict) -> str:
    scene = str(bundle.get("scene", "manual_pair")).replace("/", "_")
    idx = int(bundle.get("pair_index", 0))
    return f"pair_{idx:05d}_{scene}"


def main() -> None:
    args = parse_args()

    # Resolve pair source.
    if args.start_image is not None or args.end_image is not None:
        if args.start_image is None or args.end_image is None:
            print("[error] Provide both --start_image and --end_image.", file=sys.stderr)
            sys.exit(1)
        base_bundle = resolve_inputs_from_paths(args.start_image, args.end_image, args.prompt1)
    else:
        if args.pairs_csv is None or args.data_root is None:
            print(
                "[error] Use either (--pairs_csv and --data_root) or (--start_image and --end_image).",
                file=sys.stderr,
            )
            sys.exit(1)
        base_bundle = resolve_inputs(args.pairs_csv, args.data_root, args.pair_index, args.prompt1)

    pair_tag = args.tag or make_pair_tag(base_bundle)
    pair_root = args.out_root.resolve() / pair_tag
    pair_root.mkdir(parents=True, exist_ok=True)

    variants = [
        ("v0_A2B_p1_s1", "A2B", args.prompt1, args.seed1),
        ("v1_A2B_p2_s2", "A2B", args.prompt2, args.seed2),
        ("v2_B2A_p1_s1", "B2A", args.prompt1, args.seed1),
        ("v3_B2A_p2_s2", "B2A", args.prompt2, args.seed2),
    ]

    manifest = []
    for variant_name, direction, prompt, seed in variants:
        out_dir = pair_root / variant_name
        video_path = out_dir / "samples_separate" / "00_start_sample0.mp4"

        if direction == "A2B":
            path_start = base_bundle["path_start"]
            path_end = base_bundle["path_end"]
        else:
            path_start = base_bundle["path_end"]
            path_end = base_bundle["path_start"]

        bundle = dict(base_bundle)
        bundle["path_start"] = path_start
        bundle["path_end"] = path_end
        bundle["prompt"] = prompt

        if args.skip_existing and video_path.is_file() and video_path.stat().st_size > 0:
            print(f"\n[skip] {variant_name} | existing {video_path}")
        else:
            print(f"\n[run] {variant_name} | {direction} | seed={seed}")
            run_dynamiCrafter_interp_inference(
                bundle=bundle,
                out_dir=out_dir,
                seed=seed,
                ddim_steps=args.ddim_steps,
                frame_stride=args.frame_stride,
                cfg_scale=args.cfg_scale,
            )
        manifest.append(
            {
                "variant": variant_name,
                "direction": direction,
                "seed": seed,
                "prompt": prompt,
                "out_dir": str(out_dir),
                "video_path": str(video_path),
            }
        )

    manifest_path = pair_root / "variants_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. Pair outputs: {pair_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
