#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run DUSt3R on InterPose subset folders (2 input + 3 generated), "
            "and export relative pose A->B for each subset."
        )
    )
    p.add_argument(
        "--subset_root",
        type=Path,
        required=True,
        help="Root containing pair folders with subset/<variant>/subset_xx_* directories.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to DUSt3R checkpoint (.pth).",
    )
    p.add_argument(
        "--dust3r_root",
        type=Path,
        default=Path("dust3r"),
        help="Path to local DUSt3R repo root.",
    )
    p.add_argument("--image_size", type=int, default=512, choices=(224, 512))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--align_niter", type=int, default=300)
    p.add_argument("--align_lr", type=float, default=0.01)
    p.add_argument("--align_schedule", type=str, default="cosine")
    p.add_argument(
        "--out_manifest",
        type=Path,
        default=None,
        help="Output JSONL path (default: <subset_root>/dust3r_subset_poses.jsonl).",
    )
    return p.parse_args()


def discover_subset_dirs(subset_root: Path) -> list[Path]:
    # Expected layout from generate_subsets.py:
    # pair_xxx/subset/<variant>/subset_00_random/...
    return sorted([p for p in subset_root.glob("**/subset_*") if p.is_dir()])


def load_subset_images(subset_dir: Path) -> list[Path]:
    start = subset_dir / "00_input_start.png"
    end = subset_dir / "01_input_end.png"
    gens = sorted(subset_dir.glob("[0-9][0-9]_gen_f*.png"))
    if not start.is_file() or not end.is_file() or len(gens) < 3:
        return []
    # Exactly 5 images: start, end, first three generated.
    return [start, end, gens[0], gens[1], gens[2]]


def run_dust3r_multiview_pose(
    model,
    device: torch.device,
    image_paths: list[Path],
    image_size: int,
    batch_size: int,
    align_niter: int,
    align_lr: float,
    align_schedule: str,
) -> tuple[np.ndarray, np.ndarray]:
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    square_ok = bool(getattr(model, "square_ok", False))
    patch_size = int(getattr(model, "patch_size", 16))
    imgs = load_images(
        [str(p) for p in image_paths],
        size=image_size,
        verbose=False,
        patch_size=patch_size,
        square_ok=square_ok,
    )

    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=batch_size, verbose=False)

    # Multi-image mode per DUSt3R docs.
    mode = GlobalAlignerMode.PointCloudOptimizer if len(image_paths) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=False)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init="mst", niter=align_niter, schedule=align_schedule, lr=align_lr)

    # Scene poses are camera-to-world, one per input image in order.
    c2w = scene.get_im_poses().detach().float().cpu().numpy()
    # Convert to world-to-camera to match existing evaluation conventions.
    T_wc_A = np.linalg.inv(c2w[0]).astype(np.float64)
    T_wc_B = np.linalg.inv(c2w[1]).astype(np.float64)
    return T_wc_A, T_wc_B


def to_pose_dict(T_wc_A: np.ndarray, T_wc_B: np.ndarray) -> dict:
    T_rel = T_wc_B @ np.linalg.inv(T_wc_A)
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]
    return {
        "T_wc_A": T_wc_A.tolist(),
        "T_wc_B": T_wc_B.tolist(),
        "T_rel": T_rel.tolist(),
        "R_rel": R_rel.tolist(),
        "t_rel": t_rel.tolist(),
    }


def main() -> None:
    args = parse_args()
    subset_root = args.subset_root.resolve()
    ckpt = args.checkpoint.resolve()
    droot = args.dust3r_root.resolve()
    out_manifest = (
        args.out_manifest.resolve()
        if args.out_manifest is not None
        else subset_root / "dust3r_subset_poses.jsonl"
    )

    if str(droot) not in sys.path:
        sys.path.insert(0, str(droot))
    from dust3r.model import load_model  # pylint: disable=import-outside-toplevel

    device = torch.device(args.device)
    model = load_model(str(ckpt), device)
    model.eval()

    subset_dirs = discover_subset_dirs(subset_root)
    if not subset_dirs:
        raise FileNotFoundError(f"No subset directories found under {subset_root}")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as fout:
        for sdir in subset_dirs:
            image_paths = load_subset_images(sdir)
            if not image_paths:
                print(f"[skip] invalid subset folder: {sdir}")
                continue
            try:
                T_wc_A, T_wc_B = run_dust3r_multiview_pose(
                    model=model,
                    device=device,
                    image_paths=image_paths,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                    align_niter=args.align_niter,
                    align_lr=args.align_lr,
                    align_schedule=args.align_schedule,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                rec = {
                    "subset_dir": str(sdir),
                    "image_paths": [str(p) for p in image_paths],
                    "status": "error",
                    "error": str(e),
                }
                fout.write(json.dumps(rec) + "\n")
                print(f"[error] {sdir}: {e}")
                continue

            pose = to_pose_dict(T_wc_A, T_wc_B)
            rec = {
                "subset_dir": str(sdir),
                "image_paths": [str(p) for p in image_paths],
                "status": "ok",
                **pose,
            }
            fout.write(json.dumps(rec) + "\n")
            print(f"[ok] {sdir}")

    print(f"\nDone. Wrote: {out_manifest}")


if __name__ == "__main__":
    main()
