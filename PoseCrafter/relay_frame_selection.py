#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tvio
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "PoseCrafter Step 2: from each generated video, select 4 keyframes "
            "(start, start+1, end-1, end), run DUSt3R multiview, and export 4 camera poses."
        )
    )
    p.add_argument(
        "--video_root",
        type=Path,
        required=True,
        help="Root containing pair_xxxxx/.../samples_separate/00_start_sample0.mp4 videos.",
    )
    p.add_argument(
        "--pair_tag",
        type=str,
        default=None,
        help="Optional pair folder name filter (e.g. pair_00000_f71ac346cd0f).",
    )
    p.add_argument(
        "--pair_index",
        type=int,
        default=None,
        help="Optional pair index filter (matches pair_XXXXX_* by index).",
    )
    p.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional variant filter (e.g. v1_A2B_p2_s2). If set with --pair_tag, only one video is processed.",
    )
    p.add_argument(
        "--selected_jsonl",
        type=Path,
        default=None,
        help=(
            "Optional selected_per_pair.jsonl. If provided, one selected variant per pair "
            "is processed instead of all four videos."
        ),
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="DUSt3R checkpoint path.")
    p.add_argument("--dust3r_root", type=Path, default=Path("dust3r"))
    p.add_argument("--image_size", type=int, default=512, choices=(224, 512))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--align_niter", type=int, default=300)
    p.add_argument("--align_lr", type=float, default=0.01)
    p.add_argument("--align_schedule", type=str, default="cosine")
    p.add_argument(
        "--posecrafter_out_root",
        type=Path,
        default=None,
        help=(
            "Root directory for PoseCrafter step2 outputs. "
            "Default: <PoseCrafter>/outputs"
        ),
    )
    p.add_argument(
        "--out_manifest",
        type=Path,
        default=None,
        help="Output JSONL path (default: <posecrafter_out_root>/relay_frame_poses.jsonl).",
    )
    p.add_argument(
        "--out_list_csv",
        type=Path,
        default=None,
        help=(
            "Optional flat per-frame list CSV "
            "(default: <posecrafter_out_root>/relay_frame_pose_list.csv)."
        ),
    )
    p.add_argument(
        "--save_frames",
        action="store_true",
        help="If set, save extracted 4 keyframes under <posecrafter_out_root>/<pair>/four_frames/.",
    )
    return p.parse_args()


def discover_videos(video_root: Path) -> list[Path]:
    return sorted(video_root.glob("**/samples_separate/00_start_sample0.mp4"))


def find_pair_and_variant(video_path: Path) -> tuple[Path, str]:
    # .../<pair_dir>/<variant>/samples_separate/00_start_sample0.mp4
    variant = video_path.parents[1].name
    pair_dir = video_path.parents[2]
    return pair_dir, variant


def filter_videos(
    videos: list[Path],
    pair_tag: str | None,
    pair_index: int | None,
    variant: str | None,
    selected_jsonl: Path | None,
) -> list[Path]:
    # Build lookup first.
    by_key: dict[tuple[str, str], Path] = {}
    for vp in videos:
        pdir, var = find_pair_and_variant(vp)
        by_key[(pdir.name, var)] = vp

    # Highest priority: selected manifest (one video per pair).
    if selected_jsonl is not None:
        rows = [json.loads(l) for l in selected_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        out: list[Path] = []
        for r in rows:
            ptag = Path(r["pair_dir"]).name
            svar = r.get("selected_variant")
            if not svar:
                continue
            key = (ptag, svar)
            if key in by_key:
                out.append(by_key[key])
        return sorted(out)

    # Direct filters.
    out = []
    for vp in videos:
        pdir, var = find_pair_and_variant(vp)
        if pair_tag is not None and pdir.name != pair_tag:
            continue
        if pair_index is not None:
            pfx = f"pair_{pair_index:05d}_"
            if not pdir.name.startswith(pfx):
                continue
        if variant is not None and var != variant:
            continue
        out.append(vp)
    return sorted(out)


def select_four_indices(total_frames: int) -> list[int]:
    if total_frames < 4:
        raise ValueError(f"Video has only {total_frames} frames; need >= 4")
    return [0, 1, total_frames - 2, total_frames - 1]


def extract_four_frames(video_path: Path) -> tuple[list[np.ndarray], list[int], int]:
    frames, _, _ = tvio.read_video(str(video_path), pts_unit="sec")
    # [T,H,W,C], uint8
    if frames.numel() == 0:
        raise ValueError("Decoded video has no frames")
    frames_np = frames.numpy()
    total = int(frames_np.shape[0])
    idxs = select_four_indices(total)
    selected = [frames_np[i] for i in idxs]
    return selected, idxs, total


def save_selected_frames(frame_list: list[np.ndarray], frame_indices: list[int], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    # Keep names aligned with the requested PoseCrafter listing style.
    labels = ["00_start", "01", "02", "03_end"]
    for i, (arr, idx) in enumerate(zip(frame_list, frame_indices)):
        out_path = out_dir / f"{labels[i]}_f{idx:03d}.png"
        Image.fromarray(arr).save(out_path)
        out_paths.append(out_path)
    return out_paths


def run_dust3r_multiview_pose(
    model,
    device: torch.device,
    image_paths: list[Path],
    image_size: int,
    batch_size: int,
    align_niter: int,
    align_lr: float,
    align_schedule: str,
) -> np.ndarray:
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

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)
    scene.compute_global_alignment(init="mst", niter=align_niter, schedule=align_schedule, lr=align_lr)

    # DUSt3R returns c2w in input-image order.
    c2w = scene.get_im_poses().detach().float().cpu().numpy()
    return c2w.astype(np.float64)


def main() -> None:
    args = parse_args()
    video_root = args.video_root.resolve()
    ckpt = args.checkpoint.resolve()
    droot = args.dust3r_root.resolve()
    script_dir = Path(__file__).resolve().parent
    out_root = (
        args.posecrafter_out_root.resolve()
        if args.posecrafter_out_root is not None
        else (script_dir / "outputs")
    )
    out_manifest = (
        args.out_manifest.resolve()
        if args.out_manifest is not None
        else out_root / "relay_frame_poses.jsonl"
    )
    out_list_csv = (
        args.out_list_csv.resolve()
        if args.out_list_csv is not None
        else out_root / "relay_frame_pose_list.csv"
    )
    selected_jsonl = args.selected_jsonl.resolve() if args.selected_jsonl is not None else None

    videos = discover_videos(video_root)
    if not videos:
        raise FileNotFoundError(f"No videos found under {video_root}/**/samples_separate/00_start_sample0.mp4")
    videos = filter_videos(videos, args.pair_tag, args.pair_index, args.variant, selected_jsonl)
    if not videos:
        raise FileNotFoundError("No videos match the provided filters/selection manifest.")
    print(f"[info] videos to process: {len(videos)}")

    if str(droot) not in sys.path:
        sys.path.insert(0, str(droot))
    from dust3r.model import load_model  # pylint: disable=import-outside-toplevel

    device = torch.device(args.device)
    model = load_model(str(ckpt), device)
    model.eval()

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_list_csv.parent.mkdir(parents=True, exist_ok=True)
    frame_pose_rows: list[dict] = []
    with out_manifest.open("w", encoding="utf-8") as fout:
        for vp in videos:
            pair_dir, variant = find_pair_and_variant(vp)
            try:
                frame_list, frame_indices, total_frames = extract_four_frames(vp)
                frame_dir = out_root / pair_dir.name / "four_frames"
                image_paths = save_selected_frames(frame_list, frame_indices, frame_dir)

                c2w = run_dust3r_multiview_pose(
                    model=model,
                    device=device,
                    image_paths=image_paths,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                    align_niter=args.align_niter,
                    align_lr=args.align_lr,
                    align_schedule=args.align_schedule,
                )
                T_wc = [np.linalg.inv(c2w[i]).astype(np.float64) for i in range(4)]

                rec = {
                    "status": "ok",
                    "video_path": str(vp),
                    "pair_dir": str(pair_dir),
                    "variant": variant,
                    "total_video_frames": total_frames,
                    "selected_frame_indices": frame_indices,
                    "selected_frame_paths": [str(p) for p in image_paths],
                    # In selected-frame order: start, after_start, before_end, end
                    "c2w_poses": [c2w[i].tolist() for i in range(4)],
                    "T_wc_poses": [T_wc[i].tolist() for i in range(4)],
                }
                frame_roles = ["start", "after_start", "before_end", "end"]
                for i in range(4):
                    frame_pose_rows.append(
                        {
                            "pair_dir": str(pair_dir),
                            "variant": variant,
                            "video_path": str(vp),
                            "frame_order_idx": i,
                            "frame_role": frame_roles[i],
                            "video_frame_idx": int(frame_indices[i]),
                            "frame_path": str(image_paths[i]),
                            "c2w_pose_json": json.dumps(c2w[i].tolist()),
                            "T_wc_pose_json": json.dumps(T_wc[i].tolist()),
                        }
                    )
                print(f"[ok] {vp}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                rec = {
                    "status": "error",
                    "video_path": str(vp),
                    "pair_dir": str(pair_dir),
                    "variant": variant,
                    "error": str(e),
                }
                print(f"[error] {vp}: {e}")

            fout.write(json.dumps(rec) + "\n")

    with out_list_csv.open("w", encoding="utf-8", newline="") as fcsv:
        fieldnames = [
            "pair_dir",
            "variant",
            "video_path",
            "frame_order_idx",
            "frame_role",
            "video_frame_idx",
            "frame_path",
            "c2w_pose_json",
            "T_wc_pose_json",
        ]
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_pose_rows:
            writer.writerow(row)

    print(f"\nDone. Wrote: {out_manifest}")
    print(f"Wrote frame pose list: {out_list_csv}")


if __name__ == "__main__":
    main()
