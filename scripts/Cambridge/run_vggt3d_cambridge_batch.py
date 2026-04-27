#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VGGT4D_ROOT = _REPO_ROOT / "VGGT4D"

import sys

if str(_VGGT4D_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT4D_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt4d.utils.model_utils import inference  # noqa: E402
from vggt4d.utils.store import save_intrinsic_txt, save_tum_poses  # noqa: E402

def _load_demo_symbols(repo_root: Path):
    demo_path = repo_root / "vggt3d" / "demo_vggt4d_pipeline.py"
    spec = importlib.util.spec_from_file_location("vggt3d_demo_vggt4d_pipeline", demo_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_model, module.process_scene


PAIR_RE = re.compile(r"^pair_(\d{5})_(.+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full vggt3d pipeline on Cambridge Interpose videos: mask -> masked pose."
    )
    p.add_argument(
        "--interpose_root",
        type=Path,
        default=Path("Interpose/out_cambridge_selp_yaw50_65_286pairs"),
        help="Interpose output root containing pair_XXXXX_* folders.",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=Path("vggt3d/cambridge_interpose"),
        help="Root for generated masks/poses and temporary frame folders.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("VGGT4D/ckpts/model_tracker_fixed_e20.pt"),
        help="VGGT4D checkpoint path.",
    )
    p.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu.")
    p.add_argument("--start_index", type=int, default=0, help="Start pair_idx (inclusive).")
    p.add_argument("--end_index", type=int, default=-1, help="End pair_idx (inclusive), -1 means all.")
    p.add_argument("--max_pairs", type=int, default=0, help="Optional cap on processed pairs, 0 means all.")
    p.add_argument(
        "--mask_threshold",
        type=float,
        default=None,
        help="Optional manual threshold for mask binarization; forwarded to demo pipeline.",
    )
    p.add_argument("--skip_existing", action="store_true", help="Skip pairs with existing extrinsic_rt.txt.")
    p.add_argument("--keep_frames", action="store_true", help="Keep extracted frame folders.")
    p.add_argument(
        "--reuse_masks",
        action="store_true",
        help="Reuse existing masks under out_root/masks/<pair_name> and skip mask generation.",
    )
    p.add_argument(
        "--pose_backend",
        type=str,
        default="vggt4d",
        choices=["vggt4d", "vggt3d"],
        help="Backend for pose inference after mask generation.",
    )
    p.add_argument(
        "--use_selected_variants",
        action="store_true",
        help="Use selected variant per pair from selected_per_pair.jsonl instead of fixed v0.",
    )
    p.add_argument(
        "--selected_jsonl",
        type=Path,
        default=None,
        help="Path to selected_per_pair.jsonl (required when --use_selected_variants).",
    )
    return p.parse_args()


def _extract_pair_idx(pair_dir_name: str) -> int:
    m = PAIR_RE.match(pair_dir_name)
    if not m:
        return -1
    return int(m.group(1))


def _discover_pairs(interpose_root: Path) -> list[Path]:
    pairs = []
    for p in sorted(interpose_root.glob("pair_*")):
        if p.is_dir() and _extract_pair_idx(p.name) >= 0:
            pairs.append(p)
    return pairs


def _load_selected_variant_map(selected_jsonl: Path) -> dict[str, str]:
    if not selected_jsonl.is_file():
        raise FileNotFoundError(f"selected_jsonl not found: {selected_jsonl}")
    out: dict[str, str] = {}
    with selected_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pair_name = Path(str(rec["pair_dir"])).name
            selected_variant = str(rec.get("selected_variant", "")).strip()
            if pair_name and selected_variant:
                out[pair_name] = selected_variant
    if not out:
        raise RuntimeError(f"No selected variants loaded from {selected_jsonl}")
    return out


def _extract_frames_cv2(video_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_paths: list[Path] = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_path = out_dir / f"{idx:04d}.png"
        cv2.imwrite(str(frame_path), frame_bgr)
        frame_paths.append(frame_path)
        idx += 1
    cap.release()
    if not frame_paths:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frame_paths


def _load_binary_masks(mask_dir: Path, n_frames: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    mask_paths = sorted(mask_dir.glob("dynamic_mask_*.png"))
    if len(mask_paths) != n_frames:
        raise ValueError(
            f"Mask/frame count mismatch in {mask_dir}: masks={len(mask_paths)} frames={n_frames}"
        )
    masks = []
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Failed to read mask: {p}")
        m = (m > 0).astype(np.float32)
        m_t = torch.from_numpy(m)[None, None]  # [1,1,H,W]
        m_t = F.interpolate(m_t, size=(h, w), mode="nearest")[0, 0]
        masks.append(m_t)
    dyn_masks = (torch.stack(masks, dim=0).to(device) > 0.5)  # [S,H,W]
    # Keep first/last frames unmasked to preserve endpoint camera anchors.
    if dyn_masks.shape[0] >= 1:
        dyn_masks[0] = False
    if dyn_masks.shape[0] >= 2:
        dyn_masks[-1] = False
    return dyn_masks


def _run_masked_pose_for_scene(
    model,
    device: torch.device,
    frame_paths: list[Path],
    video_path: Path,
    mask_dir: Path,
    pose_out_dir: Path,
    pose_backend: str,
) -> None:
    if pose_backend == "vggt3d":
        script_path = _REPO_ROOT / "vggt3d" / "masked_pose_inference_vggt3d.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--video_path",
            str(video_path),
            "--mask_dir",
            str(mask_dir),
            "--out_dir",
            str(pose_out_dir),
            "--mask_layers",
            "5",
            "--device",
            str(device),
        ]
        subprocess.run(cmd, check=True)
        return

    images = load_and_preprocess_images([str(p) for p in frame_paths]).to(device)  # [S,3,H,W]
    n_frames, _, h_img, w_img = images.shape
    dyn_masks = _load_binary_masks(mask_dir, n_frames, h_img, w_img, device)
    preds, _, _, _ = inference(model, images, dyn_masks=dyn_masks)

    pose_out_dir.mkdir(parents=True, exist_ok=True)
    save_intrinsic_txt(pose_out_dir, preds["intrinsic"])
    save_tum_poses(pose_out_dir, preds["cam2world"])
    np.save(pose_out_dir / "cam2world.npy", preds["cam2world"])
    np.save(pose_out_dir / "extrinsic.npy", preds["extrinsic"])
    extr = np.asarray(preds["extrinsic"], dtype=np.float64)
    extr_rt = extr[:, :3, :4].reshape(extr.shape[0], 12)
    np.savetxt(pose_out_dir / "extrinsic_rt.txt", extr_rt, fmt="%.8f")


def main() -> None:
    args = parse_args()
    interpose_root = args.interpose_root.resolve()
    out_root = args.out_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not interpose_root.is_dir():
        raise FileNotFoundError(f"interpose_root not found: {interpose_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    load_model_fn, process_scene_fn = _load_demo_symbols(_REPO_ROOT)
    model = load_model_fn(checkpoint, device)
    selected_variant_map: dict[str, str] = {}
    if args.use_selected_variants:
        selected_jsonl = (
            args.selected_jsonl.resolve()
            if args.selected_jsonl is not None
            else (interpose_root / "selection" / "selected_per_pair.jsonl").resolve()
        )
        selected_variant_map = _load_selected_variant_map(selected_jsonl)
        print(f"[info] loaded selected variants: {len(selected_variant_map)} from {selected_jsonl}")

    masks_root = out_root / "masks"
    poses_root = out_root / "poses"
    frames_root = out_root / "_frames"
    if not args.reuse_masks:
        masks_root.mkdir(parents=True, exist_ok=True)
    poses_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(interpose_root)
    processed = 0
    for pair_dir in pairs:
        pair_idx = _extract_pair_idx(pair_dir.name)
        if pair_idx < args.start_index:
            continue
        if args.end_index >= 0 and pair_idx > args.end_index:
            continue
        if args.max_pairs > 0 and processed >= args.max_pairs:
            break

        pair_name = pair_dir.name
        variant_name = "v0_A2B_p1_s1"
        if args.use_selected_variants:
            variant_name = selected_variant_map.get(pair_name, "")
            if not variant_name:
                print(f"[skip] {pair_name}: missing selected_variant")
                continue
        video_path = pair_dir / variant_name / "samples_separate" / "00_start_sample0.mp4"
        if not video_path.is_file():
            print(f"[skip] {pair_name}: missing video for variant={variant_name}")
            continue

        pose_out_dir = poses_root / pair_name
        if args.skip_existing and (pose_out_dir / "extrinsic_rt.txt").is_file():
            print(f"[skip] {pair_name}: pose output exists")
            continue

        scene_dir = frames_root / pair_name
        scene_input_root = frames_root / f"{pair_name}_input"
        scene_input = scene_input_root / pair_name
        scene_input.mkdir(parents=True, exist_ok=True)

        print(f"[run] {pair_name} variant={variant_name}")
        frame_paths = _extract_frames_cv2(video_path, scene_dir)
        # demo pipeline expects input_root/scene_name/*.png
        # Create symlinks to avoid duplicating frame files.
        for fp in frame_paths:
            dst = scene_input / fp.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(fp)

        mask_scene_dir = masks_root / pair_name
        if args.reuse_masks:
            if not mask_scene_dir.is_dir():
                print(f"[skip] {pair_name}: missing existing mask dir {mask_scene_dir}")
                continue
            mask_files = sorted(mask_scene_dir.glob("dynamic_mask_*.png"))
            if not mask_files:
                print(f"[skip] {pair_name}: no dynamic_mask_*.png in {mask_scene_dir}")
                continue
        else:
            process_scene_fn(
                scene_input,
                mask_scene_dir,
                model,
                device,
                mask_threshold=args.mask_threshold,
            )
        _run_masked_pose_for_scene(
            model,
            device,
            frame_paths,
            video_path,
            mask_scene_dir,
            pose_out_dir,
            args.pose_backend,
        )
        processed += 1
        print(f"[done] {pair_name}")

        if not args.keep_frames:
            for p in scene_input.glob("*.png"):
                p.unlink(missing_ok=True)
            for p in scene_dir.glob("*.png"):
                p.unlink(missing_ok=True)

    print(f"[complete] processed={processed} out_root={out_root}")


if __name__ == "__main__":
    main()

