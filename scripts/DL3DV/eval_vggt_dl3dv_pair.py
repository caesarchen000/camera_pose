#!/usr/bin/env python3
"""
Evaluate VGGT on a single DL3DV pair, using DUSt3R-style pose metrics.

Pipeline:
  - Load DL3DV GT poses from transforms.json (via load_dl3dv_frames)
  - Read a pair from a DL3DV metrics CSV (rel_a, rel_b) by --pair_idx
  - Load the two images, run VGGT once
  - Convert VGGT's pose encoding to extrinsic / intrinsic
  - Build c2w for GT and VGGT, save as .npy under outputs/
  - Compute rotation / translation-direction errors on the relative pose
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

# Allow running as a standalone script from repo root (or elsewhere).
# Ensures imports like `scripts.DL3DV.*` work even though `scripts/` isn't a package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# This repo vendors the VGGT library under `./vggt/vggt/...`.
# Add `./vggt` to sys.path so imports like `from vggt.models...` work.
_VGGT_VENDOR_ROOT = _REPO_ROOT / "vggt"
if _VGGT_VENDOR_ROOT.is_dir() and str(_VGGT_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_VENDOR_ROOT))

from scripts.DL3DV.eval_dust3r_dl3dv import (
    load_dl3dv_frames,
    geodesic_rotation_error_deg,
    translation_direction_angle_error_deg,
    relative_pose_T,
    T_to_Rt,
)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_pairs_from_csv(metrics_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with metrics_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {metrics_csv}")
    return rows


def build_frame_index(frames):
    """Map (scene_hash, filename) -> index for robust lookup."""
    index = {}
    for i, f in enumerate(frames):
        parts = f.rel_path.split("/")
        scene = parts[0]
        fname = parts[-1]
        index[(scene, fname)] = i
    return index


def lookup_frame_idx(index, rel_path: str) -> int:
    parts = rel_path.split("/")
    scene = parts[0]
    fname = parts[-1]
    return index[(scene, fname)]


def to_homogeneous_4x4(T: np.ndarray) -> np.ndarray:
    """
    Normalize camera transform to homogeneous 4x4.
    Accepts either 3x4 (R|t) or 4x4.
    """
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        out = np.eye(4, dtype=T.dtype)
        out[:3, :] = T
        return out
    raise ValueError(f"Unsupported transform shape {T.shape}; expected (3,4) or (4,4).")


def load_gt_T_wc_from_rel_path(scene_root: Path, rel_path: str) -> np.ndarray:
    """
    Load GT world->camera pose for a CSV rel_path by reading the scene transforms.json.
    This is stricter than filename-only index matching and prevents accidental cross-layout mismatch.
    """
    parts = Path(rel_path).parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected rel_path format: {rel_path}")
    scene_hash = parts[0]
    scene_dir = scene_root / scene_hash
    # rel_path looks like: <scene>/nerfstudio/images_4/frame_xxxxx.png
    # Prefer transforms under the same parent ("nerfstudio" or "colmap"), then fallback.
    preferred_parent = parts[1]
    tf_candidates = [scene_dir / preferred_parent / "transforms.json"]
    if preferred_parent != "nerfstudio":
        tf_candidates.append(scene_dir / "nerfstudio" / "transforms.json")
    if preferred_parent != "colmap":
        tf_candidates.append(scene_dir / "colmap" / "transforms.json")
    tf_candidates.append(scene_dir / "transforms.json")
    tf_path = next((p for p in tf_candidates if p.is_file()), None)
    if tf_path is None:
        raise FileNotFoundError(f"No transforms.json found for scene {scene_hash}")

    with tf_path.open() as f:
        data = json.load(f)
    target_name = Path(rel_path).name
    # Match by basename to handle images/ vs images_4/ path variants in transforms.
    match = None
    for fr in data.get("frames", []):
        fp = fr.get("file_path", "")
        if Path(fp).name == target_name:
            match = fr
            break
    if match is None:
        raise KeyError((scene_hash, target_name))

    c2w = np.array(match["transform_matrix"], dtype=np.float64)
    # Same convention conversion used in eval_dust3r_dl3dv.load_dl3dv_frames
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    T_wc = np.linalg.inv(c2w @ flip)
    return T_wc


def evaluate_one_pair(
    model: VGGT,
    device: torch.device,
    frames,
    index,
    rel_a: str,
    rel_b: str,
    scene_root: Path,
    out_dir: Path,
    out_prefix: str,
    pair_idx: int,
) -> Tuple[float, float]:
    # Always use the exact CSV rel_path first (avoid implicit remapping).
    img_path_a = (scene_root / rel_a).resolve()
    img_path_b = (scene_root / rel_b).resolve()
    if not img_path_a.is_file() or not img_path_b.is_file():
        # Fallbacks for datasets where csv uses nerfstudio/images_4 but files are elsewhere.
        scene_a, fn_a = rel_a.split("/")[0], Path(rel_a).name
        scene_b, fn_b = rel_b.split("/")[0], Path(rel_b).name
        cand_a = [
            scene_root / scene_a / "nerfstudio" / "images_4" / fn_a,
            scene_root / scene_a / "colmap" / "images" / fn_a,
        ]
        cand_b = [
            scene_root / scene_b / "nerfstudio" / "images_4" / fn_b,
            scene_root / scene_b / "colmap" / "images" / fn_b,
        ]
        img_path_a = next((p for p in cand_a if p.is_file()), img_path_a)
        img_path_b = next((p for p in cand_b if p.is_file()), img_path_b)
    if not img_path_a.is_file() or not img_path_b.is_file():
        raise FileNotFoundError(f"Missing pair images: {img_path_a} | {img_path_b}")

    # Load GT strictly from transforms for the exact rel paths.
    T_wc_a = load_gt_T_wc_from_rel_path(scene_root, rel_a)
    T_wc_b = load_gt_T_wc_from_rel_path(scene_root, rel_b)
    c2w_a_gt = np.linalg.inv(T_wc_a)
    c2w_b_gt = np.linalg.inv(T_wc_b)
    gt_c2w_pair = np.stack([c2w_a_gt, c2w_b_gt], axis=0)
    gt_out = out_dir / f"{out_prefix}_pair{pair_idx}_gt_c2w.npy"
    np.save(gt_out, gt_c2w_pair)

    images = load_and_preprocess_images([str(img_path_a), str(img_path_b)]).to(device)
    with torch.no_grad():
        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            images_batched = images[None]
            aggregated_tokens_list, _ = model.aggregator(images_batched)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    T_wc_pred_a = to_homogeneous_4x4(extrinsic[0])
    T_wc_pred_b = to_homogeneous_4x4(extrinsic[1])
    c2w_pred_a = np.linalg.inv(T_wc_pred_a)
    c2w_pred_b = np.linalg.inv(T_wc_pred_b)
    pred_c2w_pair = np.stack([c2w_pred_a, c2w_pred_b], axis=0)
    pred_out = out_dir / f"{out_prefix}_pair{pair_idx}_pred_c2w.npy"
    np.save(pred_out, pred_c2w_pair)

    rel_pr = relative_pose_T(T_wc_pred_a, T_wc_pred_b)
    rel_gt = relative_pose_T(T_wc_a, T_wc_b)
    R_pr, t_pr = T_to_Rt(rel_pr)
    R_gt, t_gt = T_to_Rt(rel_gt)
    rot_err = geodesic_rotation_error_deg(R_pr, R_gt)
    trans_err = translation_direction_angle_error_deg(t_pr, t_gt)
    return rot_err, trans_err


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate VGGT on one or many DL3DV pairs (DUSt3R-style metrics)."
    )
    p.add_argument(
        "--dl3dv_root",
        type=Path,
        default=Path("DL3DV-10K-Sample"),
        help="Root of DL3DV-10K-Sample.",
    )
    p.add_argument(
        "--metrics_csv",
        type=Path,
        default=Path("outputs/dl3dv_metrics_300.csv"),
        help="CSV with rel_a, rel_b, pair_idx (from DUSt3R eval).",
    )
    p.add_argument("--pair_idx", type=int, default=None, help="Single pair index to evaluate.")
    p.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Evaluate first N rows from metrics CSV (overrides --pair_idx).",
    )
    p.add_argument(
        "--scene_root_override",
        type=Path,
        default=None,
        help=(
            "Optional: override the root where images are stored. "
            "If None, dl3dv_root is used."
        ),
    )
    p.add_argument(
        "--vggt_checkpoint",
        type=str,
        default="facebook/VGGT-1B",
        help=(
            'VGGT checkpoint; either HuggingFace id '
            '(e.g. "facebook/VGGT-1B") or a local .pt file path.'
        ),
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs"),
        help="Base output directory.",
    )
    p.add_argument(
        "--out_prefix",
        type=str,
        default="vggt",
        help="Filename prefix for saved .npy outputs.",
    )
    p.add_argument(
        "--npy_subdir",
        type=str,
        default="vggt_pairs_npy",
        help="Subdirectory under out_dir where per-pair .npy files are written.",
    )
    p.add_argument(
        "--save_metrics_csv",
        type=Path,
        default=Path("outputs/vggt_metrics.csv"),
        help="Path to save per-pair VGGT metrics CSV.",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dl3dv_root = args.dl3dv_root.resolve()
    metrics_csv = args.metrics_csv.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_out_dir = out_dir / args.npy_subdir
    npy_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load GT frames (T_wc) from DL3DV transforms.json
    print(f"[info] Loading DL3DV frames from {dl3dv_root} ...")
    frames = load_dl3dv_frames(dl3dv_root)
    index = build_frame_index(frames)

    # Resolve actual image root
    scene_root = args.scene_root_override or dl3dv_root

    print(f"[info] Loading VGGT from {args.vggt_checkpoint} ...")
    if args.vggt_checkpoint.endswith(".pt"):
        model = VGGT()
        state = torch.load(args.vggt_checkpoint, map_location="cpu")
        model.load_state_dict(state)
    else:
        model = VGGT.from_pretrained(args.vggt_checkpoint)
    model.to(device)
    model.eval()

    rows = load_pairs_from_csv(metrics_csv)
    if args.max_pairs is not None:
        rows = rows[: args.max_pairs]
    elif args.pair_idx is not None:
        rows = [r for r in rows if int(r["pair_idx"]) == args.pair_idx]
        if not rows:
            raise RuntimeError(f"pair_idx {args.pair_idx} not found in {metrics_csv}")
    else:
        raise RuntimeError("Provide either --pair_idx or --max_pairs.")

    out_rows = []
    for row in rows:
        pid = int(row["pair_idx"])
        rel_a = row["rel_a"]
        rel_b = row["rel_b"]
        try:
            rot_err, trans_err = evaluate_one_pair(
                model=model,
                device=device,
                frames=frames,
                index=index,
                rel_a=rel_a,
                rel_b=rel_b,
                scene_root=scene_root,
                out_dir=npy_out_dir,
                out_prefix=args.out_prefix,
                pair_idx=pid,
            )
        except Exception as e:
            print(f"[skip] pair_idx={pid} failed: {e}")
            continue

        out_rows.append(
            {
                "pair_idx": pid,
                "rel_a": rel_a,
                "rel_b": rel_b,
                "vggt_rot_err_deg": rot_err,
                "vggt_trans_dir_err_deg": trans_err,
                "dust3r_rot_err_deg": float(row.get("rot_err_deg", "nan")),
                "dust3r_trans_dir_err_deg": float(row.get("trans_dir_err_deg", "nan")),
            }
        )
        print(
            f"[ok] pair_idx={pid} rot_err={rot_err:.4f} "
            f"trans_dir_err={trans_err:.4f}"
        )

    if not out_rows:
        raise RuntimeError("No pairs evaluated successfully.")

    save_csv = args.save_metrics_csv.resolve()
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    with save_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_idx",
                "rel_a",
                "rel_b",
                "vggt_rot_err_deg",
                "vggt_trans_dir_err_deg",
                "dust3r_rot_err_deg",
                "dust3r_trans_dir_err_deg",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    rot_arr = np.array([r["vggt_rot_err_deg"] for r in out_rows], dtype=np.float64)
    trans_arr = np.array([r["vggt_trans_dir_err_deg"] for r in out_rows], dtype=np.float64)
    dust_rot_arr = np.array([r["dust3r_rot_err_deg"] for r in out_rows], dtype=np.float64)
    dust_trans_arr = np.array([r["dust3r_trans_dir_err_deg"] for r in out_rows], dtype=np.float64)
    print(f"\nSaved VGGT per-pair metrics: {save_csv}")
    print(f"Saved per-pair npy files under: {npy_out_dir}")
    print(f"Evaluated pairs: {len(out_rows)}")
    print(f"Mean rot err (deg): {float(np.nanmean(rot_arr)):.6f}")
    print(f"Mean trans err (deg): {float(np.nanmean(trans_arr)):.6f}")
    print(f"Mean DUSt3R rot err on same pairs (deg): {float(np.nanmean(dust_rot_arr)):.6f}")
    print(f"Mean DUSt3R trans err on same pairs (deg): {float(np.nanmean(dust_trans_arr)):.6f}")


if __name__ == "__main__":
    main()

