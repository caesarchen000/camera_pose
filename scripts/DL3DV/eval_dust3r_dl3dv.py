#!/usr/bin/env python3
"""
DL3DV-10K-Sample: select same-scene image pairs by geodesic rotation angle
and evaluate DUSt3R relative-pose accuracy with the same metrics used on KingsCollege.

Pipeline (all in one script):
  Step 1 — load GT poses from every scene's colmap/transforms.json
  Step 2 — select same-scene pairs whose geodesic rotation angle falls in [--rot_min, --rot_max],
            sampled randomly from the valid pool (uniform random, reproducible via --seed)
  Step 3 — run DUSt3R on each pair and report pairwise relative-pose metrics

Pair selection criterion — geodesic rotation angle:
  R_rel = R_b @ R_a.T
  theta = arccos(clip((trace(R_rel) - 1) / 2, -1, 1))

  This is the full SO(3) distance between two camera orientations, capturing
  yaw + pitch + roll together. It is the same formula used in the evaluation
  metric (geodesic_rotation_error_deg), making selection and measurement consistent.
  Unlike yaw_diff (2D projection), it correctly handles cameras that differ
  primarily in pitch or roll — which is common in indoor DL3DV scenes.

Metrics (identical to eval_dust3r_step3.py):
  MRE_deg, median_RE_deg
  R_acc_5 / R_acc_15 / R_acc_30
  AUC30_rot
  MTE_deg, median_TE_deg   (translation direction angle)
  t_acc_5 / t_acc_15 / t_acc_30
  AUC30_trans

Coordinate convention:
  DL3DV transforms.json stores camera-to-world (c2w) 4x4 matrices.
  We invert to world-to-camera (T_wc) to match the KingsCollege convention.

Usage example:
  python scripts/DL3DV/eval_dust3r_dl3dv.py \\
      --dl3dv_root DL3DV-10K-Sample \\
      --checkpoint checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
      --dust3r_root dust3r \\
      --yaw_min 50 --yaw_max 90 --pitch_max 90 \\
      --max_pairs 300 \\
      --seed 0 \\
      --device cuda \\
      --save_pairs_csv  outputs/dl3dv_pairs_50_90.csv \\
      --save_pairs_txt  outputs/dl3dv_pairs_50_90.txt \\
      --save_metrics_csv outputs/dl3dv_metrics_50_90.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FramePose:
    rel_path: str     # relative to dl3dv_root, e.g. "<hash>/colmap/images/frame_00001.png"
    scene_hash: str   # 64-char scene hash — used as "sequence" for same-scene gating
    T_wc: np.ndarray  # 4x4 world->camera


# ---------------------------------------------------------------------------
# Loading GT poses from DL3DV transforms.json
# ---------------------------------------------------------------------------

def load_dl3dv_frames(dl3dv_root: Path) -> List[FramePose]:
    """
    Walk every <hash>/colmap/transforms.json or <hash>/nerfstudio/transforms.json
    under dl3dv_root.
    Each frame entry has:
      "file_path": "images/frame_XXXXX.png"
      "transform_matrix": 4x4 camera-to-world (c2w)

    We store world-to-camera (T_wc = inv(c2w)) to match KingsCollege convention.
    """
    frames: List[FramePose] = []

    def resolve_rel_path(scene_hash: str, parent_name: str, file_path: str) -> str:
        """
        Resolve frame path robustly across DL3DV layouts.
        Some nerfstudio exports store file_path as "images/..." while only
        downsampled folders like "images_4/..." are present on disk.
        """
        norm = file_path.lstrip("./")
        candidates = [norm]
        if norm.startswith("images/"):
            suffix = norm[len("images/") :]
            for pfx in ("images_4", "images_2", "images_8", "images_16"):
                candidates.append(f"{pfx}/{suffix}")
        for cand in candidates:
            rel = f"{scene_hash}/{parent_name}/{cand}"
            if (dl3dv_root / rel).is_file():
                return rel
        # Keep original path if files are not available yet.
        return f"{scene_hash}/{parent_name}/{norm}"
    tf_files = sorted(dl3dv_root.glob("*/colmap/transforms.json"))
    if not tf_files:
        tf_files = sorted(dl3dv_root.glob("*/nerfstudio/transforms.json"))
    if not tf_files:
        raise FileNotFoundError(
            f"No transforms.json found under {dl3dv_root}. "
            "Make sure DL3DV-10K-Sample is downloaded correctly."
        )
    for tf_path in tf_files:
        # Works for both:
        #   <hash>/colmap/transforms.json
        #   <hash>/nerfstudio/transforms.json
        scene_hash = tf_path.parts[-3]
        parent_name = tf_path.parts[-2]  # "colmap" or "nerfstudio"
        try:
            d = json.loads(tf_path.read_text())
        except Exception as e:
            print(f"[warn] Could not parse {tf_path}: {e}")
            continue

        for fr in d.get("frames", []):
            file_path = fr.get("file_path", "")
            tm = fr.get("transform_matrix")
            if not file_path or tm is None:
                continue
            c2w = np.array(tm, dtype=np.float64)
            if c2w.shape != (4, 4):
                continue
            try:
                # DL3DV transforms.json uses NeRF/OpenGL c2w convention:
                #   X right, Y up, Z backward (camera looks along -Z).
                # DUSt3R uses OpenCV convention:
                #   X right, Y down, Z forward (camera looks along +Z).
                # Convert: flip Y and Z columns of c2w, then invert.
                FLIP = np.diag([1., -1., -1., 1.])
                T_wc = np.linalg.inv(c2w @ FLIP)
            except np.linalg.LinAlgError:
                continue

            rel_path = resolve_rel_path(scene_hash, parent_name, file_path)
            frames.append(FramePose(rel_path=rel_path, scene_hash=scene_hash, T_wc=T_wc))

    return frames


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def camera_center(T_wc: np.ndarray) -> np.ndarray:
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    return -R.T @ t


def camera_forward_world(T_wc: np.ndarray) -> np.ndarray:
    """Optical axis (+Z_cam) in world coordinates."""
    R = T_wc[:3, :3]
    v = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def camera_up_world(T_wc: np.ndarray) -> np.ndarray:
    """Camera up direction in world coordinates (OpenCV camera has +Y down)."""
    R = T_wc[:3, :3]
    v_down = R.T @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v_up = -v_down
    return v_up / (np.linalg.norm(v_up) + 1e-12)


def geodesic_angle_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """
    Full SO(3) rotation angle between two camera poses (world->camera).
    R_rel = R_b @ R_a.T  (rotation from camera A frame to camera B frame)
    theta = arccos((trace(R_rel) - 1) / 2)
    Range: [0, 180] degrees.
    """
    R_a = T_a[:3, :3]
    R_b = T_b[:3, :3]
    R_rel = R_b @ R_a.T
    c = float(np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def forward_angle_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """Angle between the two cameras' optical axes in 3D (used for ranking)."""
    fa = camera_forward_world(T_a)
    fb = camera_forward_world(T_b)
    d = float(np.clip(np.dot(fa, fb), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def yaw_diff_deg_wrt_up(f0: np.ndarray, f1: np.ndarray, up_ref: np.ndarray) -> float:
    """Yaw diff as angle between forward vectors projected onto plane orthogonal to up_ref."""
    p0 = f0 - np.dot(f0, up_ref) * up_ref
    p1 = f1 - np.dot(f1, up_ref) * up_ref
    n0 = float(np.linalg.norm(p0))
    n1 = float(np.linalg.norm(p1))
    if n0 < 1e-9 or n1 < 1e-9:
        return 0.0
    p0 /= n0
    p1 /= n1
    d = float(np.clip(np.dot(p0, p1), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def pitch_deg_wrt_up(f: np.ndarray, up_ref: np.ndarray) -> float:
    """Pitch/elevation angle wrt horizontal plane defined by up_ref."""
    s = float(np.clip(np.dot(f, up_ref), -1.0, 1.0))
    return float(np.degrees(np.arcsin(s)))


def angle_between_deg(v0: np.ndarray, v1: np.ndarray) -> float:
    d = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def relative_pose_T(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    return T_b @ np.linalg.inv(T_a)


def T_to_Rt(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return T[:3, :3].copy(), T[:3, 3].copy()


# ---------------------------------------------------------------------------
# Metrics (identical to eval_dust3r_step3.py)
# ---------------------------------------------------------------------------

def geodesic_rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_err = R_pred @ R_gt.T
    c = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def translation_direction_angle_error_deg(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    Translation direction angle error matching paper formula 4:
      dist_t(t1, t2) = arccos(|t1/||t1|| · t2/||t2|||)

    The absolute value makes this sign-agnostic, which is correct for two-view
    pose estimation where the sign of translation is ambiguous.
    """
    np_pred = np.linalg.norm(t_pred)
    np_gt = np.linalg.norm(t_gt)
    if np_gt < 1e-9:
        return float("nan")
    if np_pred < 1e-9:
        return 180.0
    v0 = t_pred / np_pred
    v1 = t_gt / np_gt
    d = float(np.clip(abs(np.dot(v0, v1)), 0.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def acc_at_deg(errors_deg: np.ndarray, thr: float) -> float:
    m = np.isfinite(errors_deg)
    if not m.any():
        return float("nan")
    return float((errors_deg[m] <= thr).mean())


def auc30(errors_deg: np.ndarray) -> float:
    """Area under the accuracy curve from 0° to 30° (thresholds 0,1,...,30)."""
    m = np.isfinite(errors_deg)
    if not m.any():
        return float("nan")
    x = np.sort(errors_deg[m])
    thrs = np.arange(0.0, 30.0 + 1e-9, 1.0)  # 0..30, matching paper wording
    acc = np.array([(x <= t).mean() for t in thrs], dtype=np.float64)
    return float(acc.mean())


def auc30_paper(rot_errs: np.ndarray, trans_errs: np.ndarray) -> float:
    """Paper-style final AUC30: min(AUC30_rot, AUC30_trans)."""
    a_rot = auc30(rot_errs)
    a_trans = auc30(trans_errs)
    if not np.isfinite(a_rot) and not np.isfinite(a_trans):
        return float("nan")
    if not np.isfinite(a_rot):
        return float(a_trans)
    if not np.isfinite(a_trans):
        return float(a_rot)
    return float(min(a_rot, a_trans))


def quick_overlap_proxy(path_a: Path, path_b: Path, max_side: int = 640) -> float:
    """
    Lightweight overlap proxy using ORB/SIFT matching + fundamental matrix RANSAC.
    Returns inlier ratio in [0,1]. Lower usually means lower overlap.
    Returns NaN if overlap cannot be estimated reliably.
    """
    import cv2

    def read_gray(p: Path) -> np.ndarray:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"Could not read image: {p}")
        h, w = im.shape[:2]
        s = float(max(h, w))
        if s > float(max_side):
            scale = float(max_side) / s
            im = cv2.resize(
                im,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return im

    a = read_gray(path_a)
    b = read_gray(path_b)
    sift_ctor = getattr(cv2, "SIFT_create", None)
    if callable(sift_ctor):
        det = cv2.SIFT_create(nfeatures=3000)
        norm = cv2.NORM_L2
    else:
        det = cv2.ORB_create(nfeatures=4000, fastThreshold=5)
        norm = cv2.NORM_HAMMING
    kpa, da = det.detectAndCompute(a, None)
    kpb, db = det.detectAndCompute(b, None)
    if da is None or db is None or len(kpa) < 8 or len(kpb) < 8:
        return float("nan")
    bf = cv2.BFMatcher(norm, crossCheck=False)
    knn = bf.knnMatch(da, db, k=2)
    good = []
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < 0.8 * n.distance:
            good.append(m)
    if len(good) < 8:
        return float("nan")
    pts_a = np.float32([kpa[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 2.0, 0.99)
    if F is None or mask is None:
        return float("nan")
    inliers = int(mask.ravel().sum())
    return float(inliers) / float(max(1, len(good)))


# ---------------------------------------------------------------------------
# Pair selection: same scene, yaw range + pitch (up/down) constraint
# ---------------------------------------------------------------------------

def select_pairs(
    frames: List[FramePose],
    dl3dv_root: Path,
    yaw_min: float,
    yaw_max: float,
    pitch_max: float,
    min_forward_angle: float,
    min_baseline: float,
    max_look_to_mid_angle: float | None,
    max_overlap_ratio: float | None,
    overlap_check_max_side: int,
    overlap_prefilter_factor: int,
    max_pairs: int,
    max_pairs_per_scene: int,
    exclude_scene_prefixes: List[str],
    seed: int = 0,
) -> List[Tuple[int, int, float, float, float, float, float]]:
    """
    Returns list of:
      (i, j, yaw_diff_deg, pitch_diff_deg, rot_deg, forward_angle_deg, baseline_m)

    Filter: same scene hash AND:
      - yaw difference in [yaw_min, yaw_max]
      - pitch difference <= pitch_max
      - forward-angle >= min_forward_angle
      - baseline >= min_baseline
      - (optional) both cameras look toward pair midpoint
      - (optional) overlap_proxy <= max_overlap_ratio
    Sample: uniform random from the valid pool (reproducible via seed),
            so pair difficulty is spread by sampling, not hard-sorted truncation.
    """
    def scene_excluded(scene_hash: str) -> bool:
        return any(scene_hash.startswith(p) for p in exclude_scene_prefixes)

    def cap_per_scene(
        rows: List[Tuple[int, int, float, float, float, float, float]]
    ) -> List[Tuple[int, int, float, float, float, float, float]]:
        if max_pairs_per_scene <= 0:
            return rows
        out: List[Tuple[int, int, float, float, float, float, float]] = []
        by_scene_count: Dict[str, int] = {}
        for row in rows:
            i, _, _, _, _, _, _ = row
            scene = frames[i].scene_hash
            c = by_scene_count.get(scene, 0)
            if c >= max_pairs_per_scene:
                continue
            out.append(row)
            by_scene_count[scene] = c + 1
        return out

    centers = [camera_center(f.T_wc) for f in frames]
    forwards = [camera_forward_world(f.T_wc) for f in frames]
    cam_ups = [camera_up_world(f.T_wc) for f in frames]

    # Per-scene reference "up" direction (robust to arbitrary world axis labels).
    scene_up: Dict[str, np.ndarray] = {}
    for fr, up in zip(frames, cam_ups):
        if fr.scene_hash not in scene_up:
            scene_up[fr.scene_hash] = up.copy()
        else:
            scene_up[fr.scene_hash] += up
    for k, v in scene_up.items():
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            scene_up[k] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            scene_up[k] = v / n

    # Pre-cache existence checks to avoid repeated disk hits
    exists = [(dl3dv_root / f.rel_path).resolve().is_file() for f in frames]

    pool: List[Tuple[int, int, float, float, float, float, float]] = []
    n = len(frames)
    for i in tqdm(range(n), desc="Building candidate pool", unit="frame"):
        if not exists[i]:
            continue
        for j in range(i + 1, n):
            if not exists[j]:
                continue
            if frames[i].scene_hash != frames[j].scene_hash:
                continue
            if scene_excluded(frames[i].scene_hash):
                continue

            up_ref = scene_up[frames[i].scene_hash]
            yaw_diff = yaw_diff_deg_wrt_up(forwards[i], forwards[j], up_ref)
            pitch_i = pitch_deg_wrt_up(forwards[i], up_ref)
            pitch_j = pitch_deg_wrt_up(forwards[j], up_ref)
            pitch_diff = abs(pitch_i - pitch_j)
            if yaw_diff < yaw_min or yaw_diff > yaw_max:
                continue
            if pitch_diff > pitch_max:
                continue

            rot = geodesic_angle_deg(frames[i].T_wc, frames[j].T_wc)
            baseline = float(np.linalg.norm(centers[i] - centers[j]))
            fwd_ang = forward_angle_deg(frames[i].T_wc, frames[j].T_wc)
            if fwd_ang < min_forward_angle:
                continue
            if baseline < min_baseline:
                continue

            if max_look_to_mid_angle is not None:
                ci = centers[i]
                cj = centers[j]
                mid = 0.5 * (ci + cj)
                vi = mid - ci
                vj = mid - cj
                ni = float(np.linalg.norm(vi))
                nj = float(np.linalg.norm(vj))
                if ni < 1e-9 or nj < 1e-9:
                    continue
                vi = vi / ni
                vj = vj / nj
                fi = camera_forward_world(frames[i].T_wc)
                fj = camera_forward_world(frames[j].T_wc)
                look_i = angle_between_deg(fi, vi)
                look_j = angle_between_deg(fj, vj)
                if look_i > max_look_to_mid_angle or look_j > max_look_to_mid_angle:
                    continue
            pool.append((i, j, yaw_diff, pitch_diff, rot, fwd_ang, baseline))

    if not pool:
        return []

    rng = np.random.default_rng(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)

    # No overlap check: just random selection.
    if max_overlap_ratio is None:
        shuffled = cap_per_scene(shuffled)
        if max_pairs > 0:
            return shuffled[:max_pairs]
        return shuffled

    # Overlap-aware selection: evaluate only a shuffled prefilter subset.
    if max_pairs > 0:
        pre_n = min(len(shuffled), max_pairs * max(1, overlap_prefilter_factor))
        candidates = shuffled[:pre_n]
    else:
        candidates = shuffled

    selected: List[Tuple[int, int, float, float, float, float, float]] = []
    pbar = tqdm(candidates, desc="Overlap screening", unit="pair")
    for row in pbar:
        i, j, _, _, _, _, _ = row
        pa = (dl3dv_root / frames[i].rel_path).resolve()
        pb = (dl3dv_root / frames[j].rel_path).resolve()
        ov = quick_overlap_proxy(pa, pb, max_side=overlap_check_max_side)
        # Keep only confidently low-overlap pairs.
        if np.isfinite(ov) and ov <= max_overlap_ratio:
            selected.append(row)
            pbar.set_postfix(found=len(selected))
            if max_pairs > 0 and len(selected) >= max_pairs:
                break

    selected = cap_per_scene(selected)
    if max_pairs > 0:
        selected = selected[:max_pairs]
    return selected


# ---------------------------------------------------------------------------
# DUSt3R inference
# ---------------------------------------------------------------------------

def run_dust3r_pair_pose(
    model,
    device: torch.device,
    path_a: Path,
    path_b: Path,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    square_ok = bool(getattr(model, "square_ok", False))
    patch_size = int(getattr(model, "patch_size", 16))
    imgs = load_images(
        [str(path_a), str(path_b)],
        size=image_size,
        verbose=False,
        patch_size=patch_size,
        square_ok=square_ok,
    )
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=False)
    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=False
    )
    c2w = scene.get_im_poses().detach().float().cpu().numpy()
    T0 = np.linalg.inv(c2w[0]).astype(np.float64)
    T1 = np.linalg.inv(c2w[1]).astype(np.float64)
    return T0, T1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DL3DV-10K-Sample: select same-scene pairs by yaw range + pitch constraint "
            "and evaluate DUSt3R relative-pose accuracy."
        )
    )
    parser.add_argument(
        "--dl3dv_root",
        type=Path,
        default=Path("DL3DV-10K-Sample"),
        help="Root of DL3DV-10K-Sample (contains <hash>/colmap/transforms.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to DUSt3R .pth checkpoint.",
    )
    parser.add_argument(
        "--dust3r_root",
        type=Path,
        default=Path("dust3r"),
        help="Path to local DUSt3R repo (added to sys.path).",
    )
    parser.add_argument("--yaw_min", type=float, default=50.0, help="Min yaw difference (degrees).")
    parser.add_argument("--yaw_max", type=float, default=90.0, help="Max yaw difference (degrees).")
    parser.add_argument(
        "--pitch_max",
        type=float,
        default=90.0,
        help="Max up/down (pitch) difference in degrees.",
    )
    parser.add_argument(
        "--min_forward_angle",
        type=float,
        default=40.0,
        help="Minimum optical-axis angle in degrees (larger => lower overlap).",
    )
    parser.add_argument(
        "--min_baseline",
        type=float,
        default=2.0,
        help="Minimum camera-center baseline in meters.",
    )
    parser.add_argument(
        "--max_look_to_mid_angle",
        type=float,
        default=-1.0,
        help=(
            "Max angle (deg) between each camera forward direction and the "
            "direction to the pair midpoint. Set <0 to disable."
        ),
    )
    parser.add_argument(
        "--max_overlap_ratio",
        type=float,
        default=0.35,
        help=(
            "Maximum overlap proxy (RANSAC inlier ratio in [0,1]). "
            "Set <0 to disable overlap-proxy filtering."
        ),
    )
    parser.add_argument(
        "--overlap_check_max_side",
        type=int,
        default=640,
        help="Max image side for overlap proxy feature matching.",
    )
    parser.add_argument(
        "--overlap_prefilter_factor",
        type=int,
        default=8,
        help="Evaluate overlap on up to max_pairs*factor shuffled candidates.",
    )
    parser.add_argument(
        "--max_pairs_per_scene",
        type=int,
        default=0,
        help="Cap selected pair count per scene (0 = disabled).",
    )
    parser.add_argument(
        "--exclude_scene",
        action="append",
        default=[],
        help="Exclude scene hash prefixes; can be passed multiple times.",
    )
    # Backward-compatible aliases from older geodesic-rotation based interface.
    parser.add_argument("--rot_min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rot_max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=300,
        help="Maximum number of pairs to randomly sample and evaluate (0 = use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for pair sampling (ensures reproducibility).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        choices=(224, 512),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save_pairs_csv",
        type=Path,
        default=None,
        help="Optional: save selected pair list as CSV.",
    )
    parser.add_argument(
        "--input_pairs_csv",
        type=Path,
        default=None,
        help=(
            "Optional: evaluate using an existing pairs CSV (must contain rel_a, rel_b). "
            "If set, pair-selection filters are skipped."
        ),
    )
    parser.add_argument(
        "--save_pairs_txt",
        type=Path,
        default=None,
        help="Optional: save selected pair list as TXT (rel_a rel_b per line).",
    )
    parser.add_argument(
        "--save_metrics_csv",
        type=Path,
        default=None,
        help="Optional: save per-pair metrics as CSV.",
    )
    parser.add_argument(
        "--debug_worst_pairs_k",
        type=int,
        default=0,
        help="Save top-K worst pairs by joint error max(RE,TE) (0 = disabled).",
    )
    parser.add_argument(
        "--debug_worst_pairs_csv",
        type=Path,
        default=None,
        help="Output CSV path for worst pairs (default: alongside save_metrics_csv).",
    )
    parser.add_argument(
        "--debug_pairs",
        type=int,
        default=0,
        help="Print detailed errors for the first N pairs.",
    )
    args = parser.parse_args()
    if args.rot_min is not None:
        args.yaw_min = float(args.rot_min)
    if args.rot_max is not None:
        args.yaw_max = float(args.rot_max)
    if args.max_overlap_ratio is not None and args.max_overlap_ratio < 0:
        args.max_overlap_ratio = None
    if args.max_look_to_mid_angle is not None and args.max_look_to_mid_angle < 0:
        args.max_look_to_mid_angle = None

    # ------------------------------------------------------------------
    # Setup DUSt3R path
    # ------------------------------------------------------------------
    droot = args.dust3r_root.resolve()
    if str(droot) not in sys.path:
        sys.path.insert(0, str(droot))
    from dust3r.model import load_model

    # ------------------------------------------------------------------
    # Step 1: load GT poses
    # ------------------------------------------------------------------
    dl3dv_root = args.dl3dv_root.resolve()
    print(f"Loading DL3DV frames from {dl3dv_root} ...")
    frames = load_dl3dv_frames(dl3dv_root)
    if not frames:
        raise RuntimeError("No frames loaded. Check --dl3dv_root.")
    print(f"  Loaded {len(frames)} frames from {len(set(f.scene_hash for f in frames))} scenes.")

    # ------------------------------------------------------------------
    # Step 2: select pairs
    # ------------------------------------------------------------------
    if args.input_pairs_csv is not None:
        pairs_csv = args.input_pairs_csv.resolve()
        print(f"\nLoading selected pairs from CSV: {pairs_csv}")
        idx_by_rel = {f.rel_path: i for i, f in enumerate(frames)}
        selected = []
        with pairs_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            if "rel_a" not in cols or "rel_b" not in cols:
                raise RuntimeError(
                    f"{pairs_csv} must contain columns rel_a and rel_b, got: {sorted(cols)}"
                )
            for row in reader:
                ra = str(row["rel_a"]).strip()
                rb = str(row["rel_b"]).strip()
                if ra not in idx_by_rel or rb not in idx_by_rel:
                    continue
                i = idx_by_rel[ra]
                j = idx_by_rel[rb]
                yd = float(row.get("yaw_diff_deg", float("nan")))
                pd = float(row.get("pitch_diff_deg", float("nan")))
                rot = float(row.get("rot_deg", float("nan")))
                fa = float(row.get("forward_angle_deg", float("nan")))
                bl = float(row.get("baseline_m", float("nan")))
                selected.append((i, j, yd, pd, rot, fa, bl))
        if args.max_pairs > 0:
            selected = selected[: args.max_pairs]
        print(f"  Loaded {len(selected)} pairs from input CSV.")
    else:
        print(
            f"\nSelecting same-scene pairs with yaw ∈ "
            f"[{args.yaw_min}, {args.yaw_max}]°, pitch_diff <= {args.pitch_max}°, "
            f"forward >= {args.min_forward_angle}°, baseline >= {args.min_baseline}m, "
            f"look_to_mid <= {args.max_look_to_mid_angle if args.max_look_to_mid_angle is not None else 'disabled'}°, "
            f"overlap <= {args.max_overlap_ratio if args.max_overlap_ratio is not None else 'disabled'} ..."
        )
        selected = select_pairs(
            frames=frames,
            dl3dv_root=dl3dv_root,
            yaw_min=args.yaw_min,
            yaw_max=args.yaw_max,
            pitch_max=args.pitch_max,
            min_forward_angle=args.min_forward_angle,
            min_baseline=args.min_baseline,
            max_look_to_mid_angle=args.max_look_to_mid_angle,
            max_overlap_ratio=args.max_overlap_ratio,
            overlap_check_max_side=args.overlap_check_max_side,
            overlap_prefilter_factor=args.overlap_prefilter_factor,
            max_pairs=args.max_pairs,
            max_pairs_per_scene=args.max_pairs_per_scene,
            exclude_scene_prefixes=list(args.exclude_scene),
            seed=args.seed,
        )
    if not selected:
        raise RuntimeError(
            f"No pairs found with yaw in [{args.yaw_min}, {args.yaw_max}] and "
            f"pitch_diff <= {args.pitch_max}. Try relaxing constraints."
        )
    print(f"  Selected {len(selected)} pairs.")
    if args.max_pairs_per_scene > 0:
        print(f"  Per-scene cap active: {args.max_pairs_per_scene} pairs/scene")
    if args.exclude_scene:
        print(f"  Excluded scene prefixes: {args.exclude_scene}")

    yaw_arr = np.array([yd for _, _, yd, _, _, _, _ in selected])
    pitch_arr = np.array([pd for _, _, _, pd, _, _, _ in selected])
    rot_arr = np.array([rot for _, _, _, _, rot, _, _ in selected])
    print(
        f"  yaw_diff stats: min={yaw_arr.min():.2f}, "
        f"median={np.median(yaw_arr):.2f}, max={yaw_arr.max():.2f}"
    )
    print(
        f"  pitch_diff stats: min={pitch_arr.min():.2f}, "
        f"median={np.median(pitch_arr):.2f}, max={pitch_arr.max():.2f}"
    )
    print(
        f"  rot_deg stats: min={rot_arr.min():.2f}, "
        f"median={np.median(rot_arr):.2f}, max={rot_arr.max():.2f}"
    )

    # Optionally save pair list
    if args.save_pairs_csv is not None:
        args.save_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_pairs_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "i", "j", "rel_a", "rel_b", "scene",
                    "yaw_diff_deg", "pitch_diff_deg", "rot_deg",
                    "forward_angle_deg", "baseline_m",
                ],
            )
            w.writeheader()
            for i, j, yd, pd, rot, fa, bl in selected:
                w.writerow(
                    dict(
                        i=i, j=j,
                        rel_a=frames[i].rel_path,
                        rel_b=frames[j].rel_path,
                        scene=frames[i].scene_hash[:12],
                        yaw_diff_deg=round(yd, 4),
                        pitch_diff_deg=round(pd, 4),
                        rot_deg=round(rot, 4),
                        forward_angle_deg=round(fa, 4),
                        baseline_m=round(bl, 4),
                    )
                )
        print(f"  Saved pairs CSV: {args.save_pairs_csv.resolve()}")

    if args.save_pairs_txt is not None:
        args.save_pairs_txt.parent.mkdir(parents=True, exist_ok=True)
        with args.save_pairs_txt.open("w") as f:
            for i, j, *_ in selected:
                f.write(f"{frames[i].rel_path} {frames[j].rel_path}\n")
        print(f"  Saved pairs TXT: {args.save_pairs_txt.resolve()}")

    # ------------------------------------------------------------------
    # Step 3: load model and evaluate
    # ------------------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nLoading DUSt3R from {args.checkpoint.resolve()} ...")
    model = load_model(str(args.checkpoint.resolve()), device=str(device), verbose=True)
    model.eval()

    rot_errs: List[float] = []
    trans_ang_errs: List[float] = []
    pair_rows: List[Dict] = []
    debug_left = int(max(0, args.debug_pairs))

    for idx, (i, j, yd, pd, rot, fa, bl) in enumerate(
        tqdm(selected, desc="Evaluating pairs", unit="pair")
    ):
        pa = (dl3dv_root / frames[i].rel_path).resolve()
        pb = (dl3dv_root / frames[j].rel_path).resolve()

        try:
            T_pred_a, T_pred_b = run_dust3r_pair_pose(
                model, device, pa, pb, image_size=args.image_size
            )
        except Exception as e:
            print(f"[skip pair {idx}] ({frames[i].rel_path}, {frames[j].rel_path}): {e}")
            continue

        T_gt_a = frames[i].T_wc
        T_gt_b = frames[j].T_wc

        rel_pr = relative_pose_T(T_pred_a, T_pred_b)
        rel_gt = relative_pose_T(T_gt_a, T_gt_b)
        R_pr, t_pr = T_to_Rt(rel_pr)
        R_gt, t_gt = T_to_Rt(rel_gt)

        re = geodesic_rotation_error_deg(R_pr, R_gt)
        te_ang = translation_direction_angle_error_deg(t_pr, t_gt)

        rot_errs.append(re)
        trans_ang_errs.append(te_ang)
        pair_rows.append(
            {
                "pair_idx": idx,
                "rel_a": frames[i].rel_path,
                "rel_b": frames[j].rel_path,
                "scene": frames[i].scene_hash[:12],
                "yaw_diff_deg": round(yd, 4),
                "pitch_diff_deg": round(pd, 4),
                "rot_deg": round(rot, 4),
                "rot_err_deg": re,
                "trans_dir_err_deg": te_ang,
            }
        )

        if debug_left > 0:
            print(
                f"[debug pair {idx}] {frames[i].rel_path} | {frames[j].rel_path} | "
                f"yaw={yd:.1f}° pitch={pd:.1f}° rot={rot:.1f}° "
                f"rot_err={re:.3f}° trans_dir_err={te_ang:.3f}°"
            )
            debug_left -= 1

    if not rot_errs:
        raise RuntimeError("No valid pairs evaluated.")

    rot_e = np.array(rot_errs, dtype=np.float64)
    trans_ang = np.array(trans_ang_errs, dtype=np.float64)

    metrics = {
        "Evaluated_pairs": float(len(rot_e)),
        "MRE_deg":         float(np.nanmean(rot_e)),
        "median_RE_deg":   float(np.nanmedian(rot_e)),
        "R_acc_5":         acc_at_deg(rot_e, 5.0),
        "R_acc_15":        acc_at_deg(rot_e, 15.0),
        "R_acc_30":        acc_at_deg(rot_e, 30.0),
        "MTE_deg":         float(np.nanmean(trans_ang)),
        "median_TE_deg":   float(np.nanmedian(trans_ang)),
        "t_acc_5":         acc_at_deg(trans_ang, 5.0),
        "t_acc_15":        acc_at_deg(trans_ang, 15.0),
        "t_acc_30":        acc_at_deg(trans_ang, 30.0),
        # Paper-style final AUC30: min(AUC30_rot, AUC30_trans)
        "AUC30":           auc30_paper(rot_e, trans_ang),
        # Separate AUC30 values for debugging
        "AUC30_rot":       auc30(rot_e),
        "AUC30_trans":     auc30(trans_ang),
    }

    print(
        f"\n=== DUSt3R on DL3DV-10K-Sample "
        f"(yaw=[{args.yaw_min},{args.yaw_max}]°, pitch<= {args.pitch_max}°) ==="
    )
    for k, v in metrics.items():
        if k == "Evaluated_pairs":
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.6g}")

    # Per-scene diagnostics: helps detect one-scene domination.
    by_scene: Dict[str, Dict[str, List[float]]] = {}
    for row in pair_rows:
        s = row["scene"]
        if s not in by_scene:
            by_scene[s] = {"re": [], "te": []}
        by_scene[s]["re"].append(float(row["rot_err_deg"]))
        by_scene[s]["te"].append(float(row["trans_dir_err_deg"]))
    if by_scene:
        print("\nPer-scene summary (sorted by hard-rate max(RE,TE)>30°):")
        scene_rows = []
        for s, d in by_scene.items():
            re = np.array(d["re"], dtype=np.float64)
            te = np.array(d["te"], dtype=np.float64)
            joint = np.maximum(re, te)
            hard = float((joint > 30.0).mean())
            scene_rows.append(
                (
                    s,
                    len(re),
                    float(np.nanmean(re)),
                    float(np.nanmean(te)),
                    hard,
                )
            )
        scene_rows.sort(key=lambda x: x[4], reverse=True)
        for s, n_pairs, mre_s, mte_s, hard_s in scene_rows:
            print(
                f"  {s}: n={n_pairs}, MRE={mre_s:.3f}, "
                f"MTE={mte_s:.3f}, hard@30={hard_s:.3f}"
            )

    if args.save_metrics_csv is not None:
        args.save_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_metrics_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "pair_idx", "rel_a", "rel_b", "scene",
                    "yaw_diff_deg", "pitch_diff_deg", "rot_deg",
                    "rot_err_deg", "trans_dir_err_deg",
                ],
            )
            w.writeheader()
            w.writerows(pair_rows)
        print(f"\nSaved per-pair metrics: {args.save_metrics_csv.resolve()}")

    if args.debug_worst_pairs_k > 0 and pair_rows:
        ranked = []
        for r in pair_rows:
            re = float(r["rot_err_deg"])
            te = float(r["trans_dir_err_deg"])
            joint = float(max(re, te)) if np.isfinite(re) and np.isfinite(te) else float("nan")
            rr = dict(r)
            rr["joint_err_deg"] = joint
            ranked.append(rr)
        ranked = [r for r in ranked if np.isfinite(r["joint_err_deg"])]
        ranked.sort(key=lambda x: x["joint_err_deg"], reverse=True)
        k = min(int(args.debug_worst_pairs_k), len(ranked))
        worst = ranked[:k]

        if args.debug_worst_pairs_csv is not None:
            out_csv = args.debug_worst_pairs_csv
        elif args.save_metrics_csv is not None:
            out_csv = args.save_metrics_csv.with_name(
                args.save_metrics_csv.stem + "_worst_pairs.csv"
            )
        else:
            out_csv = Path("outputs/dl3dv_worst_pairs.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "pair_idx", "rel_a", "rel_b", "scene",
                    "yaw_diff_deg", "pitch_diff_deg", "rot_deg",
                    "rot_err_deg", "trans_dir_err_deg", "joint_err_deg",
                ],
            )
            w.writeheader()
            w.writerows(worst)
        print(f"Saved worst-{k} pairs CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
