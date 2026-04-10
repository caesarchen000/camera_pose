#!/usr/bin/env python3
"""
DL3DV-10K-Sample: visualize DUSt3R predicted poses vs GT for selected pairs.

Reads the pairs CSV produced by eval_dust3r_dl3dv.py, runs DUSt3R on each pair,
and saves one PNG per pair showing:
  - Left panel:  the two input frames side by side (cyan / yellow borders)
  - Right panel: 3D camera frustums — GT (dashed) vs DUSt3R predicted (solid),
                 with the predicted point cloud, SE(3)-aligned to GT world.

Usage:
  python scripts/DL3DV/visualize_dust3r_dl3dv.py \\
      --dl3dv_root   DL3DV-10K-Sample \\
      --checkpoint   checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
      --pairs_csv    outputs/dl3dv_pairs_50_90.csv \\
      --viz_dir      outputs/viz_dl3dv \\
      --num_pairs    10 \\
      --device       cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def camera_center_and_forward(T_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    C = -R.T @ t
    fwd = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fwd /= np.linalg.norm(fwd) + 1e-12
    return C, fwd


def relative_pose_T(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    return T_b @ np.linalg.inv(T_a)


def geodesic_rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_err = R_pred @ R_gt.T
    c = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def translation_direction_angle_error_deg(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    np_pred = np.linalg.norm(t_pred)
    np_gt = np.linalg.norm(t_gt)
    if np_pred < 1e-9 or np_gt < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(float(np.clip(
        np.dot(t_pred / np_pred, t_gt / np_gt), -1.0, 1.0
    )))))


# ---------------------------------------------------------------------------
# Load GT poses from transforms.json (same as eval script)
# ---------------------------------------------------------------------------

def load_gt_pose(dl3dv_root: Path, rel_path: str) -> np.ndarray:
    """
    Given rel_path like '<hash>/colmap/images/frame_XXXXX.png',
    load its world->camera pose from the scene's transforms.json.
    """
    parts = rel_path.split("/")
    if len(parts) < 4:
        raise ValueError(f"Invalid rel_path format: {rel_path}")
    scene_hash = parts[0]
    parent_name = parts[1]  # "colmap" or "nerfstudio"
    file_path = "/".join(parts[2:])
    # transforms often stores "images/..." while benchmark may use "images_4/..."
    file_path_alt = file_path
    if file_path_alt.startswith("images_4/"):
        file_path_alt = "images/" + file_path_alt[len("images_4/"):]

    tf_path = dl3dv_root / scene_hash / parent_name / "transforms.json"
    d = json.loads(tf_path.read_text())
    for fr in d.get("frames", []):
        fp = fr.get("file_path", "")
        if fp == file_path or fp == file_path_alt:
            c2w = np.array(fr["transform_matrix"], dtype=np.float64)
            # DL3DV uses NeRF/OpenGL c2w (Y-up, Z-backward).
            # Convert to OpenCV (Y-down, Z-forward) before inverting.
            FLIP = np.diag([1., -1., -1., 1.])
            return np.linalg.inv(c2w @ FLIP)
    raise KeyError(f"Frame '{file_path}' not found in {tf_path}")


# ---------------------------------------------------------------------------
# DUSt3R inference (returns poses + point cloud + per-pixel colors)
# ---------------------------------------------------------------------------

def run_dust3r_pair(
    model,
    device: torch.device,
    path_a: Path,
    path_b: Path,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      T_pred_a, T_pred_b : world->cam 4x4
      world_pts          : (2, H, W, 3) 3D points in DUSt3R world space
      rgb_imgs           : (2, H, W, 3) uint8 RGB — pixel colors matching world_pts
    """
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images
    from PIL import Image as PILImage

    square_ok = bool(getattr(model, "square_ok", False))
    patch_size = int(getattr(model, "patch_size", 16))
    imgs = load_images(
        [str(path_a), str(path_b)],
        size=image_size, verbose=False,
        patch_size=patch_size, square_ok=square_ok,
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
    pts3d = scene.get_pts3d()
    world_pts = np.stack([to_numpy(p).astype(np.float64) for p in pts3d], axis=0)  # (2,H,W,3)

    # Grab pixel colors at the resolution DUSt3R used
    H, W = world_pts.shape[1], world_pts.shape[2]
    rgb_imgs = []
    for p in [path_a, path_b]:
        im = np.array(PILImage.open(p).convert("RGB").resize((W, H), PILImage.BILINEAR), dtype=np.uint8)
        rgb_imgs.append(im)
    rgb_imgs = np.stack(rgb_imgs, axis=0)  # (2,H,W,3)

    return T0, T1, world_pts, rgb_imgs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualization_png(
    out_path: Path,
    path_a: Path,
    path_b: Path,
    rel_a: str,
    rel_b: str,
    T_pred_a: np.ndarray,
    T_pred_b: np.ndarray,
    T_gt_a: np.ndarray,
    T_gt_b: np.ndarray,
    world_pts: np.ndarray,
    rgb_imgs: np.ndarray,
    rot_err_deg: float,
    trans_ang_err_deg: float,
    rot_deg: float,
) -> None:
    """Horizontal figure: label | input pair | camera poses | point cloud."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from PIL import Image

    COLOR_A_BORDER = (46, 207, 255)
    COLOR_B_BORDER = (230, 194, 0)
    COLOR_A_HEX = "#2ecfff"
    COLOR_B_HEX = "#e6c200"
    GT_A_HEX = "#156a85"
    GT_B_HEX = "#8a7218"
    SERIF = "DejaVu Serif"

    # ---- alignment: anchor pred cam A to GT cam A (SE3, s=1) ----
    T_align = (np.linalg.inv(T_gt_a) @ T_pred_a).astype(np.float64)
    R_align = T_align[:3, :3].copy()
    t_align = T_align[:3, 3].copy()
    s_align = 1.0

    Ga, _ = camera_center_and_forward(T_gt_a)
    Gb, _ = camera_center_and_forward(T_gt_b)
    baseline_gt = float(np.linalg.norm(Gb - Ga))
    if baseline_gt < 1e-6:
        baseline_gt = 1.0

    # ---- image loading helpers ----
    def load_img(p: Path, max_w: int = 520) -> np.ndarray:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w > max_w:
            h = int(h * max_w / w)
            im = im.resize((max_w, h), Image.Resampling.BICUBIC)
        return np.asarray(im)

    def add_border(arr: np.ndarray, rgb: Tuple, t: int) -> np.ndarray:
        h, w = arr.shape[:2]
        out = np.zeros((h + 2 * t, w + 2 * t, 3), dtype=np.uint8)
        out[:, :] = np.array(rgb, dtype=np.uint8)
        out[t:t + h, t:t + w] = arr
        return out

    def match_height(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ha, hb = a.shape[0], b.shape[0]
        if ha == hb:
            return a, b
        if ha < hb:
            nw = int(a.shape[1] * hb / ha)
            a = np.asarray(Image.fromarray(a).resize((nw, hb), Image.Resampling.BICUBIC))
        else:
            nw = int(b.shape[1] * ha / hb)
            b = np.asarray(Image.fromarray(b).resize((nw, ha), Image.Resampling.BICUBIC))
        return a, b

    im0 = load_img(path_a)
    im1 = load_img(path_b)
    im0, im1 = match_height(im0, im1)
    bp = max(6, min(14, im0.shape[0] // 48))
    aspect_wh = 0.5 * (
        float(im0.shape[1]) / float(max(im0.shape[0], 1)) +
        float(im1.shape[1]) / float(max(im1.shape[0], 1))
    )
    pair_strip = np.concatenate(
        [add_border(im0, COLOR_A_BORDER, bp), add_border(im1, COLOR_B_BORDER, bp)], axis=1
    )

    # ---- figure layout ----
    fig = plt.figure(figsize=(20, 7.2), dpi=120)
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 3, figure=fig,
                  width_ratios=[0.055, 1.05, 1.12],
                  wspace=0.06, left=0.02, right=0.98, top=0.92, bottom=0.08)

    ax_side = fig.add_subplot(gs[0, 0])
    ax_side.set_facecolor("white")
    ax_side.axis("off")
    ax_side.text(0.52, 0.50, "DL3DV",
                 transform=ax_side.transAxes, rotation=90,
                 fontsize=15, fontfamily=SERIF, va="center", ha="center", color="#222222")

    ax_in = fig.add_subplot(gs[0, 1])
    ax_in.imshow(pair_strip)
    ax_in.axis("off")
    ax_in.set_title("Input Image Pair", fontsize=14, fontfamily=SERIF, pad=12, color="#111111")

    ax_cam = fig.add_subplot(gs[0, 2], projection="3d")
    ax_pc = fig.add_subplot(gs[0, 2], projection="3d")
    ax_cam.remove()
    ax_pc.remove()
    gs_right = gs[0, 2].subgridspec(1, 2, wspace=0.12, width_ratios=[1.0, 1.0])
    ax_cam = fig.add_subplot(gs_right[0, 0], projection="3d")
    ax_pc = fig.add_subplot(gs_right[0, 1], projection="3d")

    def apply_align(T_wc: np.ndarray) -> np.ndarray:
        T_new = np.eye(4, dtype=np.float64)
        T_new[:3, :3] = T_wc[:3, :3] @ R_align.T
        T_new[:3, 3] = T_wc[:3, 3] - T_new[:3, :3] @ t_align
        return T_new

    def frustum_points(T_wc: np.ndarray, depth: float) -> np.ndarray:
        w = depth * 0.7
        h = depth * 0.45
        local = np.array([
            [0.0, 0.0, 0.0],     # center
            [-w, -h, depth],
            [w, -h, depth],
            [w, h, depth],
            [-w, h, depth],
        ], dtype=np.float64)
        R, t = T_wc[:3, :3], T_wc[:3, 3]
        C = -R.T @ t
        return (R.T @ local.T).T + C[None, :]

    def draw_frustum(ax, T_wc: np.ndarray, color: str, depth: float, dashed: bool) -> None:
        pts = frustum_points(T_wc, depth)
        c = pts[0]
        corners = pts[1:]
        ls = "--" if dashed else "-"
        lw = 2.2 if dashed else 1.6
        a = 0.95 if dashed else 0.9
        for k in range(4):
            p0 = corners[k]
            p1 = corners[(k + 1) % 4]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, ls=ls, lw=lw, alpha=a)
            ax.plot([c[0], p0[0]], [c[1], p0[1]], [c[2], p0[2]], color=color, ls=ls, lw=lw, alpha=a)

    T_pred_a_al = apply_align(T_pred_a)
    T_pred_b_al = apply_align(T_pred_b)
    world_pts_al = world_pts.copy()
    world_pts_al = (R_align @ world_pts_al.reshape(-1, 3).T).T + t_align[None, :]
    world_pts_al = world_pts_al.reshape(world_pts.shape)

    Ca, _ = camera_center_and_forward(T_gt_a)
    Cb, _ = camera_center_and_forward(T_gt_b)
    baseline_gt = float(np.linalg.norm(Ca - Cb))
    frustum_depth = max(0.15, min(1.4, baseline_gt * 0.30))

    draw_frustum(ax_cam, T_gt_a, GT_A_HEX, frustum_depth, dashed=True)
    draw_frustum(ax_cam, T_gt_b, GT_B_HEX, frustum_depth, dashed=True)
    draw_frustum(ax_cam, T_pred_a_al, COLOR_A_HEX, frustum_depth, dashed=False)
    draw_frustum(ax_cam, T_pred_b_al, COLOR_B_HEX, frustum_depth, dashed=False)

    cam_pts = np.vstack([
        camera_center_and_forward(T_gt_a)[0],
        camera_center_and_forward(T_gt_b)[0],
        camera_center_and_forward(T_pred_a_al)[0],
        camera_center_and_forward(T_pred_b_al)[0],
    ])
    cam_min = np.min(cam_pts, axis=0)
    cam_max = np.max(cam_pts, axis=0)
    cam_ctr = 0.5 * (cam_min + cam_max)
    cam_rad = max(0.8, 0.6 * float(np.max(cam_max - cam_min) + 1e-6))
    ax_cam.set_xlim(cam_ctr[0] - cam_rad, cam_ctr[0] + cam_rad)
    ax_cam.set_ylim(cam_ctr[1] - cam_rad, cam_ctr[1] + cam_rad)
    ax_cam.set_zlim(cam_ctr[2] - cam_rad, cam_ctr[2] + cam_rad)
    ax_cam.set_box_aspect([1, 1, 1])
    ax_cam.set_title("Predicted + GT Camera Poses", fontsize=11, fontfamily=SERIF, pad=8, color="#111111")
    ax_cam.view_init(elev=20, azim=-60)
    ax_cam.grid(True, alpha=0.2)

    # ---- build flat (N,3) point cloud + RGB colors from both views ----
    S, pH, pW, _ = world_pts_al.shape
    stride = max(2, max(pH, pW) // 80)
    pts_raw = world_pts_al[:, ::stride, ::stride, :].reshape(-1, 3)  # (N,3)
    cols_raw = rgb_imgs[:, ::stride, ::stride, :].reshape(-1, 3)    # (N,3) uint8

    cols_vis = cols_raw.astype(np.float32) / 255.0
    if pts_raw.shape[0] > 12000:
        sel = np.random.default_rng(0).choice(pts_raw.shape[0], size=12000, replace=False)
        pts_raw = pts_raw[sel]
        cols_vis = cols_vis[sel]
    ax_pc.scatter(pts_raw[:, 0], pts_raw[:, 1], pts_raw[:, 2], c=cols_vis, s=1.0, alpha=0.9, linewidths=0)
    pmin = np.percentile(pts_raw, 2, axis=0)
    pmax = np.percentile(pts_raw, 98, axis=0)
    pctr = 0.5 * (pmin + pmax)
    prad = max(0.8, 0.55 * float(np.max(pmax - pmin) + 1e-6))
    ax_pc.set_xlim(pctr[0] - prad, pctr[0] + prad)
    ax_pc.set_ylim(pctr[1] - prad, pctr[1] + prad)
    ax_pc.set_zlim(pctr[2] - prad, pctr[2] + prad)
    ax_pc.set_box_aspect([1, 1, 1])
    ax_pc.view_init(elev=20, azim=-60)
    ax_pc.grid(False)
    ax_pc.set_axis_off()

    frame_a = Path(rel_a).name
    frame_b = Path(rel_b).name
    fig.text(0.50, 0.015,
             f"GT rot={rot_deg:.1f}°  |  rot_err={rot_err_deg:.2f}°  "
             f"trans_dir_err={trans_ang_err_deg:.2f}°  |  "
             f"{frame_a} — {frame_b}",
             ha="center", fontsize=8, fontfamily=SERIF, color="#333333")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize DUSt3R predictions on DL3DV pairs (PNG per pair)."
    )
    parser.add_argument(
        "--dl3dv_root",
        type=Path,
        default=Path("DL3DV-10K-Sample"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="DUSt3R .pth checkpoint.",
    )
    parser.add_argument(
        "--dust3r_root",
        type=Path,
        default=Path("dust3r"),
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        required=True,
        help="CSV produced by eval_dust3r_dl3dv.py (columns: rel_a, rel_b, rot_deg, ...).",
    )
    parser.add_argument(
        "--viz_dir",
        type=Path,
        default=Path("outputs/viz_dl3dv"),
        help="Output directory for PNG files.",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=10,
        help="Number of pairs to visualize.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        choices=(224, 512),
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # DUSt3R setup
    droot = args.dust3r_root.resolve()
    if str(droot) not in sys.path:
        sys.path.insert(0, str(droot))
    from dust3r.model import load_model

    # Load pair list
    dl3dv_root = args.dl3dv_root.resolve()
    pairs: List[Dict] = []
    with args.pairs_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            pairs.append(row)
    pairs = pairs[: args.num_pairs]
    if not pairs:
        raise RuntimeError(f"No pairs found in {args.pairs_csv}")
    print(f"Visualizing {len(pairs)} pairs → {args.viz_dir.resolve()}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading DUSt3R from {args.checkpoint.resolve()} ...")
    model = load_model(str(args.checkpoint.resolve()), device=str(device), verbose=True)
    model.eval()

    args.viz_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(tqdm(pairs, desc="Visualizing", unit="pair")):
        rel_a = row["rel_a"]
        rel_b = row["rel_b"]
        rot_deg = float(row.get("rot_deg", 0.0))

        pa = (dl3dv_root / rel_a).resolve()
        pb = (dl3dv_root / rel_b).resolve()
        if not pa.is_file() or not pb.is_file():
            print(f"[skip {idx}] missing image files")
            continue

        try:
            T_gt_a = load_gt_pose(dl3dv_root, rel_a)
            T_gt_b = load_gt_pose(dl3dv_root, rel_b)
        except Exception as e:
            print(f"[skip {idx}] GT pose error: {e}")
            continue

        try:
            T_pred_a, T_pred_b, world_pts, rgb_imgs = run_dust3r_pair(
                model, device, pa, pb, image_size=args.image_size
            )
        except Exception as e:
            print(f"[skip {idx}] DUSt3R error: {e}")
            continue

        rel_pr = relative_pose_T(T_pred_a, T_pred_b)
        rel_gt = relative_pose_T(T_gt_a, T_gt_b)
        R_pr, t_pr = rel_pr[:3, :3], rel_pr[:3, 3]
        R_gt_r, t_gt_r = rel_gt[:3, :3], rel_gt[:3, 3]
        rot_err = geodesic_rotation_error_deg(R_pr, R_gt_r)
        trans_err = translation_direction_angle_error_deg(t_pr, t_gt_r)

        frame_a = Path(rel_a).stem
        frame_b = Path(rel_b).stem
        scene = rel_a.split("/")[0][:12]
        out_png = args.viz_dir / f"pair_{idx:03d}_{scene}_{frame_a}_{frame_b}.png"

        try:
            save_visualization_png(
                out_path=out_png,
                path_a=pa, path_b=pb,
                rel_a=rel_a, rel_b=rel_b,
                T_pred_a=T_pred_a, T_pred_b=T_pred_b,
                T_gt_a=T_gt_a, T_gt_b=T_gt_b,
                world_pts=world_pts,
                rgb_imgs=rgb_imgs,
                rot_err_deg=rot_err,
                trans_ang_err_deg=trans_err,
                rot_deg=rot_deg,
            )
        except Exception as e:
            print(f"[viz failed {idx}] {e}")
            continue

    print(f"\nDone. PNGs saved under: {args.viz_dir.resolve()}")


if __name__ == "__main__":
    main()
