#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render one pair from metrics CSV as: input pair | reconstructed 3D scene. "
            "Supports method=dust3r or method=interpose."
        )
    )
    p.add_argument("--metrics_csv", type=Path, required=True)
    p.add_argument("--pair_index", type=int, required=True)
    p.add_argument("--method", type=str, choices=("dust3r", "interpose"), required=True)
    p.add_argument("--output_image", type=Path, required=True)

    p.add_argument(
        "--dl3dv_root",
        type=Path,
        default=Path("/home/caesar/camera_pose/DL3DV-Benchmark/DL3DV-10K-Benchmark"),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/caesar/camera_pose/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
    )
    p.add_argument(
        "--dust3r_root",
        type=Path,
        default=Path("/home/caesar/camera_pose/dust3r"),
    )
    p.add_argument("--image_size", type=int, default=512, choices=(224, 512))

    p.add_argument(
        "--interpose_root",
        type=Path,
        default=Path("/home/caesar/camera_pose/Interpose/out_interpose_batch"),
        help="InterPose batch output root containing pair_* folders.",
    )
    p.add_argument(
        "--interpose_selected_jsonl",
        type=Path,
        default=None,
        help="Path to selected_per_pair.jsonl (default: <interpose_root>/selection/selected_per_pair.jsonl).",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def parse_metrics_row(metrics_csv: Path, pair_index: int) -> Dict[str, str]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {metrics_csv}")

    # Prefer pair_idx lookup if present.
    for r in rows:
        if "pair_idx" in r:
            try:
                if int(r["pair_idx"]) == pair_index:
                    return r
            except Exception:
                pass
    # Fallback: positional index.
    if pair_index < 0 or pair_index >= len(rows):
        raise IndexError(f"pair_index={pair_index} out of range [0, {len(rows)-1}] for {metrics_csv}")
    return rows[pair_index]


def load_gt_T_wc_from_rel(dl3dv_root: Path, rel_path: str) -> np.ndarray:
    parts = rel_path.strip().split("/")
    if len(parts) < 4:
        raise ValueError(f"Invalid rel path: {rel_path}")
    scene_hash, parent_name = parts[0], parts[1]
    file_path = "/".join(parts[2:])
    file_path_alt = file_path
    if file_path.startswith("images_4/"):
        file_path_alt = "images/" + file_path[len("images_4/") :]

    tf_path = dl3dv_root / scene_hash / parent_name / "transforms.json"
    d = json.loads(tf_path.read_text())
    for fr in d.get("frames", []):
        fp = fr.get("file_path", "")
        if fp != file_path and fp != file_path_alt:
            continue
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        return np.linalg.inv(c2w @ flip)
    raise KeyError(f"Frame not found in transforms: {rel_path}")


def relative_pose_T(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    return T_b @ np.linalg.inv(T_a)


def geodesic_rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_err = R_pred @ R_gt.T
    c = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def translation_direction_angle_error_deg(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    n0 = float(np.linalg.norm(t_pred))
    n1 = float(np.linalg.norm(t_gt))
    if n0 < 1e-9 or n1 < 1e-9:
        return float("nan")
    d = float(np.clip(np.dot(t_pred / n0, t_gt / n1), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def run_dust3r_pair(
    model,
    device: torch.device,
    path_a: Path,
    path_b: Path,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        size=image_size,
        verbose=False,
        patch_size=patch_size,
        square_ok=square_ok,
    )
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=False)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=False)

    c2w = scene.get_im_poses().detach().float().cpu().numpy()
    T_a = np.linalg.inv(c2w[0]).astype(np.float64)
    T_b = np.linalg.inv(c2w[1]).astype(np.float64)

    pts3d = scene.get_pts3d()
    world_pts = np.stack([to_numpy(p).astype(np.float64) for p in pts3d], axis=0)  # (2,H,W,3)

    H, W = world_pts.shape[1], world_pts.shape[2]
    rgb_imgs = []
    for p in [path_a, path_b]:
        im = np.array(PILImage.open(p).convert("RGB").resize((W, H), PILImage.BILINEAR), dtype=np.uint8)
        rgb_imgs.append(im)
    rgb_imgs = np.stack(rgb_imgs, axis=0)
    return T_a, T_b, world_pts, rgb_imgs


def load_interpose_selected(interpose_root: Path, pair_index: int, selected_jsonl: Path) -> Tuple[Path, np.ndarray]:
    pair_glob = list(interpose_root.glob(f"pair_{pair_index:05d}_*"))
    if not pair_glob:
        raise FileNotFoundError(f"No InterPose pair folder for pair_index={pair_index} under {interpose_root}")
    pair_dir = pair_glob[0].resolve()

    rec = None
    with selected_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if Path(row.get("pair_dir", "")).resolve() == pair_dir:
                rec = row
                break
    if rec is None:
        raise RuntimeError(f"pair_dir not found in {selected_jsonl}: {pair_dir}")

    T_rel = np.array(rec["T_rel_selected"], dtype=np.float64)
    selected_variant = str(rec.get("selected_variant", "v0_A2B_p1_s1"))
    variant_dir = pair_dir / selected_variant
    pa = variant_dir / "pre_model_inputs" / "00_start_pre_model.png"
    pb = variant_dir / "pre_model_inputs" / "01_end_pre_model.png"
    if not pa.is_file() or not pb.is_file():
        raise FileNotFoundError(f"InterPose pre_model inputs missing under {variant_dir}")
    return pair_dir, T_rel


def make_figure(
    out_path: Path,
    path_a: Path,
    path_b: Path,
    world_pts: np.ndarray,
    rgb_imgs: np.ndarray,
    T_render_a: np.ndarray,
    T_render_b: np.ndarray,
    T_gt_rel_for_overlay: Optional[np.ndarray],
    method: str,
    pair_index: int,
    re_deg: float,
    te_deg: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from PIL import Image

    def camera_center(T_wc: np.ndarray) -> np.ndarray:
        R, t = T_wc[:3, :3], T_wc[:3, 3]
        return -R.T @ t

    def camera_up_world(T_wc: np.ndarray) -> np.ndarray:
        # OpenCV camera has +Y down, so up is negative of that axis in world.
        R = T_wc[:3, :3]
        v_down = R.T @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v_up = -v_down
        n = np.linalg.norm(v_up) + 1e-12
        return v_up / n

    def rot_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        v = np.cross(a, b)
        s = float(np.linalg.norm(v))
        c = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if s < 1e-10:
            if c > 0:
                return np.eye(3, dtype=np.float64)
            # 180 deg: pick any orthogonal axis
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = axis - np.dot(axis, a) * a
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            K = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
                dtype=np.float64,
            )
            return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - c) / (s * s))

    def orient_by_camera_pose(
        points: np.ndarray, T_a: np.ndarray, T_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        # 1) align average camera up to +Z
        up_avg = camera_up_world(T_a) + camera_up_world(T_b)
        if np.linalg.norm(up_avg) < 1e-9:
            up_avg = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R_up = rot_from_a_to_b(up_avg, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        P = (R_up @ points.T).T
        R_disp = R_up.copy()

        # 2) stabilize yaw using projected A->B baseline to +X
        Ca = (R_up @ camera_center(T_a).reshape(3, 1)).reshape(3)
        Cb = (R_up @ camera_center(T_b).reshape(3, 1)).reshape(3)
        b = Cb - Ca
        b[2] = 0.0
        nb = np.linalg.norm(b)
        if nb > 1e-9:
            b = b / nb
            yaw = float(np.arctan2(b[1], b[0]))
            cy, sy = np.cos(-yaw), np.sin(-yaw)
            Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            P = (Rz @ P.T).T
            R_disp = Rz @ R_disp

        # 3) move floor near z=0 for consistent framing
        z_floor = float(np.percentile(P[:, 2], 2))
        P[:, 2] -= z_floor
        return P, R_disp, z_floor

    def transform_world_point(p: np.ndarray, R_disp: np.ndarray, z_floor: float) -> np.ndarray:
        q = (R_disp @ p.reshape(3, 1)).reshape(3)
        q[2] -= z_floor
        return q

    def draw_camera_frustum(
        ax,
        T_wc: np.ndarray,
        R_disp: np.ndarray,
        z_floor: float,
        color: str,
        scale: float,
        lw: float = 1.6,
        center_size: float = 18.0,
        alpha: float = 0.95,
        linestyle: str = "-",
    ) -> None:
        R = T_wc[:3, :3]
        t = T_wc[:3, 3]
        C = -R.T @ t
        C = transform_world_point(C, R_disp, z_floor)
        r = R.T[:, 0]
        d = R.T[:, 1]
        f = R.T[:, 2]
        r = (R_disp @ r.reshape(3, 1)).reshape(3)
        d = (R_disp @ d.reshape(3, 1)).reshape(3)
        f = (R_disp @ f.reshape(3, 1)).reshape(3)
        r /= np.linalg.norm(r) + 1e-12
        d /= np.linalg.norm(d) + 1e-12
        f /= np.linalg.norm(f) + 1e-12

        depth = scale
        half_w = 0.7 * depth
        half_h = 0.45 * depth
        p1 = C + f * depth - r * half_w - d * half_h
        p2 = C + f * depth + r * half_w - d * half_h
        p3 = C + f * depth + r * half_w + d * half_h
        p4 = C + f * depth - r * half_w + d * half_h
        corners = [p1, p2, p3, p4]
        for k in range(4):
            a = corners[k]
            b = corners[(k + 1) % 4]
            ax.plot(
                [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=color, lw=lw, alpha=alpha, zorder=20, linestyle=linestyle
            )
            ax.plot(
                [C[0], a[0]], [C[1], a[1]], [C[2], a[2]],
                color=color, lw=lw * 0.9, alpha=alpha, zorder=20, linestyle=linestyle
            )
        f_tip = C + f * (1.35 * depth)
        ax.plot(
            [C[0], f_tip[0]], [C[1], f_tip[1]], [C[2], f_tip[2]],
            color=color, lw=lw * 1.35, alpha=alpha, zorder=21, linestyle=linestyle
        )
        ax.scatter([C[0]], [C[1]], [C[2]], color=color, s=center_size, alpha=alpha, zorder=22)

    def load_img(p: Path, max_w: int = 640) -> np.ndarray:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w > max_w:
            h = int(h * max_w / w)
            im = im.resize((max_w, h), Image.Resampling.BICUBIC)
        return np.asarray(im)

    def add_border(arr: np.ndarray, rgb: Tuple[int, int, int], t: int) -> np.ndarray:
        h, w = arr.shape[:2]
        out = np.zeros((h + 2 * t, w + 2 * t, 3), dtype=np.uint8)
        out[:, :] = np.array(rgb, dtype=np.uint8)
        out[t:t + h, t:t + w] = arr
        return out

    def match_width(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if a.shape[1] == b.shape[1]:
            return a, b
        if a.shape[1] < b.shape[1]:
            nh = int(a.shape[0] * b.shape[1] / a.shape[1])
            a = np.asarray(Image.fromarray(a).resize((b.shape[1], nh), Image.Resampling.BICUBIC))
        else:
            nh = int(b.shape[0] * a.shape[1] / b.shape[1])
            b = np.asarray(Image.fromarray(b).resize((a.shape[1], nh), Image.Resampling.BICUBIC))
        return a, b

    im0 = load_img(path_a)
    im1 = load_img(path_b)
    im0, im1 = match_width(im0, im1)
    t = max(6, min(12, im0.shape[0] // 48))
    pair_strip = np.concatenate(
        [add_border(im0, (46, 207, 255), t), add_border(im1, (230, 194, 0), t)],
        axis=0,
    )

    S, H, W, _ = world_pts.shape
    # Very dense visualization mode: start from full-resolution DUSt3R points.
    # DUSt3R provides one 3D point per pixel per view in PairViewer mode.
    stride = 1
    pts = world_pts[:, ::stride, ::stride, :].reshape(-1, 3)
    cols = rgb_imgs[:, ::stride, ::stride, :].reshape(-1, 3).astype(np.float32) / 255.0
    finite_mask = np.isfinite(pts).all(axis=1)
    if int(finite_mask.sum()) >= 500:
        pts = pts[finite_mask]
        cols = cols[finite_mask]
    # Keep a high cap to preserve density while avoiding extremely slow renders.
    if pts.shape[0] > 350000:
        sel = np.random.default_rng(0).choice(pts.shape[0], size=350000, replace=False)
        pts = pts[sel]
        cols = cols[sel]
    # Keep broader scene support; trim only very extreme outliers.
    q_lo = np.percentile(pts, 0.5, axis=0)
    q_hi = np.percentile(pts, 99.5, axis=0)
    core_mask = np.all((pts >= q_lo[None, :]) & (pts <= q_hi[None, :]), axis=1)
    if int(core_mask.sum()) >= 500:
        pts = pts[core_mask]
        cols = cols[core_mask]
    pts, R_disp, z_floor = orient_by_camera_pose(pts, T_render_a, T_render_b)

    fig = plt.figure(figsize=(20, 8), dpi=130)
    # Keep a true half-canvas right panel for reconstruction.
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.02)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(pair_strip)
    ax0.axis("off")
    ax0.set_title("Input Image Pair", fontsize=13)

    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    ax1.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=cols, s=1.1, alpha=0.62, linewidths=0, depthshade=False, zorder=1
    )
    pmin = np.percentile(pts, 2, axis=0)
    pmax = np.percentile(pts, 98, axis=0)
    ctr = 0.5 * (pmin + pmax)
    # Zoom out to ~0.7x scene size so camera frustums are clearly visible.
    rad = max(0.30, 0.32 * float(np.max(pmax - pmin) + 1e-6))
    ax1.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax1.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax1.set_zlim(ctr[2] - rad, ctr[2] + rad)
    ax1.set_box_aspect([1, 1, 1])
    ax1.view_init(elev=20, azim=-60)
    # Stronger perspective zoom than default.
    if hasattr(ax1, "set_proj_type"):
        ax1.set_proj_type("persp", focal_length=2.4)
    ax1.grid(False)
    ax1.set_axis_off()
    ax1.set_title("Reconstructed 3D Scene", fontsize=13)
    cam_scale = 0.12 * rad
    draw_camera_frustum(ax1, T_render_a, R_disp, z_floor, color="#2ecfff", scale=cam_scale, lw=1.9, center_size=22)
    draw_camera_frustum(ax1, T_render_b, R_disp, z_floor, color="#e6c200", scale=cam_scale, lw=1.9, center_size=22)
    # User-requested GT overlay:
    # - GT camera A is identical to blue camera A (no separate draw needed).
    # - Draw only GT camera B derived from GT relative pose wrt A.
    if T_gt_rel_for_overlay is not None:
        # Align GT relative translation scale to predicted relative translation scale
        # before anchoring at predicted A. This avoids GT-B appearing unrealistically far
        # when DUSt3R and GT are in different global scales.
        T_pred_rel = relative_pose_T(T_render_a, T_render_b)
        t_pred = T_pred_rel[:3, 3]
        t_gt = T_gt_rel_for_overlay[:3, 3]
        n_pred = float(np.linalg.norm(t_pred))
        n_gt = float(np.linalg.norm(t_gt))
        T_gt_rel_vis = T_gt_rel_for_overlay.copy()
        if n_gt > 1e-9 and n_pred > 1e-9:
            T_gt_rel_vis[:3, 3] = t_gt * (n_pred / n_gt)
        T_gt_b_anchor = T_gt_rel_vis @ T_render_a
        draw_camera_frustum(
            ax1,
            T_gt_b_anchor,
            R_disp,
            z_floor,
            color="#ff2020",
            scale=cam_scale * 1.25,
            lw=3.0,
            center_size=42,
            alpha=1.0,
            linestyle="--",
        )
    # Expand subplot area inside its grid cell (less whitespace around 3D axes).
    pos = ax1.get_position()
    ax1.set_position([pos.x0 - 0.06, pos.y0 - 0.03, pos.width + 0.03, pos.height + 0.08])

    fig.text(
        0.5,
        0.02,
        f"method={method}  pair_idx={pair_index}  RE={re_deg:.3f} deg  TE={te_deg:.3f} deg",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    droot = args.dust3r_root.resolve()
    if str(droot) not in sys.path:
        sys.path.insert(0, str(droot))
    from dust3r.model import load_model

    row = parse_metrics_row(args.metrics_csv.resolve(), args.pair_index)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(str(args.checkpoint.resolve()), device=str(device), verbose=True)
    model.eval()

    re_deg = float("nan")
    te_deg = float("nan")
    T_gt_rel_for_overlay: Optional[np.ndarray] = None
    path_a: Path
    path_b: Path

    if args.method == "dust3r":
        rel_a = row.get("rel_a", "")
        rel_b = row.get("rel_b", "")
        if not rel_a or not rel_b:
            raise RuntimeError("dust3r mode requires rel_a and rel_b columns in metrics CSV.")
        path_a = (args.dl3dv_root.resolve() / rel_a).resolve()
        path_b = (args.dl3dv_root.resolve() / rel_b).resolve()
        T_pred_a, T_pred_b, world_pts, rgb_imgs = run_dust3r_pair(
            model, device, path_a, path_b, image_size=args.image_size
        )

        if "rot_err_deg" in row and "trans_dir_err_deg" in row:
            re_deg = float(row["rot_err_deg"])
            te_deg = float(row["trans_dir_err_deg"])
            # Still load GT rel for requested GT-B overlay.
            T_gt_a = load_gt_T_wc_from_rel(args.dl3dv_root.resolve(), rel_a)
            T_gt_b = load_gt_T_wc_from_rel(args.dl3dv_root.resolve(), rel_b)
            T_gt_rel_for_overlay = relative_pose_T(T_gt_a, T_gt_b)
        else:
            T_gt_a = load_gt_T_wc_from_rel(args.dl3dv_root.resolve(), rel_a)
            T_gt_b = load_gt_T_wc_from_rel(args.dl3dv_root.resolve(), rel_b)
            rel_pr = relative_pose_T(T_pred_a, T_pred_b)
            rel_gt = relative_pose_T(T_gt_a, T_gt_b)
            T_gt_rel_for_overlay = rel_gt
            re_deg = geodesic_rotation_error_deg(rel_pr[:3, :3], rel_gt[:3, :3])
            te_deg = translation_direction_angle_error_deg(rel_pr[:3, 3], rel_gt[:3, 3])

    else:
        interpose_root = args.interpose_root.resolve()
        selected_jsonl = (
            args.interpose_selected_jsonl.resolve()
            if args.interpose_selected_jsonl is not None
            else (interpose_root / "selection" / "selected_per_pair.jsonl").resolve()
        )
        pair_dir, T_rel_selected = load_interpose_selected(interpose_root, args.pair_index, selected_jsonl)

        # InterPose pipeline images (already in the variant folder, no extra files generated).
        selected_variant = None
        with selected_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if Path(rec.get("pair_dir", "")).resolve() == pair_dir:
                    selected_variant = str(rec.get("selected_variant", "v0_A2B_p1_s1"))
                    break
        if selected_variant is None:
            raise RuntimeError(f"Could not locate selected variant for {pair_dir}")

        variant_dir = pair_dir / selected_variant
        path_a = variant_dir / "pre_model_inputs" / "00_start_pre_model.png"
        path_b = variant_dir / "pre_model_inputs" / "01_end_pre_model.png"
        T_pred_a, T_pred_b, world_pts, rgb_imgs = run_dust3r_pair(
            model, device, path_a, path_b, image_size=args.image_size
        )
        # Keep InterPose selected relative pose as camera-pose source in this branch.
        _ = T_rel_selected

        if "interpose_RE_deg" in row and "interpose_TE_deg" in row:
            re_deg = float(row["interpose_RE_deg"])
            te_deg = float(row["interpose_TE_deg"])
        elif "rot_err_deg" in row and "trans_dir_err_deg" in row:
            re_deg = float(row["rot_err_deg"])
            te_deg = float(row["trans_dir_err_deg"])

    make_figure(
        out_path=args.output_image.resolve(),
        path_a=path_a,
        path_b=path_b,
        world_pts=world_pts,
        rgb_imgs=rgb_imgs,
        T_render_a=T_pred_a,
        T_render_b=T_pred_b,
        T_gt_rel_for_overlay=T_gt_rel_for_overlay,
        method=args.method,
        pair_index=args.pair_index,
        re_deg=re_deg,
        te_deg=te_deg,
    )
    print(f"Saved image: {args.output_image.resolve()}")


if __name__ == "__main__":
    main()
