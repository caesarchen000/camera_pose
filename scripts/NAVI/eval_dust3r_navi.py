#!/usr/bin/env python3
"""
Evaluate DUSt3R on NAVI pairs and save per-pair camera poses + errors.

Inputs:
  - NAVI root
  - NAVI pair CSV (e.g. outputs/NAVI/navi_pairs_yaw50_90_300.csv)
  - DUSt3R checkpoint

Outputs:
  - Per-pair camera poses JSONL (GT + pred, world->camera)
  - Per-pair metrics CSV
  - Printed aggregate metrics (MRE/MTE/R_acc/t_acc/AUC30)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.DL3DV.eval_dust3r_dl3dv import (  # noqa: E402
    geodesic_rotation_error_deg,
    translation_direction_angle_error_deg,
    acc_at_deg,
    auc30,
    auc30_paper,
    relative_pose_T,
    T_to_Rt,
)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    qw, qx, qy, qz = q.astype(np.float64)
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def build_navi_gt_pose_index(navi_root: Path) -> Dict[str, np.ndarray]:
    """
    Build rel_path -> T_wc mapping from NAVI annotations.

    We assume NAVI camera.q/camera.t are camera extrinsics (world->camera).
    """
    pose_index: Dict[str, np.ndarray] = {}
    for obj_dir in sorted([p for p in navi_root.iterdir() if p.is_dir()]):
        object_id = obj_dir.name
        for scene_dir in sorted([p for p in obj_dir.iterdir() if p.is_dir()]):
            ann = scene_dir / "annotations.json"
            if not ann.is_file():
                continue
            data = json.loads(ann.read_text())
            scene_name = scene_dir.name
            for rec in data:
                filename = rec.get("filename")
                cam = rec.get("camera", {})
                q = np.array(cam.get("q", []), dtype=np.float64)
                t = np.array(cam.get("t", []), dtype=np.float64)
                if filename is None or q.shape != (4,) or t.shape != (3,):
                    continue
                R_wc = quaternion_to_rotation_matrix(q)
                T_wc = np.eye(4, dtype=np.float64)
                T_wc[:3, :3] = R_wc
                T_wc[:3, 3] = t
                rel = f"{object_id}/{scene_name}/images/{filename}"
                pose_index[rel] = T_wc
    return pose_index


def run_dust3r_pair_pose_optimized(
    model,
    device: torch.device,
    path_a: Path,
    path_b: Path,
    image_size: int,
    niter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run DUSt3R pair inference and recover relative pose from dense 3D-3D matches.
    Returns T_wc (world->camera) for the two views in a pair-local gauge:
      - T_wc_a = I
      - T_wc_b = transform from cam-a to cam-b.
    """
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

    # DUSt3R semantics:
    # - pred1[e]['pts3d'] are points of view1[e] in view1[e] frame.
    # - pred2[e]['pts3d_in_other_view'] are points of view2[e] in view1[e] frame.
    #
    # With symmetrize=True and two images (A,B), batch entries are:
    #   e=0: view1=A, view2=B
    #   e=1: view1=B, view2=A
    #
    # Therefore for pixels of image B we can form aligned correspondences:
    #   x_A = pred2[0]['pts3d_in_other_view']  (B pixels in A frame)
    #   x_B = pred1[1]['pts3d']                (B pixels in B frame)
    # and solve x_B ~= R_BA * x_A + t_BA.
    x_a = (
        output["pred2"]["pts3d_in_other_view"][0]
        .reshape(-1, 3)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    x_b = (
        output["pred1"]["pts3d"][1]
        .reshape(-1, 3)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    c_a = (
        output["pred2"]["conf"][0].reshape(-1).detach().cpu().numpy().astype(np.float64)
    )
    c_b = (
        output["pred1"]["conf"][1].reshape(-1).detach().cpu().numpy().astype(np.float64)
    )
    w = np.sqrt(np.clip(c_a, 1e-8, None) * np.clip(c_b, 1e-8, None))

    finite = (
        np.isfinite(x_a).all(axis=1)
        & np.isfinite(x_b).all(axis=1)
        & np.isfinite(w)
        & (w > 0)
    )
    x_a = x_a[finite]
    x_b = x_b[finite]
    w = w[finite]
    if x_a.shape[0] < 16:
        raise RuntimeError("Not enough valid dense correspondences for pose recovery")

    # Keep only top-confidence correspondences for robustness/speed.
    max_pts = 50000
    if x_a.shape[0] > max_pts:
        idx = np.argpartition(w, -max_pts)[-max_pts:]
        x_a = x_a[idx]
        x_b = x_b[idx]
        w = w[idx]

    def weighted_rigid(a: np.ndarray, b: np.ndarray, ww: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ww = ww / (ww.sum() + 1e-12)
        ma = (a * ww[:, None]).sum(axis=0)
        mb = (b * ww[:, None]).sum(axis=0)
        ac = a - ma
        bc = b - mb
        H = (ww[:, None] * ac).T @ bc
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T
        t = mb - R @ ma
        return R, t

    R, t = weighted_rigid(x_a, x_b, w)
    T_ba = np.eye(4, dtype=np.float64)
    T_ba[:3, :3] = R
    T_ba[:3, 3] = t
    T_ab = np.linalg.inv(T_ba)
    T0 = np.eye(4, dtype=np.float64)
    T1 = np.eye(4, dtype=np.float64)
    T1[:] = T_ab
    return T0, T1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DUSt3R on NAVI pairs.")
    parser.add_argument(
        "--navi_root",
        type=Path,
        default=Path("datasets/NAVI/navi_v1.5"),
        help="NAVI dataset root.",
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        default=Path("outputs/NAVI/navi_pairs_yaw50_90_300.csv"),
        help="Pair CSV from scripts/NAVI/make_navi_pairs.py.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
        help="DUSt3R checkpoint path.",
    )
    parser.add_argument(
        "--dust3r_root",
        type=Path,
        default=Path("dust3r"),
        help="Local DUSt3R repo root.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        choices=(224, 512),
    )
    parser.add_argument(
        "--align_niter",
        type=int,
        default=100,
        help="DUSt3R global alignment iterations per pair.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Optional cap on rows from pairs CSV (0 = all).",
    )
    parser.add_argument(
        "--save_metrics_csv",
        type=Path,
        default=Path("outputs/NAVI/dust3r_navi_metrics_300.csv"),
        help="Per-pair metric CSV output path.",
    )
    parser.add_argument(
        "--save_pose_jsonl",
        type=Path,
        default=Path("outputs/NAVI/dust3r_navi_pair_poses_300.jsonl"),
        help="Per-pair GT/pred camera pose JSONL output path.",
    )
    args = parser.parse_args()

    navi_root = args.navi_root.resolve()
    pairs_csv = args.pairs_csv.resolve()
    checkpoint = args.checkpoint.resolve()
    dust3r_root = args.dust3r_root.resolve()
    save_metrics_csv = args.save_metrics_csv.resolve()
    save_pose_jsonl = args.save_pose_jsonl.resolve()

    if str(dust3r_root) not in sys.path:
        sys.path.insert(0, str(dust3r_root))
    from dust3r.model import load_model  # noqa: E402

    pair_rows = list(csv.DictReader(pairs_csv.open()))
    if args.max_pairs > 0:
        pair_rows = pair_rows[: args.max_pairs]
    if not pair_rows:
        raise RuntimeError(f"No pair rows found in {pairs_csv}")

    print(f"[info] Building NAVI GT pose index from {navi_root} ...")
    gt_pose_index = build_navi_gt_pose_index(navi_root)
    print(f"[info] GT indexed images: {len(gt_pose_index)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] Loading DUSt3R checkpoint: {checkpoint}")
    model = load_model(str(checkpoint), device=str(device), verbose=True)
    model.eval()

    save_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    save_pose_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rot_errs = []
    trans_errs = []
    metric_rows = []
    pose_rows = []

    for row in tqdm(pair_rows, desc="Evaluating NAVI pairs", unit="pair"):
        pair_idx = int(row["pair_idx"])
        rel_a = row["rel_a"]
        rel_b = row["rel_b"]
        img_a = (navi_root / rel_a).resolve()
        img_b = (navi_root / rel_b).resolve()
        if not img_a.is_file() or not img_b.is_file():
            print(f"[skip] pair_idx={pair_idx} missing image")
            continue
        if rel_a not in gt_pose_index or rel_b not in gt_pose_index:
            print(f"[skip] pair_idx={pair_idx} missing GT pose index")
            continue

        T_gt_a = gt_pose_index[rel_a]
        T_gt_b = gt_pose_index[rel_b]

        try:
            T_pred_a, T_pred_b = run_dust3r_pair_pose_optimized(
                model=model,
                device=device,
                path_a=img_a,
                path_b=img_b,
                image_size=args.image_size,
                niter=args.align_niter,
            )
        except Exception as e:
            print(f"[skip] pair_idx={pair_idx} DUSt3R failed: {e}")
            continue
        if np.allclose(T_pred_a, np.eye(4), atol=1e-7) and np.allclose(
            T_pred_b, np.eye(4), atol=1e-7
        ):
            print(f"[skip] pair_idx={pair_idx} DUSt3R returned identity poses")
            continue

        rel_pr = relative_pose_T(T_pred_a, T_pred_b)
        rel_gt = relative_pose_T(T_gt_a, T_gt_b)
        R_pr, t_pr = T_to_Rt(rel_pr)
        R_gt, t_gt = T_to_Rt(rel_gt)
        re = geodesic_rotation_error_deg(R_pr, R_gt)
        te = translation_direction_angle_error_deg(t_pr, t_gt)

        rot_errs.append(re)
        trans_errs.append(te)

        metric_rows.append(
            {
                "pair_idx": pair_idx,
                "object_id": row.get("object_id", ""),
                "scene_a": row.get("scene_a", ""),
                "scene_b": row.get("scene_b", ""),
                "rel_a": rel_a,
                "rel_b": rel_b,
                "yaw_diff_deg": row.get("yaw_diff_deg", ""),
                "rot_err_deg": re,
                "trans_dir_err_deg": te,
            }
        )
        pose_rows.append(
            {
                "pair_idx": pair_idx,
                "rel_a": rel_a,
                "rel_b": rel_b,
                "T_wc_gt_a": T_gt_a.tolist(),
                "T_wc_gt_b": T_gt_b.tolist(),
                "T_wc_pred_a": T_pred_a.tolist(),
                "T_wc_pred_b": T_pred_b.tolist(),
            }
        )

    if not metric_rows:
        raise RuntimeError("No valid NAVI pairs evaluated.")

    with save_metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_idx",
                "object_id",
                "scene_a",
                "scene_b",
                "rel_a",
                "rel_b",
                "yaw_diff_deg",
                "rot_err_deg",
                "trans_dir_err_deg",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    with save_pose_jsonl.open("w") as f:
        for r in pose_rows:
            f.write(json.dumps(r) + "\n")

    rot_e = np.array(rot_errs, dtype=np.float64)
    trans_e = np.array(trans_errs, dtype=np.float64)

    print(f"\nSaved per-pair metrics: {save_metrics_csv}")
    print(f"Saved per-pair poses:   {save_pose_jsonl}")
    print(f"Evaluated pairs: {len(metric_rows)}")
    print(f"MRE_deg: {float(np.nanmean(rot_e)):.6g}")
    print(f"MTE_deg: {float(np.nanmean(trans_e)):.6g}")
    print(f"R_acc_5:  {acc_at_deg(rot_e, 5.0):.6g}")
    print(f"R_acc_15: {acc_at_deg(rot_e, 15.0):.6g}")
    print(f"R_acc_30: {acc_at_deg(rot_e, 30.0):.6g}")
    print(f"t_acc_5:  {acc_at_deg(trans_e, 5.0):.6g}")
    print(f"t_acc_15: {acc_at_deg(trans_e, 15.0):.6g}")
    print(f"t_acc_30: {acc_at_deg(trans_e, 30.0):.6g}")
    print(f"AUC30: {auc30_paper(rot_e, trans_e):.6g}")
    print(f"AUC30_rot: {auc30(rot_e):.6g}")
    print(f"AUC30_trans: {auc30(trans_e):.6g}")


if __name__ == "__main__":
    main()

