#!/usr/bin/env python3
"""
Evaluate VGGT on NAVI pairs and save per-pair camera poses + errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_VGGT_VENDOR_ROOT = _REPO_ROOT / "vggt"
if _VGGT_VENDOR_ROOT.is_dir() and str(_VGGT_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_VENDOR_ROOT))

from scripts.DL3DV.eval_dust3r_dl3dv import (  # noqa: E402
    geodesic_rotation_error_deg,
    translation_direction_angle_error_deg,
    acc_at_deg,
    auc30,
    auc30_paper,
    relative_pose_T,
    T_to_Rt,
)
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
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


def to_homogeneous_4x4(T: np.ndarray) -> np.ndarray:
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        out = np.eye(4, dtype=T.dtype)
        out[:3, :] = T
        return out
    raise ValueError(f"Unsupported transform shape {T.shape}")


def run_vggt_pair_pose(
    model: VGGT,
    device: torch.device,
    path_a: Path,
    path_b: Path,
) -> tuple[np.ndarray, np.ndarray]:
    images = load_and_preprocess_images([str(path_a), str(path_b)]).to(device)
    with torch.no_grad():
        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            images_batched = images[None]
            aggregated_tokens_list, _ = model.aggregator(images_batched)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    extrinsic = extrinsic.squeeze(0).detach().float().cpu().numpy()
    T_wc_a = to_homogeneous_4x4(extrinsic[0]).astype(np.float64)
    T_wc_b = to_homogeneous_4x4(extrinsic[1]).astype(np.float64)
    return T_wc_a, T_wc_b


def load_vggt_model(checkpoint: str, device: torch.device) -> VGGT:
    if checkpoint.endswith(".pt"):
        model = VGGT()
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    else:
        model = VGGT.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VGGT on NAVI pairs.")
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
        "--vggt_checkpoint",
        type=str,
        default="checkpoints/vggt.pt",
        help='VGGT checkpoint: local .pt or HF id (e.g. "facebook/VGGT-1B").',
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
        default=Path("outputs/NAVI/vggt_navi_metrics_300.csv"),
        help="Per-pair metric CSV output path.",
    )
    parser.add_argument(
        "--save_pose_jsonl",
        type=Path,
        default=Path("outputs/NAVI/vggt_navi_pair_poses_300.jsonl"),
        help="Per-pair GT/pred camera pose JSONL output path.",
    )
    args = parser.parse_args()

    navi_root = args.navi_root.resolve()
    pairs_csv = args.pairs_csv.resolve()
    save_metrics_csv = args.save_metrics_csv.resolve()
    save_pose_jsonl = args.save_pose_jsonl.resolve()

    pair_rows = list(csv.DictReader(pairs_csv.open()))
    if args.max_pairs > 0:
        pair_rows = pair_rows[: args.max_pairs]
    if not pair_rows:
        raise RuntimeError(f"No pair rows found in {pairs_csv}")

    print(f"[info] Building NAVI GT pose index from {navi_root} ...")
    gt_pose_index = build_navi_gt_pose_index(navi_root)
    print(f"[info] GT indexed images: {len(gt_pose_index)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] Loading VGGT checkpoint: {args.vggt_checkpoint}")
    model = load_vggt_model(args.vggt_checkpoint, device)

    save_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    save_pose_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rot_errs = []
    trans_errs = []
    metric_rows = []
    pose_rows = []

    for row in tqdm(pair_rows, desc="Evaluating NAVI pairs (VGGT)", unit="pair"):
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
            T_pred_a, T_pred_b = run_vggt_pair_pose(
                model=model,
                device=device,
                path_a=img_a,
                path_b=img_b,
            )
        except Exception as e:
            print(f"[skip] pair_idx={pair_idx} VGGT failed: {e}")
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

