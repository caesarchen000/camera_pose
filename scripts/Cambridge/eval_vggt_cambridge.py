#!/usr/bin/env python3
"""
Evaluate VGGT on Cambridge Landmarks image pairs.

Inputs:
  - Cambridge dataset root
  - pair CSV from scripts/Cambridge/make_cambridge_pairs.py
  - VGGT checkpoint or HuggingFace model id

Outputs:
  - per-pair metric CSV
  - per-pair GT/pred pose JSONL
  - printed aggregate relative-pose metrics

For each pair (A, B):
  1. run VGGT on the two input images to get two predicted camera poses
  2. compute predicted relative pose of B with respect to A:
       T_pred_ba = T_pred_b @ inv(T_pred_a)
  3. compute GT relative pose the same way:
       T_gt_ba = T_gt_b @ inv(T_gt_a)
  4. evaluate rotation geodesic error and translation-direction angle error
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
    acc_at_deg,
    auc30,
    auc30_paper,
    geodesic_rotation_error_deg,
    relative_pose_T,
    T_to_Rt,
    translation_direction_angle_error_deg,
)
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


@dataclass(frozen=True)
class CambridgeGtFrame:
    rel_path: str
    scene_name: str
    sequence_name: str
    split_name: str
    T_wc: np.ndarray


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


def build_cambridge_gt_pose_index(
    cambridge_root: Path,
    scene_names: Optional[Iterable[str]] = None,
) -> Dict[str, CambridgeGtFrame]:
    wanted = set(scene_names) if scene_names else None
    pose_index: Dict[str, CambridgeGtFrame] = {}

    for scene_dir in sorted(p for p in cambridge_root.iterdir() if p.is_dir()):
        scene_name = scene_dir.name
        if wanted is not None and scene_name not in wanted:
            continue

        for split_file in ("dataset_train.txt", "dataset_test.txt"):
            pose_path = scene_dir / split_file
            if not pose_path.is_file():
                continue
            split_name = "train" if "train" in split_file else "test"

            for line in pose_path.read_text().splitlines()[3:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue

                rel_img = parts[0]
                rel_path = f"{scene_name}/{rel_img}"
                abs_img = cambridge_root / rel_path
                if not abs_img.is_file():
                    continue

                cam_center_w = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])],
                    dtype=np.float64,
                )
                q_wc = np.array(
                    [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                    dtype=np.float64,
                )
                R_wc = quaternion_to_rotation_matrix(q_wc)
                t_wc = -R_wc @ cam_center_w

                T_wc = np.eye(4, dtype=np.float64)
                T_wc[:3, :3] = R_wc
                T_wc[:3, 3] = t_wc

                sequence_name = Path(rel_img).parts[0]
                pose_index[rel_path] = CambridgeGtFrame(
                    rel_path=rel_path,
                    scene_name=scene_name,
                    sequence_name=sequence_name,
                    split_name=split_name,
                    T_wc=T_wc,
                )

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


def maybe_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None:
        return float("nan")
    value = str(value).strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VGGT on Cambridge pairs.")
    parser.add_argument(
        "--cambridge_root",
        type=Path,
        default=Path("datasets/cambridge"),
        help="Cambridge dataset root.",
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        default=Path("outputs/Cambridge/cambridge_pairs_yaw50_65_300.csv"),
        help="Pair CSV from scripts/Cambridge/make_cambridge_pairs.py.",
    )
    parser.add_argument(
        "--vggt_checkpoint",
        type=str,
        default="facebook/VGGT-1B",
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
        default=Path("outputs/Cambridge/vggt_cambridge_yaw50_65_allscene.csv"),
        help="Per-pair metric CSV output path.",
    )
    parser.add_argument(
        "--save_pose_jsonl",
        type=Path,
        default=Path("outputs/Cambridge/vggt_cambridge_pair_poses_yaw50_65_allscene.jsonl"),
        help="Per-pair GT/pred camera pose JSONL output path.",
    )
    parser.add_argument(
        "--save_summary_json",
        type=Path,
        default=Path("outputs/Cambridge/vggt_cambridge_yaw50_65_summary.json"),
        help="Aggregate metric JSON output path.",
    )
    args = parser.parse_args()

    cambridge_root = args.cambridge_root.resolve()
    pairs_csv = args.pairs_csv.resolve()
    save_metrics_csv = args.save_metrics_csv.resolve()
    save_pose_jsonl = args.save_pose_jsonl.resolve()
    save_summary_json = args.save_summary_json.resolve()

    if not cambridge_root.is_dir():
        raise FileNotFoundError(f"Cambridge root not found: {cambridge_root}")
    if not pairs_csv.is_file():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")

    pair_rows = list(csv.DictReader(pairs_csv.open()))
    if args.max_pairs > 0:
        pair_rows = pair_rows[: args.max_pairs]
    if not pair_rows:
        raise RuntimeError(f"No pair rows found in {pairs_csv}")

    scenes_from_csv = sorted(
        {
            str(row.get("scene_name", "")).strip()
            for row in pair_rows
            if str(row.get("scene_name", "")).strip()
        }
    )

    print(f"[info] Building Cambridge GT pose index from {cambridge_root} ...")
    gt_pose_index = build_cambridge_gt_pose_index(
        cambridge_root=cambridge_root,
        scene_names=scenes_from_csv if scenes_from_csv else None,
    )
    print(f"[info] GT indexed images: {len(gt_pose_index)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] Loading VGGT checkpoint: {args.vggt_checkpoint}")
    model = load_vggt_model(args.vggt_checkpoint, device)

    save_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    save_pose_jsonl.parent.mkdir(parents=True, exist_ok=True)
    save_summary_json.parent.mkdir(parents=True, exist_ok=True)

    rot_errs: List[float] = []
    trans_errs: List[float] = []
    metric_rows: List[Dict[str, object]] = []
    pose_rows: List[Dict[str, object]] = []

    for row in tqdm(pair_rows, desc="Evaluating Cambridge pairs (VGGT)", unit="pair"):
        pair_idx = int(row["pair_idx"])
        rel_a = str(row["rel_a"]).strip()
        rel_b = str(row["rel_b"]).strip()
        img_a = (cambridge_root / rel_a).resolve()
        img_b = (cambridge_root / rel_b).resolve()

        if not img_a.is_file() or not img_b.is_file():
            print(f"[skip] pair_idx={pair_idx} missing image")
            continue
        if rel_a not in gt_pose_index or rel_b not in gt_pose_index:
            print(f"[skip] pair_idx={pair_idx} missing GT pose index")
            continue

        T_gt_a = gt_pose_index[rel_a].T_wc
        T_gt_b = gt_pose_index[rel_b].T_wc

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
                "scene_name": row.get("scene_name", ""),
                "sequence_name": row.get("sequence_name", ""),
                "split_a": row.get("split_a", ""),
                "split_b": row.get("split_b", ""),
                "rel_a": rel_a,
                "rel_b": rel_b,
                "yaw_diff_deg": maybe_float(row, "yaw_diff_deg"),
                "pitch_diff_deg": maybe_float(row, "pitch_diff_deg"),
                "geodesic_rot_deg": maybe_float(row, "geodesic_rot_deg"),
                "baseline": maybe_float(row, "baseline"),
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
                "T_rel_gt_ba": rel_gt.tolist(),
                "T_rel_pred_ba": rel_pr.tolist(),
            }
        )

    if not metric_rows:
        raise RuntimeError("No valid Cambridge pairs evaluated.")

    with save_metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_idx",
                "scene_name",
                "sequence_name",
                "split_a",
                "split_b",
                "rel_a",
                "rel_b",
                "yaw_diff_deg",
                "pitch_diff_deg",
                "geodesic_rot_deg",
                "baseline",
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
    summary = {
        "Evaluated_pairs": int(len(metric_rows)),
        "MRE_deg": float(np.nanmean(rot_e)),
        "median_RE_deg": float(np.nanmedian(rot_e)),
        "R_acc_5": acc_at_deg(rot_e, 5.0),
        "R_acc_15": acc_at_deg(rot_e, 15.0),
        "R_acc_30": acc_at_deg(rot_e, 30.0),
        "MTE_deg": float(np.nanmean(trans_e)),
        "median_TE_deg": float(np.nanmedian(trans_e)),
        "t_acc_5": acc_at_deg(trans_e, 5.0),
        "t_acc_15": acc_at_deg(trans_e, 15.0),
        "t_acc_30": acc_at_deg(trans_e, 30.0),
        "AUC30": auc30_paper(rot_e, trans_e),
        "AUC30_rot": auc30(rot_e),
        "AUC30_trans": auc30(trans_e),
    }
    save_summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nSaved per-pair metrics: {save_metrics_csv}")
    print(f"Saved per-pair poses:   {save_pose_jsonl}")
    print(f"Saved summary JSON:    {save_summary_json}")
    print(f"Evaluated pairs: {summary['Evaluated_pairs']}")
    print(f"MRE_deg: {summary['MRE_deg']:.6g}")
    print(f"median_RE_deg: {summary['median_RE_deg']:.6g}")
    print(f"MTE_deg: {summary['MTE_deg']:.6g}")
    print(f"median_TE_deg: {summary['median_TE_deg']:.6g}")
    print(f"R_acc_5:  {summary['R_acc_5']:.6g}")
    print(f"R_acc_15: {summary['R_acc_15']:.6g}")
    print(f"R_acc_30: {summary['R_acc_30']:.6g}")
    print(f"t_acc_5:  {summary['t_acc_5']:.6g}")
    print(f"t_acc_15: {summary['t_acc_15']:.6g}")
    print(f"t_acc_30: {summary['t_acc_30']:.6g}")
    print(f"AUC30: {summary['AUC30']:.6g}")
    print(f"AUC30_rot: {summary['AUC30_rot']:.6g}")
    print(f"AUC30_trans: {summary['AUC30_trans']:.6g}")


if __name__ == "__main__":
    main()
