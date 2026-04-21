#!/usr/bin/env python3
"""
Evaluate DUSt3R on Cambridge Landmarks image pairs.

Inputs:
  - Cambridge dataset root
  - pair CSV from scripts/Cambridge/make_cambridge_pairs.py
  - DUSt3R checkpoint

Outputs:
  - per-pair metric CSV
  - per-pair GT/pred pose JSONL
  - printed aggregate relative-pose metrics

For each pair (A, B):
  1. run DUSt3R on the two input images to get two predicted camera poses
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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.DL3DV.eval_dust3r_dl3dv import (  # noqa: E402
    acc_at_deg,
    auc30,
    auc30_paper,
    geodesic_rotation_error_deg,
    relative_pose_T,
    run_dust3r_pair_pose,
    translation_direction_angle_error_deg,
    T_to_Rt,
)


@dataclass(frozen=True)
class CambridgeGtFrame:
    rel_path: str
    scene_name: str
    sequence_name: str
    split_name: str
    T_wc: np.ndarray


@dataclass(frozen=True)
class MetadataQuatFrame:
    rel_path: str
    scene_name: str
    q_wxyz: np.ndarray


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


def build_pose_from_quaternion(q_wxyz: np.ndarray, quat_convention: str) -> np.ndarray:
    R = quaternion_to_rotation_matrix(q_wxyz)
    R_wc = R if quat_convention == "w2c" else R.T
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    return T_wc


def normalize_metadata_rel_path(raw_path: str) -> str:
    return Path(raw_path.replace("\\", "/").lstrip("./")).as_posix()


def build_metadata_quat_index(
    metadata_npy: Path,
    scene_names: Optional[Iterable[str]] = None,
) -> Dict[str, MetadataQuatFrame]:
    wanted = set(scene_names) if scene_names else None
    metadata_obj = np.load(metadata_npy, allow_pickle=True).item()
    if not isinstance(metadata_obj, dict):
        raise ValueError(f"Unexpected metadata format in {metadata_npy}")
    quat_index: Dict[str, MetadataQuatFrame] = {}
    for pair in metadata_obj.values():
        if not isinstance(pair, dict):
            continue
        for image_key in ("img1", "img2"):
            image_meta = pair.get(image_key, {})
            if not isinstance(image_meta, dict):
                continue
            rel_path = normalize_metadata_rel_path(str(image_meta.get("path", "")).strip())
            if not rel_path or rel_path in quat_index:
                continue
            scene_name = str(pair.get("scene", "")).strip() or Path(rel_path).parts[0]
            if wanted is not None and scene_name not in wanted:
                continue
            quat_index[rel_path] = MetadataQuatFrame(
                rel_path=rel_path,
                scene_name=scene_name,
                q_wxyz=np.array(
                    [
                        float(image_meta.get("qw", 1.0)),
                        float(image_meta.get("qx", 0.0)),
                        float(image_meta.get("qy", 0.0)),
                        float(image_meta.get("qz", 0.0)),
                    ],
                    dtype=np.float64,
                ),
            )
    return quat_index


def build_cambridge_gt_pose_index(
    cambridge_root: Path,
    scene_names: Optional[Iterable[str]] = None,
) -> Dict[str, CambridgeGtFrame]:
    """
    Build rel_path -> GT world-to-camera pose index from Cambridge pose text files.

    Cambridge rows are:
      ImageFile, Camera Position [X Y Z W P Q R]

    We treat [W P Q R] as a world->camera quaternion, matching the pair-building
    script and yielding a consistent camera-forward convention for Cambridge.
    """
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

                cam_center_w = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                q_wc = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
                R_wc = quaternion_to_rotation_matrix(q_wc)
                # Cambridge stores camera center in world coordinates. Convert it to
                # the world->camera translation term: x_cam = R_wc * x_world + t_wc,
                # so t_wc = -R_wc * C_world.
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


def resolve_image_path(rel_path: str, cambridge_root: Path, metadata_image_root: Optional[Path]) -> Path:
    candidates = [cambridge_root / rel_path]
    if metadata_image_root is not None:
        candidates.append(metadata_image_root / rel_path)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return candidates[0].resolve()


def resolve_gt_pose(
    rel_path: str,
    quat_convention: str,
    gt_pose_index: Dict[str, CambridgeGtFrame],
    metadata_quat_index: Dict[str, MetadataQuatFrame],
) -> Optional[np.ndarray]:
    if rel_path in gt_pose_index:
        return gt_pose_index[rel_path].T_wc
    m = metadata_quat_index.get(rel_path)
    if m is None:
        return None
    return build_pose_from_quaternion(m.q_wxyz, quat_convention)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DUSt3R on Cambridge pairs.")
    parser.add_argument(
        "--cambridge_root",
        type=Path,
        default=Path("datasets/cambridge"),
        help="Cambridge dataset root.",
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        default=Path("outputs/Cambridge/cambridge_pairs_selp_metadata_only_strict_yaw50_65.csv"),
        help="Pair CSV from scripts/Cambridge/make_cambridge_pairs.py.",
    )
    parser.add_argument(
        "--metadata_npy",
        type=Path,
        default=Path("metadata/metadata/selp_test_set.npy"),
        help="Optional sELP metadata .npy used for fallback GT quaternions.",
    )
    parser.add_argument(
        "--metadata_image_root",
        type=Path,
        default=Path("metadata/metadata/images_to_npys/test_scenes_images/selp"),
        help="Optional root for metadata-backed images such as Street.",
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
        default=Path("outputs/Cambridge/dust3r_cambridge_yaw50_65_allscene.csv"),
        help="Per-pair metric CSV output path.",
    )
    parser.add_argument(
        "--save_pose_jsonl",
        type=Path,
        default=Path("outputs/Cambridge/dust3r_cambridge_pair_poses_yaw50_65_allscene.jsonl"),
        help="Per-pair GT/pred pose JSONL output path.",
    )
    parser.add_argument(
        "--save_summary_json",
        type=Path,
        default=Path("outputs/Cambridge/dust3r_cambridge_yaw50_65_summary.json"),
        help="Aggregate metric JSON output path.",
    )
    args = parser.parse_args()

    cambridge_root = args.cambridge_root.resolve()
    pairs_csv = args.pairs_csv.resolve()
    metadata_npy = args.metadata_npy.resolve()
    metadata_image_root = args.metadata_image_root.resolve()
    checkpoint = args.checkpoint.resolve()
    dust3r_root = args.dust3r_root.resolve()
    save_metrics_csv = args.save_metrics_csv.resolve()
    save_pose_jsonl = args.save_pose_jsonl.resolve()
    save_summary_json = args.save_summary_json.resolve()

    if not cambridge_root.is_dir():
        raise FileNotFoundError(f"Cambridge root not found: {cambridge_root}")
    if not pairs_csv.is_file():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DUSt3R checkpoint not found: {checkpoint}")
    if not metadata_image_root.is_dir():
        print(f"[warn] Metadata image root not found, fallback disabled: {metadata_image_root}")
        metadata_image_root = None

    if str(dust3r_root) not in sys.path:
        sys.path.insert(0, str(dust3r_root))
    from dust3r.model import load_model  # noqa: E402

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
    metadata_quat_index: Dict[str, MetadataQuatFrame] = {}
    if metadata_npy.is_file():
        print(f"[info] Building metadata quaternion index from {metadata_npy} ...")
        metadata_quat_index = build_metadata_quat_index(
            metadata_npy=metadata_npy,
            scene_names=scenes_from_csv if scenes_from_csv else None,
        )
        print(f"[info] Metadata indexed images: {len(metadata_quat_index)}")
    else:
        print(f"[warn] Metadata .npy not found, fallback GT disabled: {metadata_npy}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] Loading DUSt3R checkpoint: {checkpoint}")
    model = load_model(str(checkpoint), device=str(device), verbose=True)
    model.eval()

    save_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    save_pose_jsonl.parent.mkdir(parents=True, exist_ok=True)
    save_summary_json.parent.mkdir(parents=True, exist_ok=True)

    rot_errs: List[float] = []
    trans_errs: List[float] = []
    metric_rows: List[Dict[str, object]] = []
    pose_rows: List[Dict[str, object]] = []

    for row in tqdm(pair_rows, desc="Evaluating Cambridge pairs", unit="pair"):
        pair_idx = int(row["pair_idx"])
        rel_a = str(row["rel_a"]).strip()
        rel_b = str(row["rel_b"]).strip()
        quat_convention = str(row.get("quat_convention", "w2c")).strip() or "w2c"
        img_a = resolve_image_path(rel_a, cambridge_root, metadata_image_root)
        img_b = resolve_image_path(rel_b, cambridge_root, metadata_image_root)

        if not img_a.is_file() or not img_b.is_file():
            print(f"[skip] pair_idx={pair_idx} missing image")
            continue
        T_gt_a = resolve_gt_pose(rel_a, quat_convention, gt_pose_index, metadata_quat_index)
        T_gt_b = resolve_gt_pose(rel_b, quat_convention, gt_pose_index, metadata_quat_index)
        if T_gt_a is None or T_gt_b is None:
            print(f"[skip] pair_idx={pair_idx} missing GT pose index")
            continue

        try:
            T_pred_a, T_pred_b = run_dust3r_pair_pose(
                model=model,
                device=device,
                path_a=img_a,
                path_b=img_b,
                image_size=args.image_size,
            )
        except Exception as e:
            print(f"[skip] pair_idx={pair_idx} DUSt3R failed: {e}")
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
