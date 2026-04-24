#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def q_to_r(q: np.ndarray) -> np.ndarray:
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


def geodesic_deg(r: np.ndarray) -> float:
    x = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(x))


def normalize_rel_path(raw: str) -> str:
    return Path(raw.replace("\\", "/").lstrip("./")).as_posix()


def build_quat_index(metadata_npy: Path) -> dict[str, np.ndarray]:
    obj = np.load(metadata_npy, allow_pickle=True).item()
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected metadata format in {metadata_npy}")

    quat_index: dict[str, np.ndarray] = {}
    for pair in obj.values():
        if not isinstance(pair, dict):
            continue
        for key in ("img1", "img2"):
            image_meta = pair.get(key, {})
            if not isinstance(image_meta, dict):
                continue
            rel = normalize_rel_path(str(image_meta.get("path", "")).strip())
            if not rel or rel in quat_index:
                continue
            quat_index[rel] = np.array(
                [
                    float(image_meta.get("qw", 1.0)),
                    float(image_meta.get("qx", 0.0)),
                    float(image_meta.get("qy", 0.0)),
                    float(image_meta.get("qz", 0.0)),
                ],
                dtype=np.float64,
            )
    return quat_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate endpoint relative rotation error from vggt3d extrinsic_rt.txt outputs."
    )
    p.add_argument(
        "--pairs_csv",
        type=Path,
        default=Path("outputs/Cambridge/cambridge_pairs_selp_metadata_only_strict_yaw50_65.csv"),
        help="Input pair CSV containing pair_idx/scene_name/rel_a/rel_b/yaw_diff_deg.",
    )
    p.add_argument(
        "--metadata_npy",
        type=Path,
        default=Path("metadata/metadata/selp_test_set.npy"),
        help="sELP metadata file containing quaternion GT.",
    )
    p.add_argument(
        "--pose_root",
        type=Path,
        required=True,
        help="Root containing per-pair folders with extrinsic_rt.txt, e.g. pair_00000_Street/extrinsic_rt.txt.",
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        default=Path("outputs/Cambridge/vggt3d_286_rot_metrics.csv"),
        help="Output CSV path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    quat_index = build_quat_index(args.metadata_npy.resolve())

    out_rows = []
    with args.pairs_csv.resolve().open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_idx = int(row["pair_idx"])
            scene_name = str(row["scene_name"]).strip()
            rel_a = str(row["rel_a"]).strip()
            rel_b = str(row["rel_b"]).strip()
            yaw_diff = float(row["yaw_diff_deg"])

            pair_dir = args.pose_root.resolve() / f"pair_{pair_idx:05d}_{scene_name}"
            rt_path = pair_dir / "extrinsic_rt.txt"

            rot_err_deg = float("nan")
            gt_rel_rot_deg = float("nan")
            pred_rel_rot_deg = float("nan")
            status = "ok"

            if not rt_path.is_file():
                status = "missing_pred"
            elif rel_a not in quat_index or rel_b not in quat_index:
                status = "missing_gt_quat"
            else:
                pred = np.loadtxt(rt_path)
                if pred.ndim == 1:
                    pred = pred.reshape(1, -1)
                if pred.shape[0] < 2 or pred.shape[1] != 12:
                    status = "bad_pred_shape"
                else:
                    r_pred_a = pred[0].reshape(3, 4)[:, :3]
                    r_pred_b = pred[-1].reshape(3, 4)[:, :3]
                    r_pred_rel = r_pred_b @ r_pred_a.T

                    r_gt_a = q_to_r(quat_index[rel_a])
                    r_gt_b = q_to_r(quat_index[rel_b])
                    r_gt_rel = r_gt_b @ r_gt_a.T

                    gt_rel_rot_deg = geodesic_deg(r_gt_rel)
                    pred_rel_rot_deg = geodesic_deg(r_pred_rel)
                    rot_err_deg = geodesic_deg(r_pred_rel @ r_gt_rel.T)

            out_rows.append(
                {
                    "pair_idx": pair_idx,
                    "scene_name": scene_name,
                    "rel_a": rel_a,
                    "rel_b": rel_b,
                    "yaw_diff_deg": yaw_diff,
                    "rot_err_deg": rot_err_deg,
                    "gt_rel_rot_deg": gt_rel_rot_deg,
                    "pred_rel_rot_deg": pred_rel_rot_deg,
                    "status": status,
                }
            )

    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_idx",
                "scene_name",
                "rel_a",
                "rel_b",
                "yaw_diff_deg",
                "rot_err_deg",
                "gt_rel_rot_deg",
                "pred_rel_rot_deg",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    valid = [r["rot_err_deg"] for r in out_rows if not math.isnan(r["rot_err_deg"])]
    mean_err = float(np.mean(valid)) if valid else float("nan")
    print(f"[done] wrote {out_csv}")
    print(f"[info] valid_pairs={len(valid)}/{len(out_rows)} mean_rot_err_deg={mean_err:.4f}")


if __name__ == "__main__":
    main()

