#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.Cambridge.eval_dust3r_cambridge import (  # noqa: E402
    MetadataQuatFrame,
    acc_at_deg,
    auc30,
    auc30_paper,
    build_cambridge_gt_pose_index,
    build_metadata_quat_index,
    geodesic_rotation_error_deg,
    maybe_float,
    relative_pose_T,
    resolve_gt_pose,
    translation_direction_angle_error_deg,
)


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pair_idx_from_pair_dir(pair_dir: str) -> int:
    name = Path(pair_dir).name
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected pair dir name: {name}")
    return int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate InterPose selected relative poses on Cambridge pairs."
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        default=Path("outputs/Cambridge/cambridge_selp_official_pairs.csv"),
        help="Pair CSV used by InterPose run.",
    )
    parser.add_argument(
        "--selected_jsonl",
        type=Path,
        default=Path(
            "Interpose/out_cambridge_selp_yaw50_65_286pairs/selection/selected_per_pair.jsonl"
        ),
        help="InterPose selected_per_pair.jsonl file.",
    )
    parser.add_argument(
        "--metadata_npy",
        type=Path,
        default=Path("metadata/metadata/selp_test_set.npy"),
        help="Optional sELP metadata .npy for GT quaternion fallback.",
    )
    parser.add_argument(
        "--cambridge_root",
        type=Path,
        default=Path("datasets/cambridge"),
        help="Cambridge dataset root.",
    )
    parser.add_argument(
        "--save_metrics_csv",
        type=Path,
        default=Path("outputs/Cambridge/interpose_selected.csv"),
        help="Per-pair metrics CSV output path.",
    )
    parser.add_argument(
        "--save_pose_jsonl",
        type=Path,
        default=Path("outputs/Cambridge/interpose_selected.jsonl"),
        help="Per-pair pose JSONL output path.",
    )
    parser.add_argument(
        "--save_summary_json",
        type=Path,
        default=Path("outputs/Cambridge/interpose_summary.json"),
        help="Aggregate metrics summary JSON output path.",
    )
    args = parser.parse_args()

    pairs_csv = args.pairs_csv.resolve()
    selected_jsonl = args.selected_jsonl.resolve()
    metadata_npy = args.metadata_npy.resolve()
    cambridge_root = args.cambridge_root.resolve()
    save_metrics_csv = args.save_metrics_csv.resolve()
    save_pose_jsonl = args.save_pose_jsonl.resolve()
    save_summary_json = args.save_summary_json.resolve()

    if not pairs_csv.is_file():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")
    if not selected_jsonl.is_file():
        raise FileNotFoundError(f"Selected JSONL not found: {selected_jsonl}")
    if not cambridge_root.is_dir():
        raise FileNotFoundError(f"Cambridge root not found: {cambridge_root}")

    pair_rows = list(csv.DictReader(pairs_csv.open("r", encoding="utf-8", newline="")))
    if not pair_rows:
        raise RuntimeError(f"No rows in pairs CSV: {pairs_csv}")
    pair_by_idx: Dict[int, Dict[str, str]] = {}
    for i, row in enumerate(pair_rows):
        raw_idx = str(row.get("pair_idx", "")).strip()
        idx = int(raw_idx) if raw_idx else i
        pair_by_idx[idx] = row

    selected_rows = _read_jsonl(selected_jsonl)
    if not selected_rows:
        raise RuntimeError(f"No rows in selected JSONL: {selected_jsonl}")

    # Guard against evaluating with a mismatched CSV (e.g. full official list vs
    # the specific 286-pair list used to generate selected_per_pair.jsonl).
    mismatch_count = 0
    checked = 0
    for rec in selected_rows[: min(50, len(selected_rows))]:
        pair_idx = _pair_idx_from_pair_dir(str(rec["pair_dir"]))
        row = pair_by_idx.get(pair_idx)
        if row is None:
            continue
        pair_scene = Path(str(rec["pair_dir"])).name.split("_", 2)[-1]
        row_scene = str(row.get("scene_name", "")).strip()
        if pair_scene and row_scene and pair_scene != row_scene:
            mismatch_count += 1
        checked += 1
    if checked > 0 and mismatch_count > checked * 0.6:
        raise RuntimeError(
            "Selected JSONL appears mismatched with pairs CSV (scene names do not align for most "
            "pair_idx rows). Use the same CSV that was used to run InterPose, e.g. "
            "outputs/Cambridge/cambridge_pairs_selp_metadata_only_strict_yaw50_65.csv."
        )

    scenes_from_csv = sorted(
        {
            str(row.get("scene_name", "")).strip()
            for row in pair_rows
            if str(row.get("scene_name", "")).strip()
        }
    )
    gt_pose_index = build_cambridge_gt_pose_index(
        cambridge_root=cambridge_root,
        scene_names=scenes_from_csv if scenes_from_csv else None,
    )
    metadata_quat_index: Dict[str, MetadataQuatFrame] = {}
    if metadata_npy.is_file():
        metadata_quat_index = build_metadata_quat_index(
            metadata_npy=metadata_npy,
            scene_names=scenes_from_csv if scenes_from_csv else None,
        )

    save_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    save_pose_jsonl.parent.mkdir(parents=True, exist_ok=True)
    save_summary_json.parent.mkdir(parents=True, exist_ok=True)

    rot_errs: List[float] = []
    trans_errs: List[float] = []
    metric_rows: List[Dict[str, object]] = []
    pose_rows: List[Dict[str, object]] = []

    for rec in selected_rows:
        pair_idx = _pair_idx_from_pair_dir(str(rec["pair_dir"]))
        row = pair_by_idx.get(pair_idx)
        if row is None:
            print(f"[skip] pair_idx={pair_idx} not found in pairs CSV")
            continue

        rel_a = str(row["rel_a"]).strip()
        rel_b = str(row["rel_b"]).strip()
        quat_convention = str(row.get("quat_convention", "w2c")).strip() or "w2c"
        T_gt_a = resolve_gt_pose(rel_a, quat_convention, gt_pose_index, metadata_quat_index)
        T_gt_b = resolve_gt_pose(rel_b, quat_convention, gt_pose_index, metadata_quat_index)
        if T_gt_a is None or T_gt_b is None:
            print(f"[skip] pair_idx={pair_idx} missing GT pose")
            continue

        T_rel_pred = np.array(rec["T_rel_selected"], dtype=np.float64)
        T_rel_gt = relative_pose_T(T_gt_a, T_gt_b)
        re = geodesic_rotation_error_deg(T_rel_pred[:3, :3], T_rel_gt[:3, :3])
        te = translation_direction_angle_error_deg(T_rel_pred[:3, 3], T_rel_gt[:3, 3])
        rot_errs.append(re)
        trans_errs.append(te)

        T_wc_pred_b = np.eye(4, dtype=np.float64)
        T_wc_pred_a = np.linalg.inv(T_rel_pred)
        metric_rows.append(
            {
                "pair_idx": pair_idx,
                "scene_name": row.get("scene_name", ""),
                "sequence_name": row.get("sequence_name", ""),
                "split_a": row.get("split_a", ""),
                "split_b": row.get("split_b", ""),
                "rel_a": rel_a,
                "rel_b": rel_b,
                "selected_variant": rec.get("selected_variant", ""),
                "yaw_diff_deg": maybe_float(row, "yaw_diff_deg"),
                "pitch_diff_deg": maybe_float(row, "pitch_diff_deg"),
                "geodesic_rot_deg": maybe_float(row, "geodesic_rot_deg"),
                "baseline": maybe_float(row, "baseline"),
                "D_total": rec.get("D_total", float("nan")),
                "D_med": rec.get("D_med", float("nan")),
                "D_bias": rec.get("D_bias", float("nan")),
                "rot_err_deg": re,
                "trans_dir_err_deg": te,
            }
        )
        pose_rows.append(
            {
                "pair_idx": pair_idx,
                "rel_a": rel_a,
                "rel_b": rel_b,
                "selected_variant": rec.get("selected_variant", ""),
                "pose_direction": rec.get("pose_direction", ""),
                "T_wc_gt_a": T_gt_a.tolist(),
                "T_wc_gt_b": T_gt_b.tolist(),
                "T_wc_pred_a": T_wc_pred_a.tolist(),
                "T_wc_pred_b": T_wc_pred_b.tolist(),
                "T_rel_gt_ba": T_rel_gt.tolist(),
                "T_rel_pred_ba": T_rel_pred.tolist(),
            }
        )

    if not metric_rows:
        raise RuntimeError("No valid pairs evaluated from selected JSONL.")

    metric_rows.sort(key=lambda r: int(r["pair_idx"]))
    pose_rows.sort(key=lambda r: int(r["pair_idx"]))

    with save_metrics_csv.open("w", encoding="utf-8", newline="") as f:
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
                "selected_variant",
                "yaw_diff_deg",
                "pitch_diff_deg",
                "geodesic_rot_deg",
                "baseline",
                "D_total",
                "D_med",
                "D_bias",
                "rot_err_deg",
                "trans_dir_err_deg",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    with save_pose_jsonl.open("w", encoding="utf-8") as f:
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
    save_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved per-pair metrics: {save_metrics_csv}")
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
