#!/usr/bin/env python3
"""
Extract the official Cambridge sELP pairs from ExtremeRotations metadata.

This script reads the official `metadata/selp_test_set.npy` file from
ExtremeRotationsInTheWild, filters the Cambridge Landmarks pairs, resolves them
to the local `datasets/cambridge` layout, and writes a separate CSV so it does
not overwrite any custom pair lists already created in this repo.

Default output:
  outputs/Cambridge/cambridge_selp_official_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CambridgeFrame:
    rel_path: str
    scene_name: str
    sequence_name: str
    split_name: str
    T_wc: np.ndarray
    camera_center_w: np.ndarray


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


def clip_unit(x: float) -> float:
    return max(-1.0, min(1.0, x))


def normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return fallback.copy()
    return v / n


def camera_forward_world(T_wc: np.ndarray) -> np.ndarray:
    forward = T_wc[:3, :3].T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normalize(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64))


def camera_up_world(T_wc: np.ndarray) -> np.ndarray:
    down = T_wc[:3, :3].T @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up = -down
    return normalize(up, np.array([0.0, 1.0, 0.0], dtype=np.float64))


def yaw_diff_deg_wrt_up(f0: np.ndarray, f1: np.ndarray, up_ref: np.ndarray) -> float:
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
    return float(np.degrees(np.arcsin(clip_unit(float(np.dot(f, up_ref))))))


def geodesic_rotation_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    R_rel = T_b[:3, :3] @ T_a[:3, :3].T
    c = float(np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def load_cambridge_gt_pose_index(cambridge_root: Path) -> Dict[str, CambridgeFrame]:
    pose_index: Dict[str, CambridgeFrame] = {}

    for scene_dir in sorted(p for p in cambridge_root.iterdir() if p.is_dir()):
        scene_name = scene_dir.name
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

                pose_index[rel_path] = CambridgeFrame(
                    rel_path=rel_path,
                    scene_name=scene_name,
                    sequence_name=Path(rel_img).parts[0],
                    split_name=split_name,
                    T_wc=T_wc,
                    camera_center_w=cam_center_w,
                )

    return pose_index


def iter_pairs_object(metadata_obj) -> Iterable[dict]:
    if isinstance(metadata_obj, dict):
        keys = list(metadata_obj.keys())
        if keys and all(isinstance(k, (int, np.integer)) for k in keys):
            for k in sorted(keys):
                yield metadata_obj[k]
        else:
            for _, v in metadata_obj.items():
                yield v
    elif isinstance(metadata_obj, (list, tuple, np.ndarray)):
        for item in metadata_obj:
            yield item
    else:
        raise TypeError(f"Unsupported metadata object type: {type(metadata_obj)!r}")


def resolve_local_cambridge_rel_path(
    raw_path: str,
    cambridge_root: Path,
    scene_names: Sequence[str],
) -> Optional[str]:
    raw_norm = raw_path.replace("\\", "/").lstrip("./")
    candidate = cambridge_root / raw_norm
    if candidate.is_file():
        return raw_norm

    parts = Path(raw_norm).parts
    for idx, part in enumerate(parts):
        if part in scene_names:
            sub = Path(*parts[idx:]).as_posix()
            if (cambridge_root / sub).is_file():
                return sub

    for scene_name in scene_names:
        marker = f"/{scene_name}/"
        pos = raw_norm.find(marker)
        if pos >= 0:
            sub = raw_norm[pos + 1 :]
            if (cambridge_root / sub).is_file():
                return sub
        if raw_norm.startswith(scene_name + "/"):
            if (cambridge_root / raw_norm).is_file():
                return raw_norm

    return None


def build_row(
    pair_idx: int,
    rel_a: str,
    rel_b: str,
    gt_index: Dict[str, CambridgeFrame],
    overlap_amount: str,
    metadata_scene: str,
) -> dict:
    fa = gt_index[rel_a]
    fb = gt_index[rel_b]
    up_ref = normalize(
        camera_up_world(fa.T_wc) + camera_up_world(fb.T_wc),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    fwd_a = camera_forward_world(fa.T_wc)
    fwd_b = camera_forward_world(fb.T_wc)
    yaw_diff = yaw_diff_deg_wrt_up(fwd_a, fwd_b, up_ref)
    pitch_a = pitch_deg_wrt_up(fwd_a, up_ref)
    pitch_b = pitch_deg_wrt_up(fwd_b, up_ref)
    pitch_diff = abs(pitch_a - pitch_b)
    geodesic_rot = geodesic_rotation_deg(fa.T_wc, fb.T_wc)
    baseline = float(np.linalg.norm(fa.camera_center_w - fb.camera_center_w))

    return {
        "pair_idx": pair_idx,
        "scene_name": fa.scene_name,
        "sequence_name": fa.sequence_name if fa.sequence_name == fb.sequence_name else "",
        "split_a": fa.split_name,
        "split_b": fb.split_name,
        "rel_a": rel_a,
        "rel_b": rel_b,
        "yaw_diff_deg": f"{yaw_diff:.6f}",
        "pitch_diff_deg": f"{pitch_diff:.6f}",
        "geodesic_rot_deg": f"{geodesic_rot:.6f}",
        "baseline": f"{baseline:.6f}",
        "overlap_amount": overlap_amount,
        "metadata_scene": metadata_scene,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract official Cambridge sELP pairs from ExtremeRotations metadata."
    )
    parser.add_argument(
        "--metadata_npy",
        type=Path,
        default=Path("metadata/selp_test_set.npy"),
        help="Official ExtremeRotations sELP pair metadata .npy file.",
    )
    parser.add_argument(
        "--cambridge_root",
        type=Path,
        default=Path("datasets/cambridge"),
        help="Local Cambridge dataset root.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("outputs/Cambridge/cambridge_selp_official_pairs.csv"),
        help="Output CSV path for extracted official Cambridge pairs.",
    )
    args = parser.parse_args()

    metadata_npy = args.metadata_npy.resolve()
    cambridge_root = args.cambridge_root.resolve()
    out_csv = args.out_csv.resolve()

    if not metadata_npy.is_file():
        raise FileNotFoundError(
            f"Official metadata file not found: {metadata_npy}. "
            "Download metadata.zip and extract metadata/selp_test_set.npy first."
        )
    if not cambridge_root.is_dir():
        raise FileNotFoundError(f"Cambridge root not found: {cambridge_root}")

    gt_index = load_cambridge_gt_pose_index(cambridge_root)
    if not gt_index:
        raise RuntimeError(f"No Cambridge GT poses found under {cambridge_root}")

    scene_names = sorted({fr.scene_name for fr in gt_index.values()})
    metadata_obj = np.load(metadata_npy, allow_pickle=True).item()

    rows: List[dict] = []
    skipped_not_cambridge = 0
    skipped_unresolved = 0

    for pair in iter_pairs_object(metadata_obj):
        img1 = pair.get("img1", {})
        img2 = pair.get("img2", {})
        raw_a = str(img1.get("path", "")).strip()
        raw_b = str(img2.get("path", "")).strip()
        metadata_scene = str(pair.get("scene", "")).strip()
        overlap_amount = str(pair.get("overlap_amount", "")).strip()

        rel_a = resolve_local_cambridge_rel_path(raw_a, cambridge_root, scene_names)
        rel_b = resolve_local_cambridge_rel_path(raw_b, cambridge_root, scene_names)

        if rel_a is None or rel_b is None:
            skipped_unresolved += 1
            continue
        if rel_a.split("/")[0] not in scene_names or rel_b.split("/")[0] not in scene_names:
            skipped_not_cambridge += 1
            continue
        if rel_a not in gt_index or rel_b not in gt_index:
            skipped_unresolved += 1
            continue

        rows.append(
            build_row(
                pair_idx=len(rows),
                rel_a=rel_a,
                rel_b=rel_b,
                gt_index=gt_index,
                overlap_amount=overlap_amount,
                metadata_scene=metadata_scene,
            )
        )

    if not rows:
        raise RuntimeError(
            "No Cambridge pairs were extracted from the official metadata. "
            "Check the metadata file layout and local Cambridge dataset paths."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
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
                "overlap_amount",
                "metadata_scene",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved extracted official Cambridge pairs: {out_csv}")
    print(f"Pairs: {len(rows)}")
    print(f"Skipped unresolved paths: {skipped_unresolved}")
    print(f"Skipped non-Cambridge pairs: {skipped_not_cambridge}")


if __name__ == "__main__":
    main()
