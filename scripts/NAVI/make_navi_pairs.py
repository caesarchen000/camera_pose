#!/usr/bin/env python3
"""
Build NAVI evaluation pairs with yaw range filtering.

Default protocol:
  - randomly sample pairs from 36 objects
  - total 300 pairs
  - yaw difference in [50, 90] degrees
  - scenes from multiview/video only
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class NaviFrame:
    object_id: str
    scene_name: str
    rel_image_path: str  # rel to navi_root, e.g. object/scene/images/000.jpg
    q: np.ndarray  # [qw, qx, qy, qz]
    t: np.ndarray  # [tx, ty, tz]
    split: str
    camera_model: str


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert [qw, qx, qy, qz] to 3x3 rotation matrix."""
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


def camera_forward_world(q: np.ndarray) -> np.ndarray:
    """
    Optical axis (+Z_cam) in world coordinates.
    Assumes q represents world->camera rotation.
    """
    R_wc = quaternion_to_rotation_matrix(q)
    f = R_wc.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = np.linalg.norm(f)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return f / n


def camera_up_world(q: np.ndarray) -> np.ndarray:
    """Camera up vector in world coordinates (+Y down in camera -> negate to get up)."""
    R_wc = quaternion_to_rotation_matrix(q)
    v_down = R_wc.T @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v_up = -v_down
    n = np.linalg.norm(v_up)
    if n < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return v_up / n


def yaw_diff_deg_wrt_up(f0: np.ndarray, f1: np.ndarray, up_ref: np.ndarray) -> float:
    """Yaw diff as angle between forward vectors projected to plane orthogonal to up_ref."""
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


def load_navi_frames(
    navi_root: Path,
    include_scene_prefixes: Tuple[str, ...],
    split: str,
) -> Dict[str, List[NaviFrame]]:
    """
    Returns frames grouped by object_id.
    """
    by_object: Dict[str, List[NaviFrame]] = {}
    object_dirs = sorted([p for p in navi_root.iterdir() if p.is_dir()])
    for obj_dir in object_dirs:
        object_id = obj_dir.name
        obj_frames: List[NaviFrame] = []
        for scene_dir in sorted([p for p in obj_dir.iterdir() if p.is_dir()]):
            scene_name = scene_dir.name
            if not scene_name.startswith(include_scene_prefixes):
                continue
            ann_path = scene_dir / "annotations.json"
            if not ann_path.is_file():
                continue
            data = json.loads(ann_path.read_text())
            for rec in data:
                rec_split = rec.get("split", "")
                if split != "all" and rec_split != split:
                    continue
                filename = rec.get("filename")
                if not filename:
                    continue
                image_rel = f"{object_id}/{scene_name}/images/{filename}"
                image_abs = navi_root / image_rel
                if not image_abs.is_file():
                    continue
                cam = rec.get("camera", {})
                q = np.array(cam.get("q", []), dtype=np.float64)
                t = np.array(cam.get("t", []), dtype=np.float64)
                if q.shape != (4,) or t.shape != (3,):
                    continue
                obj_frames.append(
                    NaviFrame(
                        object_id=object_id,
                        scene_name=scene_name,
                        rel_image_path=image_rel,
                        q=q,
                        t=t,
                        split=rec_split,
                        camera_model=str(cam.get("camera_model", "")),
                    )
                )
        if obj_frames:
            by_object[object_id] = obj_frames
    return by_object


def build_valid_pairs(
    frames: List[NaviFrame],
    yaw_min: float,
    yaw_max: float,
    same_scene_only: bool = True,
) -> List[Tuple[int, int, float]]:
    """
    Build valid pairs within one object.
    Returns tuples: (i, j, yaw_diff_deg), using indices into `frames`.
    """
    forwards = [camera_forward_world(f.q) for f in frames]
    ups = [camera_up_world(f.q) for f in frames]
    up_ref = np.sum(np.stack(ups, axis=0), axis=0)
    up_ref = up_ref / (np.linalg.norm(up_ref) + 1e-12)

    pairs: List[Tuple[int, int, float]] = []
    n = len(frames)
    for i in range(n):
        for j in range(i + 1, n):
            if same_scene_only and frames[i].scene_name != frames[j].scene_name:
                continue
            yaw = yaw_diff_deg_wrt_up(forwards[i], forwards[j], up_ref)
            if yaw_min <= yaw <= yaw_max:
                pairs.append((i, j, yaw))
    return pairs


def sample_pairs_balanced(
    by_object_pairs: Dict[str, List[Tuple[int, int, float]]],
    max_pairs: int,
    num_objects: int,
    seed: int,
) -> List[Tuple[str, int, int, float]]:
    """
    Return sampled tuples: (object_id, i, j, yaw_diff_deg).
    """
    rng = np.random.default_rng(seed)
    object_ids = [o for o, ps in by_object_pairs.items() if ps]
    if len(object_ids) < num_objects:
        print(
            f"[warn] requested {num_objects} objects but only {len(object_ids)} "
            "have valid pairs; using all available."
        )
        chosen_objects = object_ids
    else:
        chosen_objects = list(rng.choice(object_ids, size=num_objects, replace=False))

    # Shuffle each object's candidate list for random sampling.
    shuffled: Dict[str, List[Tuple[int, int, float]]] = {}
    for obj in chosen_objects:
        ps = list(by_object_pairs[obj])
        rng.shuffle(ps)
        shuffled[obj] = ps

    # Round-robin to avoid domination by objects with huge candidate pools.
    sampled: List[Tuple[str, int, int, float]] = []
    obj_ptr: Dict[str, int] = {o: 0 for o in chosen_objects}
    while len(sampled) < max_pairs:
        progressed = False
        for obj in chosen_objects:
            p = obj_ptr[obj]
            if p >= len(shuffled[obj]):
                continue
            i, j, yaw = shuffled[obj][p]
            sampled.append((obj, i, j, yaw))
            obj_ptr[obj] += 1
            progressed = True
            if len(sampled) >= max_pairs:
                break
        if not progressed:
            break
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Make NAVI pair CSV by yaw range.")
    parser.add_argument(
        "--navi_root",
        type=Path,
        default=Path("datasets/NAVI/navi_v1.5"),
        help="Root directory of extracted NAVI v1.5 data.",
    )
    parser.add_argument("--yaw_min", type=float, default=50.0)
    parser.add_argument("--yaw_max", type=float, default=90.0)
    parser.add_argument("--max_pairs", type=int, default=300)
    parser.add_argument("--num_objects", type=int, default=36)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "all"))
    parser.add_argument(
        "--include_video",
        action="store_true",
        help="Include video-* scenes in pair construction.",
    )
    parser.add_argument(
        "--include_wild",
        action="store_true",
        help="Include wild_set scenes in pair construction.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow_cross_scene_pairs",
        action="store_true",
        help="If set, allow pairing images from different scene folders of the same object.",
    )
    parser.add_argument(
        "--max_frames_per_object",
        type=int,
        default=0,
        help=(
            "Optional cap per object before pair construction (0 = no cap). "
            "Useful to keep pair generation fast when including video scenes."
        ),
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("outputs/navi_pairs_yaw50_90_300.csv"),
    )
    args = parser.parse_args()

    navi_root = args.navi_root.resolve()
    if not navi_root.is_dir():
        raise FileNotFoundError(f"NAVI root not found: {navi_root}")

    scene_prefixes: List[str] = ["multiview-"]
    if args.include_video:
        scene_prefixes.append("video-")
    if args.include_wild:
        scene_prefixes.append("wild_set")

    by_object_frames = load_navi_frames(
        navi_root=navi_root,
        include_scene_prefixes=tuple(scene_prefixes),
        split=args.split,
    )
    if not by_object_frames:
        raise RuntimeError("No NAVI frames loaded. Check --navi_root and scene/split filters.")

    if args.max_frames_per_object > 0:
        rng_cap = np.random.default_rng(args.seed)
        for obj in list(by_object_frames.keys()):
            fs = by_object_frames[obj]
            if len(fs) > args.max_frames_per_object:
                idx = rng_cap.choice(len(fs), size=args.max_frames_per_object, replace=False)
                by_object_frames[obj] = [fs[i] for i in sorted(idx.tolist())]

    by_object_pairs: Dict[str, List[Tuple[int, int, float]]] = {}
    for obj, frames in by_object_frames.items():
        by_object_pairs[obj] = build_valid_pairs(
            frames,
            args.yaw_min,
            args.yaw_max,
            same_scene_only=not args.allow_cross_scene_pairs,
        )

    selected = sample_pairs_balanced(
        by_object_pairs=by_object_pairs,
        max_pairs=args.max_pairs,
        num_objects=args.num_objects,
        seed=args.seed,
    )
    if len(selected) < args.max_pairs:
        print(f"[warn] sampled only {len(selected)} pairs (requested {args.max_pairs}).")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
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
                "split_a",
                "split_b",
                "camera_model_a",
                "camera_model_b",
            ],
        )
        writer.writeheader()
        for pair_idx, (obj, i, j, yaw) in enumerate(selected):
            fa = by_object_frames[obj][i]
            fb = by_object_frames[obj][j]
            writer.writerow(
                {
                    "pair_idx": pair_idx,
                    "object_id": obj,
                    "scene_a": fa.scene_name,
                    "scene_b": fb.scene_name,
                    "rel_a": fa.rel_image_path,
                    "rel_b": fb.rel_image_path,
                    "yaw_diff_deg": round(yaw, 6),
                    "split_a": fa.split,
                    "split_b": fb.split,
                    "camera_model_a": fa.camera_model,
                    "camera_model_b": fb.camera_model,
                }
            )

    used_objects = sorted({x[0] for x in selected})
    print(f"Saved pair CSV: {args.out_csv.resolve()}")
    print(f"Pairs: {len(selected)} | Objects used: {len(used_objects)}")
    print(f"Yaw range filter: [{args.yaw_min}, {args.yaw_max}]")
    print(f"Split: {args.split} | Scene prefixes: {scene_prefixes}")


if __name__ == "__main__":
    main()

