#!/usr/bin/env python3
"""
Build Cambridge Landmarks image pairs with large yaw change.

The Cambridge `dataset_train.txt` / `dataset_test.txt` files use rows like:

    ImageFile, Camera Position [X Y Z W P Q R]

where:
  - `ImageFile` is the path relative to the scene folder, e.g. `seq1/frame00028.png`
  - `X Y Z` is the camera center in world coordinates
  - `W P Q R` is the pose quaternion in WXYZ order

For pair construction we treat the quaternion as a world-to-camera rotation. This
matches the orientation math already used elsewhere in this repository and gives a
stable per-sequence "up" direction for Cambridge.

Default behavior:
  - load all Cambridge Landmarks scenes
  - keep only pairs from the same sequence
  - keep only pairs whose yaw difference is in [50, 65] degrees
  - record yaw difference, pitch difference, geodesic rotation angle, and baseline
  - sample 300 pairs in a random round-robin way across sequences
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CambridgeFrame:
    scene_name: str
    split_name: str
    rel_image_path: str
    sequence_name: str
    frame_idx: int
    position_xyz: Tuple[float, float, float]
    quat_wxyz: Tuple[float, float, float, float]


@dataclass(frozen=True)
class PairCandidate:
    frame_a: CambridgeFrame
    frame_b: CambridgeFrame
    yaw_diff_deg: float
    pitch_diff_deg: float
    geodesic_rot_deg: float
    baseline: float


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def vec_sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_norm(v: Sequence[float]) -> float:
    return math.sqrt(dot(v, v))


def normalize(v: Sequence[float], fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = vec_norm(v)
    if n < 1e-12:
        return fallback
    return (v[0] / n, v[1] / n, v[2] / n)


def clip_unit(x: float) -> float:
    return max(-1.0, min(1.0, x))


def mat_transpose(m: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


def mat_vec(m: Sequence[Sequence[float]], v: Sequence[float]) -> Tuple[float, float, float]:
    return (
        sum(m[0][k] * v[k] for k in range(3)),
        sum(m[1][k] * v[k] for k in range(3)),
        sum(m[2][k] * v[k] for k in range(3)),
    )


def mat_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(3))
    return out


def quat_wxyz_to_rotmat(q: Sequence[float]) -> List[List[float]]:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]


def camera_forward_world(quat_wxyz: Sequence[float]) -> Tuple[float, float, float]:
    """
    Optical axis (+Z_cam) in world coordinates.
    Cambridge quaternions are treated as world->camera rotations.
    """
    r_wc = quat_wxyz_to_rotmat(quat_wxyz)
    forward = mat_vec(mat_transpose(r_wc), (0.0, 0.0, 1.0))
    return normalize(forward, (0.0, 0.0, 1.0))


def camera_up_world(quat_wxyz: Sequence[float]) -> Tuple[float, float, float]:
    """
    Camera up vector in world coordinates.
    In the OpenCV camera convention, +Y points down, so we negate it.
    """
    r_wc = quat_wxyz_to_rotmat(quat_wxyz)
    down = mat_vec(mat_transpose(r_wc), (0.0, 1.0, 0.0))
    up = (-down[0], -down[1], -down[2])
    return normalize(up, (0.0, 1.0, 0.0))


def yaw_diff_deg_wrt_up(
    forward_a: Sequence[float],
    forward_b: Sequence[float],
    up_ref: Sequence[float],
) -> float:
    proj_a = (
        forward_a[0] - dot(forward_a, up_ref) * up_ref[0],
        forward_a[1] - dot(forward_a, up_ref) * up_ref[1],
        forward_a[2] - dot(forward_a, up_ref) * up_ref[2],
    )
    proj_b = (
        forward_b[0] - dot(forward_b, up_ref) * up_ref[0],
        forward_b[1] - dot(forward_b, up_ref) * up_ref[1],
        forward_b[2] - dot(forward_b, up_ref) * up_ref[2],
    )
    proj_a = normalize(proj_a, (1.0, 0.0, 0.0))
    proj_b = normalize(proj_b, (1.0, 0.0, 0.0))
    return math.degrees(math.acos(clip_unit(dot(proj_a, proj_b))))


def pitch_deg_wrt_up(forward: Sequence[float], up_ref: Sequence[float]) -> float:
    return math.degrees(math.asin(clip_unit(dot(forward, up_ref))))


def geodesic_rotation_deg(quat_a: Sequence[float], quat_b: Sequence[float]) -> float:
    r_a = quat_wxyz_to_rotmat(quat_a)
    r_b = quat_wxyz_to_rotmat(quat_b)
    r_rel = mat_mul(r_b, mat_transpose(r_a))
    trace = r_rel[0][0] + r_rel[1][1] + r_rel[2][2]
    cos_theta = clip_unit((trace - 1.0) * 0.5)
    return math.degrees(math.acos(cos_theta))


def parse_frame_index(rel_image_path: str) -> int:
    match = re.search(r"frame(\d+)", Path(rel_image_path).stem)
    return int(match.group(1)) if match else -1


def load_cambridge_frames(
    cambridge_root: Path,
    scene_names: Optional[Iterable[str]] = None,
) -> Dict[Tuple[str, str], List[CambridgeFrame]]:
    """
    Returns frames grouped by (scene_name, sequence_name).
    """
    wanted = set(scene_names) if scene_names else None
    frames_by_group: Dict[Tuple[str, str], List[CambridgeFrame]] = {}

    for scene_dir in sorted(p for p in cambridge_root.iterdir() if p.is_dir()):
        scene_name = scene_dir.name
        if wanted is not None and scene_name not in wanted:
            continue

        for split_file in ("dataset_train.txt", "dataset_test.txt"):
            pose_path = scene_dir / split_file
            if not pose_path.is_file():
                continue

            split_name = "train" if "train" in split_file else "test"
            lines = pose_path.read_text().splitlines()
            for line in lines[3:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue

                rel_image_path = parts[0]
                abs_image_path = scene_dir / rel_image_path
                if not abs_image_path.is_file():
                    continue

                position_xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
                quat_wxyz = (float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
                sequence_name = Path(rel_image_path).parts[0]
                frame = CambridgeFrame(
                    scene_name=scene_name,
                    split_name=split_name,
                    rel_image_path=f"{scene_name}/{rel_image_path}",
                    sequence_name=sequence_name,
                    frame_idx=parse_frame_index(rel_image_path),
                    position_xyz=position_xyz,
                    quat_wxyz=quat_wxyz,
                )
                frames_by_group.setdefault((scene_name, sequence_name), []).append(frame)

    for key in list(frames_by_group.keys()):
        frames_by_group[key] = sorted(
            frames_by_group[key],
            key=lambda fr: (fr.frame_idx, fr.rel_image_path),
        )
    return frames_by_group


def filter_frames_by_available_images(
    frames_by_group: Dict[Tuple[str, str], List[CambridgeFrame]],
    available_images_root: Path,
) -> Dict[Tuple[str, str], List[CambridgeFrame]]:
    filtered: Dict[Tuple[str, str], List[CambridgeFrame]] = {}
    for group, frames in frames_by_group.items():
        kept = [
            frame
            for frame in frames
            if (available_images_root / frame.rel_image_path).is_file()
        ]
        if kept:
            filtered[group] = kept
    return filtered


def build_candidates_for_sequence(
    frames: Sequence[CambridgeFrame],
    yaw_min: float,
    yaw_max: float,
    max_pitch_diff: Optional[float],
    max_baseline: Optional[float],
) -> List[PairCandidate]:
    forwards = [camera_forward_world(fr.quat_wxyz) for fr in frames]
    ups = [camera_up_world(fr.quat_wxyz) for fr in frames]
    up_ref = normalize(
        (
            sum(v[0] for v in ups),
            sum(v[1] for v in ups),
            sum(v[2] for v in ups),
        ),
        (0.0, 1.0, 0.0),
    )

    out: List[PairCandidate] = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            yaw_diff = yaw_diff_deg_wrt_up(forwards[i], forwards[j], up_ref)
            if yaw_diff < yaw_min or yaw_diff > yaw_max:
                continue

            pitch_i = pitch_deg_wrt_up(forwards[i], up_ref)
            pitch_j = pitch_deg_wrt_up(forwards[j], up_ref)
            pitch_diff = abs(pitch_i - pitch_j)
            if max_pitch_diff is not None and pitch_diff > max_pitch_diff:
                continue

            baseline = vec_norm(vec_sub(frames[i].position_xyz, frames[j].position_xyz))
            if max_baseline is not None and baseline > max_baseline:
                continue

            out.append(
                PairCandidate(
                    frame_a=frames[i],
                    frame_b=frames[j],
                    yaw_diff_deg=yaw_diff,
                    pitch_diff_deg=pitch_diff,
                    geodesic_rot_deg=geodesic_rotation_deg(frames[i].quat_wxyz, frames[j].quat_wxyz),
                    baseline=baseline,
                )
            )

    return out


def round_robin_sample(
    grouped_candidates: Dict[Tuple[str, str], List[PairCandidate]],
    max_pairs: int,
    seed: int,
) -> List[PairCandidate]:
    rng = random.Random(seed)
    shuffled_candidates: Dict[Tuple[str, str], List[PairCandidate]] = {}
    groups = []
    for group, pairs in grouped_candidates.items():
        if not pairs:
            continue
        pairs_copy = list(pairs)
        rng.shuffle(pairs_copy)
        shuffled_candidates[group] = pairs_copy
        groups.append(group)
    rng.shuffle(groups)

    chosen: List[PairCandidate] = []
    group_ptrs = {group: 0 for group in groups}
    while len(chosen) < max_pairs:
        progressed = False
        for group in groups:
            ptr = group_ptrs[group]
            pairs = shuffled_candidates[group]
            if ptr >= len(pairs):
                continue
            chosen.append(pairs[ptr])
            group_ptrs[group] = ptr + 1
            progressed = True
            if len(chosen) >= max_pairs:
                break
        if not progressed:
            break
    return chosen


def save_pairs_csv(pairs: Sequence[PairCandidate], out_csv: Path) -> None:
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
                "frame_idx_a",
                "frame_idx_b",
            ],
        )
        writer.writeheader()
        for pair_idx, cand in enumerate(pairs):
            writer.writerow(
                {
                    "pair_idx": pair_idx,
                    "scene_name": cand.frame_a.scene_name,
                    "sequence_name": cand.frame_a.sequence_name,
                    "split_a": cand.frame_a.split_name,
                    "split_b": cand.frame_b.split_name,
                    "rel_a": cand.frame_a.rel_image_path,
                    "rel_b": cand.frame_b.rel_image_path,
                    "yaw_diff_deg": f"{cand.yaw_diff_deg:.6f}",
                    "pitch_diff_deg": f"{cand.pitch_diff_deg:.6f}",
                    "geodesic_rot_deg": f"{cand.geodesic_rot_deg:.6f}",
                    "baseline": f"{cand.baseline:.6f}",
                    "frame_idx_a": cand.frame_a.frame_idx,
                    "frame_idx_b": cand.frame_b.frame_idx,
                }
            )


def parse_scene_list(scene_csv: str) -> Optional[List[str]]:
    if not scene_csv.strip():
        return None
    return [x.strip() for x in scene_csv.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build same-sequence Cambridge pairs in a target yaw range."
    )
    parser.add_argument(
        "--cambridge_root",
        type=Path,
        default=Path("datasets/cambridge"),
        help="Root directory that contains the Cambridge scene folders.",
    )
    parser.add_argument("--yaw_min", type=float, default=50.0, help="Minimum yaw difference in degrees.")
    parser.add_argument("--yaw_max", type=float, default=65.0, help="Maximum yaw difference in degrees.")
    parser.add_argument("--max_pairs", type=int, default=300, help="Maximum number of pairs to save.")
    parser.add_argument(
        "--max_pitch_diff",
        type=float,
        default=None,
        help="Optional maximum pitch difference in degrees.",
    )
    parser.add_argument(
        "--max_baseline",
        type=float,
        default=None,
        help="Optional maximum camera-center distance.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Optional comma-separated scene subset, e.g. KingsCollege,GreatCourt",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for group order shuffling.")
    parser.add_argument(
        "--available_images_root",
        type=Path,
        default=None,
        help=(
            "Optional image root used to restrict selection to frames whose image files "
            "exist under a second mirrored directory tree."
        ),
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("outputs/Cambridge/cambridge_pairs_yaw50_65_300.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    cambridge_root = args.cambridge_root.resolve()
    if not cambridge_root.is_dir():
        raise FileNotFoundError(f"Cambridge root not found: {cambridge_root}")

    if args.yaw_min > args.yaw_max:
        raise ValueError("--yaw_min must be <= --yaw_max")

    frames_by_group = load_cambridge_frames(
        cambridge_root=cambridge_root,
        scene_names=parse_scene_list(args.scenes),
    )
    if args.available_images_root is not None:
        available_images_root = args.available_images_root.resolve()
        if not available_images_root.is_dir():
            raise FileNotFoundError(
                f"Available images root not found: {available_images_root}"
            )
        frames_by_group = filter_frames_by_available_images(
            frames_by_group=frames_by_group,
            available_images_root=available_images_root,
        )
    if not frames_by_group:
        raise RuntimeError("No Cambridge frames loaded. Check --cambridge_root and --scenes.")

    grouped_candidates: Dict[Tuple[str, str], List[PairCandidate]] = {}
    total_frames = 0
    for group, frames in sorted(frames_by_group.items()):
        total_frames += len(frames)
        grouped_candidates[group] = build_candidates_for_sequence(
            frames=frames,
            yaw_min=args.yaw_min,
            yaw_max=args.yaw_max,
            max_pitch_diff=args.max_pitch_diff,
            max_baseline=args.max_baseline,
        )

    selected = round_robin_sample(
        grouped_candidates=grouped_candidates,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No valid Cambridge pairs found with the requested filters.")

    save_pairs_csv(selected, args.out_csv)

    valid_groups = sum(1 for pairs in grouped_candidates.values() if pairs)
    print(f"Loaded frames: {total_frames}")
    print(f"Valid sequences with candidate pairs: {valid_groups}")
    print(f"Saved pairs: {len(selected)}")
    print(f"CSV: {args.out_csv.resolve()}")
    print(
        "Yaw filter: "
        f"[{args.yaw_min:.2f}, {args.yaw_max:.2f}] deg"
        + (
            ""
            if args.max_pitch_diff is None
            else f" | max pitch diff: {args.max_pitch_diff:.2f} deg"
        )
        + (
            ""
            if args.max_baseline is None
            else f" | max baseline: {args.max_baseline:.4f}"
        )
    )
    if args.available_images_root is not None:
        print(f"Available images root: {args.available_images_root.resolve()}")
    if len(selected) < args.max_pairs:
        print(f"[warn] requested {args.max_pairs} pairs, but only {len(selected)} were available.")


if __name__ == "__main__":
    main()
