#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "PoseCrafter Step 3: interpolate 4 relay-frame camera poses to a 25-pose trajectory "
            "(23 interpolated + 2 endpoints)."
        )
    )
    p.add_argument(
        "--relay_manifest",
        type=Path,
        default=Path("PoseCrafter/outputs/relay_frame_poses.jsonl"),
        help="Step-2 output JSONL from PoseCrafter/relay_frame_selection.py",
    )
    p.add_argument(
        "--num_output_poses",
        type=int,
        default=25,
        help="Number of output poses for the final trajectory (default: 25).",
    )
    p.add_argument(
        "--out_manifest",
        type=Path,
        default=None,
        help="Output JSONL path (default: <relay_manifest_dir>/interpolated_camera_poses.jsonl).",
    )
    return p.parse_args()


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    # Returns [w, x, y, z]
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize_quat(np.array([w, x, y, z], dtype=np.float64))


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0n = normalize_quat(q0)
    q1n = normalize_quat(q1)
    dot = float(np.dot(q0n, q1n))
    if dot < 0.0:
        q1n = -q1n
        dot = -dot
    if dot > 0.9995:
        return normalize_quat((1.0 - u) * q0n + u * q1n)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_0 = np.sin(theta_0)
    theta = theta_0 * u
    s0 = np.sin(theta_0 - theta) / sin_0
    s1 = np.sin(theta) / sin_0
    return normalize_quat(s0 * q0n + s1 * q1n)


def interp_pose_c2w(T0: np.ndarray, T1: np.ndarray, u: float) -> np.ndarray:
    R0, t0 = T0[:3, :3], T0[:3, 3]
    R1, t1 = T1[:3, :3], T1[:3, 3]
    q0 = rotmat_to_quat(R0)
    q1 = rotmat_to_quat(R1)
    q = slerp(q0, q1, u)
    R = quat_to_rotmat(q)
    t = (1.0 - u) * t0 + u * t1
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def interpolate_from_anchors(
    anchor_c2w: list[np.ndarray],
    anchor_frame_indices: list[int],
    total_video_frames: int,
    num_output_poses: int,
) -> list[np.ndarray]:
    if len(anchor_c2w) != 4 or len(anchor_frame_indices) != 4:
        raise ValueError("Need exactly 4 anchors for PoseCrafter step3.")
    if total_video_frames < 2:
        raise ValueError("total_video_frames must be >= 2.")
    if num_output_poses < 2:
        raise ValueError("num_output_poses must be >= 2.")

    # Anchor times from original 16-frame timeline, normalized to [0, 1].
    denom = float(total_video_frames - 1)
    anchor_t = [float(i) / denom for i in anchor_frame_indices]
    if sorted(anchor_t) != anchor_t:
        raise ValueError("Anchor frame indices must be non-decreasing.")

    # Output times for 25-frame trajectory.
    out_t = np.linspace(0.0, 1.0, num_output_poses, dtype=np.float64)
    out_poses: list[np.ndarray] = []

    for t in out_t:
        # find segment k so anchor_t[k] <= t <= anchor_t[k+1]
        if t <= anchor_t[0]:
            out_poses.append(anchor_c2w[0].copy())
            continue
        if t >= anchor_t[-1]:
            out_poses.append(anchor_c2w[-1].copy())
            continue
        k = 0
        for i in range(3):
            if anchor_t[i] <= t <= anchor_t[i + 1]:
                k = i
                break
        t0, t1 = anchor_t[k], anchor_t[k + 1]
        if abs(t1 - t0) < 1e-12:
            u = 0.0
        else:
            u = float((t - t0) / (t1 - t0))
        out_poses.append(interp_pose_c2w(anchor_c2w[k], anchor_c2w[k + 1], u))

    return out_poses


def main() -> None:
    args = parse_args()
    relay_manifest = args.relay_manifest.resolve()
    out_manifest = (
        args.out_manifest.resolve()
        if args.out_manifest is not None
        else relay_manifest.parent / "interpolated_camera_poses.jsonl"
    )

    rows = [json.loads(l) for l in relay_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"No rows in relay manifest: {relay_manifest}")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as f:
        for rec in rows:
            if rec.get("status") != "ok":
                continue
            anchor_c2w = [np.array(T, dtype=np.float64) for T in rec["c2w_poses"]]
            anchor_idx = [int(x) for x in rec["selected_frame_indices"]]
            total_frames = int(rec["total_video_frames"])
            poses_25 = interpolate_from_anchors(
                anchor_c2w=anchor_c2w,
                anchor_frame_indices=anchor_idx,
                total_video_frames=total_frames,
                num_output_poses=args.num_output_poses,
            )
            out = {
                "status": "ok",
                "pair_dir": rec["pair_dir"],
                "variant": rec["variant"],
                "video_path": rec["video_path"],
                "anchor_frame_indices": anchor_idx,
                "anchor_c2w_poses": rec["c2w_poses"],
                "num_output_poses": args.num_output_poses,
                "trajectory_c2w_poses": [p.tolist() for p in poses_25],
                "interpolated_only_c2w_poses": [p.tolist() for p in poses_25[1:-1]],
            }
            f.write(json.dumps(out) + "\n")

    print(f"Wrote interpolated trajectories: {out_manifest}")


if __name__ == "__main__":
    main()
