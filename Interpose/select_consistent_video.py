#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "InterPose medoid selection: compute D_med (and optional D_total with pair-only DUSt3R bias), "
            "then select the best video per pair."
        )
    )
    p.add_argument(
        "--subset_pose_manifest",
        type=Path,
        required=True,
        help="JSONL from Interpose/run_dust3r_subsets.py",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for scored video manifests.",
    )
    p.add_argument(
        "--use_bias",
        action="store_true",
        help="If set, compute D_total = D_med + D_bias using pair-only DUSt3R baseline.",
    )
    p.add_argument("--checkpoint", type=Path, default=None, help="DUSt3R checkpoint for pair-only baseline.")
    p.add_argument("--dust3r_root", type=Path, default=Path("dust3r"))
    p.add_argument("--image_size", type=int, default=512, choices=(224, 512))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def rot_dist_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    Re = R2 @ R1.T
    c = float(np.clip((np.trace(Re) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def trans_dir_dist_deg(t1: np.ndarray, t2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(t1))
    n2 = float(np.linalg.norm(t2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    d = float(np.dot(t1 / n1, t2 / n2))
    # Same as paper equation with absolute value for sign ambiguity.
    d = float(np.clip(abs(d), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def pose_dist(T1: np.ndarray, T2: np.ndarray) -> float:
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    return rot_dist_deg(R1, R2) + trans_dir_dist_deg(t1, t2)


def canonicalize_to_a2b(T_rel: np.ndarray, variant: str) -> np.ndarray:
    """Normalize all variant poses to canonical A->B direction."""
    # v*_B2A_* variants produce B->A relative poses by construction.
    # For consistent scoring/selection against GT(A->B), invert them.
    if "B2A" in variant:
        return np.linalg.inv(T_rel).astype(np.float64)
    return T_rel


def medoid_index_and_score(poses: list[np.ndarray]) -> tuple[int, float]:
    n = len(poses)
    if n == 1:
        return 0, 0.0
    dmat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = pose_dist(poses[i], poses[j])
            dmat[i, j] = d
            dmat[j, i] = d
    avg = np.sum(dmat, axis=1) / max(1, (n - 1))
    m_idx = int(np.argmin(avg))
    return m_idx, float(avg[m_idx])


def run_dust3r_pair_pose(
    model,
    device: torch.device,
    path_a: Path,
    path_b: Path,
    image_size: int,
    batch_size: int,
) -> np.ndarray:
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
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
        output = inference(pairs, model, device, batch_size=batch_size, verbose=False)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=False)
    c2w = scene.get_im_poses().detach().float().cpu().numpy()
    T_wc_A = np.linalg.inv(c2w[0]).astype(np.float64)
    T_wc_B = np.linalg.inv(c2w[1]).astype(np.float64)
    return T_wc_B @ np.linalg.inv(T_wc_A)


def main() -> None:
    args = parse_args()
    args.subset_pose_manifest = args.subset_pose_manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    args.dust3r_root = args.dust3r_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_bias:
        if args.checkpoint is None:
            raise ValueError("--use_bias requires --checkpoint")
        if str(args.dust3r_root) not in sys.path:
            sys.path.insert(0, str(args.dust3r_root))
        from dust3r.model import load_model  # pylint: disable=import-outside-toplevel
        device = torch.device(args.device)
        model = load_model(str(args.checkpoint.resolve()), device)
        model.eval()
    else:
        device = None
        model = None

    # Group by (pair_dir, variant). Each row already corresponds to one subset prediction.
    by_video: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with args.subset_pose_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            pair_dir = rec.get("pair_dir")
            variant = rec.get("variant")
            if pair_dir is None or variant is None:
                # Backward compatibility fallback from subset_dir path.
                sdir = Path(rec["subset_dir"])
                # .../pair_xxx/subset/<variant>/subset_xx
                if "subset" in sdir.parts:
                    i = sdir.parts.index("subset")
                    if i >= 1 and i + 1 < len(sdir.parts):
                        pair_dir = str(Path(*sdir.parts[:i]))
                        variant = sdir.parts[i + 1]
            if pair_dir is None or variant is None:
                continue
            by_video[(pair_dir, variant)].append(rec)

    # Score each video.
    video_scores = []
    for (pair_dir, variant), recs in by_video.items():
        poses = [canonicalize_to_a2b(np.array(r["T_rel"], dtype=np.float64), variant) for r in recs]
        m_idx, d_med = medoid_index_and_score(poses)
        T_med = poses[m_idx]

        d_bias = None
        d_total = d_med
        if args.use_bias:
            # Paper Eq. (9): compare medoid with pair-only pose from original inputs.
            # In our subset pipeline, each subset folder explicitly stores those two inputs
            # as 00_input_start.png and 01_input_end.png; use them directly.
            subset_dir = Path(recs[0]["subset_dir"])
            start_img = subset_dir / "00_input_start.png"
            end_img = subset_dir / "01_input_end.png"
            if start_img.is_file() and end_img.is_file():
                T_pair = run_dust3r_pair_pose(
                    model=model,
                    device=device,
                    path_a=start_img,
                    path_b=end_img,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                )
                T_pair = canonicalize_to_a2b(T_pair, variant)
                d_bias = pose_dist(T_med, T_pair)
                d_total = d_med + d_bias
            else:
                d_bias = float("nan")
                d_total = float("inf")

        video_scores.append(
            {
                "pair_dir": pair_dir,
                "variant": variant,
                "num_subsets": len(recs),
                "medoid_subset_idx": m_idx,
                "D_med": d_med,
                "D_bias": d_bias,
                "D_total": d_total,
                "pose_direction": "A2B",
                "T_med": T_med.tolist(),
            }
        )

    # Select best video per pair by D_total.
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for s in video_scores:
        by_pair[s["pair_dir"]].append(s)

    selected = []
    for pair_dir, items in by_pair.items():
        best = min(items, key=lambda x: float(x["D_total"]))
        selected.append(
            {
                "pair_dir": pair_dir,
                "selected_variant": best["variant"],
                "D_total": best["D_total"],
                "D_med": best["D_med"],
                "D_bias": best["D_bias"],
                "pose_direction": "A2B",
                "T_rel_selected": best["T_med"],
            }
        )

    # Write outputs.
    video_scores_path = args.out_dir / "video_scores.jsonl"
    selected_path = args.out_dir / "selected_per_pair.jsonl"
    with video_scores_path.open("w", encoding="utf-8") as f:
        for r in video_scores:
            f.write(json.dumps(r) + "\n")
    with selected_path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote video scores: {video_scores_path}")
    print(f"Wrote selected per pair: {selected_path}")


if __name__ == "__main__":
    main()
