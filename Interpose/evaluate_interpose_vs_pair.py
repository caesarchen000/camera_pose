#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate InterPose-selected poses against GT and compare with direct DUSt3R pair-only "
            "poses on the same pairs."
        )
    )
    p.add_argument("--pairs_csv", type=Path, required=True, help="CSV used to define pair list (rel_a/rel_b).")
    p.add_argument("--data_root", type=Path, required=True, help="DL3DV benchmark root.")
    p.add_argument(
        "--selected_jsonl",
        type=Path,
        required=True,
        help="Selection output JSONL (e.g. out_root/selection/selected_per_pair.jsonl).",
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="DUSt3R checkpoint path.")
    p.add_argument("--dust3r_root", type=Path, default=Path("dust3r"))
    p.add_argument("--image_size", type=int, default=512, choices=(224, 512))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--interpose_only",
        action="store_true",
        help="Evaluate only InterPose-selected poses against GT (skip pair-only DUSt3R baseline).",
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Optional per-pair metrics CSV output (default: next to selected_jsonl).",
    )
    return p.parse_args()


def geodesic_rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_err = R_pred @ R_gt.T
    c = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def translation_direction_angle_error_deg(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    np_pred = float(np.linalg.norm(t_pred))
    np_gt = float(np.linalg.norm(t_gt))
    if np_gt < 1e-9:
        return float("nan")
    if np_pred < 1e-9:
        return 180.0
    d = float(np.clip(np.dot(t_pred / np_pred, t_gt / np_gt), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def load_gt_rel(data_root: Path, rel_a: str, rel_b: str) -> np.ndarray:
    scene_hash = rel_a.split("/")[0]
    tf_path = data_root / scene_hash / "nerfstudio" / "transforms.json"
    d = json.loads(tf_path.read_text(encoding="utf-8"))
    frame_map = {fr["file_path"].lstrip("./"): np.array(fr["transform_matrix"], dtype=np.float64) for fr in d["frames"]}

    key_a = "/".join(rel_a.split("/")[2:]).replace("images_4/", "images/")
    key_b = "/".join(rel_b.split("/")[2:]).replace("images_4/", "images/")
    if key_a not in frame_map or key_b not in frame_map:
        raise KeyError(f"GT keys missing in transforms: {key_a}, {key_b}")

    # Same convention conversion used in eval_dust3r_dl3dv.py
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    T_wc_a = np.linalg.inv(frame_map[key_a] @ flip)
    T_wc_b = np.linalg.inv(frame_map[key_b] @ flip)
    return T_wc_b @ np.linalg.inv(T_wc_a)


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


def summary(vals: list[float]) -> tuple[float, float]:
    arr = np.array(vals, dtype=np.float64)
    m = np.isfinite(arr)
    if not m.any():
        return float("nan"), float("nan")
    arr = arr[m]
    return float(np.mean(arr)), float(np.median(arr))


def pose_auc_deg(errors_deg: list[float], thresholds: tuple[float, ...] = (5.0, 10.0, 20.0)) -> dict[float, float]:
    arr = np.array(errors_deg, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {th: float("nan") for th in thresholds}
    arr = np.sort(arr)
    out: dict[float, float] = {}
    for th in thresholds:
        capped = np.minimum(arr, th)
        # Normalized integral of recall-error curve in [0, th].
        out[th] = float(np.sum((th - capped) / th) / arr.size)
    return out


def main() -> None:
    args = parse_args()
    pairs_csv = args.pairs_csv.resolve()
    data_root = args.data_root.resolve()
    selected_jsonl = args.selected_jsonl.resolve()
    ckpt = args.checkpoint.resolve()
    dust3r_root = args.dust3r_root.resolve()
    out_csv = (
        args.out_csv.resolve()
        if args.out_csv is not None
        else selected_jsonl.parent / "interpose_vs_pair_metrics.csv"
    )

    with pairs_csv.open("r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    selected_rows = [json.loads(l) for l in selected_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not selected_rows:
        raise ValueError(f"No rows in selected file: {selected_jsonl}")

    device = torch.device(args.device)
    model = None
    if not args.interpose_only:
        if str(dust3r_root) not in sys.path:
            sys.path.insert(0, str(dust3r_root))
        from dust3r.model import load_model  # pylint: disable=import-outside-toplevel

        model = load_model(str(ckpt), device)
        model.eval()

    metrics_rows: list[dict] = []
    inter_re_all, inter_te_all = [], []
    pair_re_all, pair_te_all = [], []
    delta_re_all, delta_te_all = [], []

    for rec in selected_rows:
        pair_name = Path(rec["pair_dir"]).name
        pair_idx = int(pair_name.split("_")[1])
        if pair_idx < 0 or pair_idx >= len(csv_rows):
            print(f"[skip] pair index {pair_idx} out of CSV range")
            continue

        row = csv_rows[pair_idx]
        rel_a = row["rel_a"]
        rel_b = row["rel_b"]
        path_a = data_root / rel_a
        path_b = data_root / rel_b

        T_gt = load_gt_rel(data_root, rel_a, rel_b)
        T_inter = np.array(rec["T_rel_selected"], dtype=np.float64)
        if not args.interpose_only:
            assert model is not None
            T_pair = run_dust3r_pair_pose(
                model=model,
                device=device,
                path_a=path_a,
                path_b=path_b,
                image_size=args.image_size,
                batch_size=args.batch_size,
            )

        inter_re = geodesic_rotation_error_deg(T_inter[:3, :3], T_gt[:3, :3])
        inter_te = translation_direction_angle_error_deg(T_inter[:3, 3], T_gt[:3, 3])
        if not args.interpose_only:
            pair_re = geodesic_rotation_error_deg(T_pair[:3, :3], T_gt[:3, :3])
            pair_te = translation_direction_angle_error_deg(T_pair[:3, 3], T_gt[:3, 3])
            delta_re = inter_re - pair_re
            delta_te = inter_te - pair_te
            pair_re_all.append(pair_re)
            pair_te_all.append(pair_te)
            delta_re_all.append(delta_re)
            delta_te_all.append(delta_te)
        else:
            pair_re = float("nan")
            pair_te = float("nan")
            delta_re = float("nan")
            delta_te = float("nan")

        inter_re_all.append(inter_re)
        inter_te_all.append(inter_te)

        metrics_rows.append(
            {
                "pair_idx": pair_idx,
                "pair_dir": rec["pair_dir"],
                "selected_variant": rec.get("selected_variant", ""),
                "interpose_RE_deg": inter_re,
                "interpose_TE_deg": inter_te,
                "pair_only_RE_deg": pair_re,
                "pair_only_TE_deg": pair_te,
                "delta_RE_deg_inter_minus_pair": delta_re,
                "delta_TE_deg_inter_minus_pair": delta_te,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_idx",
        "pair_dir",
        "selected_variant",
        "interpose_RE_deg",
        "interpose_TE_deg",
        "pair_only_RE_deg",
        "pair_only_TE_deg",
        "delta_RE_deg_inter_minus_pair",
        "delta_TE_deg_inter_minus_pair",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in metrics_rows:
            w.writerow(r)

    inter_re_mean, inter_re_med = summary(inter_re_all)
    inter_te_mean, inter_te_med = summary(inter_te_all)
    inter_pose_err = [max(r, t) for r, t in zip(inter_re_all, inter_te_all)]
    inter_auc = pose_auc_deg(inter_pose_err)
    pair_re_mean, pair_re_med = summary(pair_re_all)
    pair_te_mean, pair_te_med = summary(pair_te_all)
    delta_re_mean, delta_re_med = summary(delta_re_all)
    delta_te_mean, delta_te_med = summary(delta_te_all)
    pair_pose_err = [max(r, t) for r, t in zip(pair_re_all, pair_te_all)]
    pair_auc = pose_auc_deg(pair_pose_err)

    print(f"Wrote per-pair metrics: {out_csv}")
    print(f"Pairs evaluated: {len(metrics_rows)}")
    print("InterPose   RE mean/med: %.4f / %.4f | TE mean/med: %.4f / %.4f" % (inter_re_mean, inter_re_med, inter_te_mean, inter_te_med))
    print(
        "InterPose   AUC@5/10/20: %.4f / %.4f / %.4f"
        % (inter_auc[5.0], inter_auc[10.0], inter_auc[20.0])
    )
    if not args.interpose_only:
        print("Pair-only   RE mean/med: %.4f / %.4f | TE mean/med: %.4f / %.4f" % (pair_re_mean, pair_re_med, pair_te_mean, pair_te_med))
        print(
            "Pair-only   AUC@5/10/20: %.4f / %.4f / %.4f"
            % (pair_auc[5.0], pair_auc[10.0], pair_auc[20.0])
        )
        print("Delta(I-P)  RE mean/med: %.4f / %.4f | TE mean/med: %.4f / %.4f" % (delta_re_mean, delta_re_med, delta_te_mean, delta_te_med))


if __name__ == "__main__":
    main()
