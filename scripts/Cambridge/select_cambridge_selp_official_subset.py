#!/usr/bin/env python3
"""
Select a paper-approx Cambridge subset from the public official sELP metadata.

The public `selp_test_set.npy` metadata exposes many Cambridge pairs, but it does
not appear to contain the exact unpublished 290-pair slice referenced in the
paper. This script therefore creates a transparent, reproducible approximation:

1. start from the official Cambridge pair CSV extracted from the public metadata
2. keep only low-overlap pairs (`none` / `small` by default)
3. prefer pairs whose yaw difference already falls inside the requested band
4. if there are not enough such pairs, fill the remainder by smallest distance
   to the yaw band

It writes a new CSV plus a JSON summary so downstream evaluation can use a stable
subset without overwriting any earlier outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a paper-approx Cambridge subset from official sELP pairs."
    )
    parser.add_argument(
        "--in_csv",
        type=Path,
        default=Path("outputs/Cambridge/cambridge_selp_official_pairs.csv"),
        help="Input CSV created by extract_cambridge_selp_pairs.py.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("outputs/Cambridge/cambridge_selp_official_paper_approx_290.csv"),
        help="Selected output CSV.",
    )
    parser.add_argument(
        "--out_summary_json",
        type=Path,
        default=Path(
            "outputs/Cambridge/cambridge_selp_official_paper_approx_290_summary.json"
        ),
        help="Selection summary JSON.",
    )
    parser.add_argument(
        "--target_pairs",
        type=int,
        default=290,
        help="Requested number of selected pairs.",
    )
    parser.add_argument(
        "--yaw_min",
        type=float,
        default=50.0,
        help="Lower bound of the preferred yaw band in degrees.",
    )
    parser.add_argument(
        "--yaw_max",
        type=float,
        default=65.0,
        help="Upper bound of the preferred yaw band in degrees.",
    )
    parser.add_argument(
        "--allowed_overlap",
        nargs="+",
        default=["none", "small"],
        help="Allowed official overlap labels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for scene-balanced sampling.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def yaw_band_distance(yaw_deg: float, yaw_min: float, yaw_max: float) -> float:
    if yaw_deg < yaw_min:
        return yaw_min - yaw_deg
    if yaw_deg > yaw_max:
        return yaw_deg - yaw_max
    return 0.0


def midpoint_distance(yaw_deg: float, yaw_min: float, yaw_max: float) -> float:
    return abs(yaw_deg - 0.5 * (yaw_min + yaw_max))


def rank_rows(
    rows: Iterable[dict],
    yaw_min: float,
    yaw_max: float,
    allowed_overlap: List[str],
) -> List[dict]:
    overlap_rank = {name: idx for idx, name in enumerate(allowed_overlap)}
    ranked = []
    for row in rows:
        overlap = row.get("overlap_amount", "").strip()
        if overlap not in overlap_rank:
            continue
        yaw = float(row["yaw_diff_deg"])
        score = (
            yaw_band_distance(yaw, yaw_min, yaw_max),
            overlap_rank[overlap],
            midpoint_distance(yaw, yaw_min, yaw_max),
            row.get("scene_name", ""),
            row.get("sequence_name", ""),
            row.get("rel_a", ""),
            row.get("rel_b", ""),
        )
        enriched = dict(row)
        enriched["_yaw"] = yaw
        enriched["_score"] = score
        ranked.append(enriched)
    ranked.sort(key=lambda row: row["_score"])
    return ranked


def shuffle_equal_score_runs(rows: List[dict], rng: random.Random) -> List[dict]:
    shuffled = list(rows)
    start = 0
    while start < len(shuffled):
        end = start + 1
        while end < len(shuffled) and shuffled[end]["_score"] == shuffled[start]["_score"]:
            end += 1
        if end - start > 1:
            chunk = shuffled[start:end]
            rng.shuffle(chunk)
            shuffled[start:end] = chunk
        start = end
    return shuffled


def round_robin_pick(grouped_rows: Dict[str, List[dict]], limit: int, rng: random.Random) -> List[dict]:
    active_scenes = [scene for scene, rows in grouped_rows.items() if rows]
    rng.shuffle(active_scenes)
    selected: List[dict] = []
    ptrs = {scene: 0 for scene in active_scenes}

    while len(selected) < limit and active_scenes:
        progressed = False
        next_active = []
        for scene in active_scenes:
            ptr = ptrs[scene]
            rows = grouped_rows[scene]
            if ptr >= len(rows):
                continue
            selected.append(rows[ptr])
            ptrs[scene] = ptr + 1
            progressed = True
            if ptrs[scene] < len(rows):
                next_active.append(scene)
            if len(selected) >= limit:
                break
        if not progressed:
            break
        rng.shuffle(next_active)
        active_scenes = next_active

    return selected


def select_scene_balanced(
    ranked: List[dict],
    target_pairs: int,
    yaw_min: float,
    yaw_max: float,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    in_band = [
        row for row in ranked if yaw_band_distance(float(row["yaw_diff_deg"]), yaw_min, yaw_max) == 0.0
    ]
    out_of_band = [
        row for row in ranked if yaw_band_distance(float(row["yaw_diff_deg"]), yaw_min, yaw_max) != 0.0
    ]

    def group_and_shuffle(rows: List[dict]) -> Dict[str, List[dict]]:
        grouped: Dict[str, List[dict]] = {}
        for row in rows:
            grouped.setdefault(row.get("scene_name", ""), []).append(row)
        for scene, scene_rows in grouped.items():
            grouped[scene] = shuffle_equal_score_runs(scene_rows, rng)
        return grouped

    selected = round_robin_pick(group_and_shuffle(in_band), target_pairs, rng)
    if len(selected) < target_pairs:
        remaining = target_pairs - len(selected)
        selected.extend(round_robin_pick(group_and_shuffle(out_of_band), remaining, rng))
    return selected[:target_pairs]


def summarize(
    selected: List[dict],
    candidate_count: int,
    yaw_min: float,
    yaw_max: float,
    allowed_overlap: List[str],
    seed: int,
) -> dict:
    if not selected:
        return {
            "note": "No pairs satisfied the requested overlap filter.",
            "candidate_count_after_overlap_filter": candidate_count,
            "selected_count": 0,
            "target_yaw_range_deg": [yaw_min, yaw_max],
            "allowed_overlap": allowed_overlap,
            "seed": seed,
        }

    yaw_vals = [float(row["yaw_diff_deg"]) for row in selected]
    in_band = [
        row for row in selected if yaw_band_distance(float(row["yaw_diff_deg"]), yaw_min, yaw_max) == 0.0
    ]
    overlap_counts = Counter(row.get("overlap_amount", "") for row in selected)
    per_scene_counts = Counter(row.get("scene_name", "") for row in selected)

    return {
        "note": (
            "This is a public-metadata approximation of the paper subset: filter to "
            "official Cambridge pairs with allowed overlap labels, then sample in a "
            "scene-balanced round-robin order while preferring yaw in range and "
            "filling the remaining quota from the nearest out-of-range pairs."
        ),
        "candidate_count_after_overlap_filter": candidate_count,
        "selected_count": len(selected),
        "selected_in_target_yaw_range_count": len(in_band),
        "selected_outside_target_yaw_range_count": len(selected) - len(in_band),
        "target_yaw_range_deg": [yaw_min, yaw_max],
        "allowed_overlap": allowed_overlap,
        "seed": seed,
        "selected_overlap_counts": dict(sorted(overlap_counts.items())),
        "selected_scene_counts": dict(sorted(per_scene_counts.items())),
        "selected_yaw_stats_deg": {
            "min": min(yaw_vals),
            "median": sorted(yaw_vals)[len(yaw_vals) // 2],
            "max": max(yaw_vals),
            "mean": sum(yaw_vals) / len(yaw_vals),
        },
    }


def main() -> None:
    args = parse_args()
    rows = read_rows(args.in_csv)
    ranked = rank_rows(rows, args.yaw_min, args.yaw_max, args.allowed_overlap)
    selected = select_scene_balanced(
        ranked=ranked,
        target_pairs=args.target_pairs,
        yaw_min=args.yaw_min,
        yaw_max=args.yaw_max,
        seed=args.seed,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError(f"Input CSV is empty: {args.in_csv}")
    fieldnames = list(rows[0].keys())
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = summarize(
        selected=selected,
        candidate_count=len(ranked),
        yaw_min=args.yaw_min,
        yaw_max=args.yaw_max,
        allowed_overlap=args.allowed_overlap,
        seed=args.seed,
    )
    args.out_summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"Wrote {len(selected)} pairs to {args.out_csv} "
        f"(from {len(ranked)} overlap-filtered candidates)."
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
