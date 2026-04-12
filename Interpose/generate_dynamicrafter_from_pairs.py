#!/usr/bin/env python3
"""Pairs CSV → video via DynamiCrafter only (512 interp or 1024 I2V). No alternate synthesizers."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms


FIXED_PROMPT = (
    "indoor greenhouse with mannequins on a white platform, plants and flowers, "
    "smooth natural camera motion"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DynamiCrafter-only: read a pairs CSV (rel_a, rel_b) and render MP4s. "
            "320_512 = DynamiCrafter 512_interp (two keyframes + prompt). "
            "576_1024 = DynamiCrafter 1024 single-image I2V (one keyframe + prompt; not two-frame interp)."
        )
    )
    parser.add_argument("--pairs_csv", type=Path, required=True, help="CSV with rel_a and rel_b columns.")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root directory where rel_a and rel_b paths are resolved from.",
    )
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--resolution",
        type=str,
        default="320_512",
        choices=["320_512", "576_1024"],
        help=(
            "320_512 = DynamiCrafter512_interp (two-keyframe interp). "
            "576_1024 = DynamiCrafter1024 single-image I2V (uses start image only per job)."
        ),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument(
        "--fs",
        type=int,
        default=-1,
        help=(
            "FS / fps conditioning token. "
            "If omitted (default -1): use 5 for 320_512 interp, 10 for 576_1024 (matches scripts/run.sh 1024)."
        ),
    )
    parser.add_argument("--seed1", type=int, default=123)
    parser.add_argument("--seed2", type=int, default=456)
    parser.add_argument("--max_pairs", type=int, default=0, help="0 means use all pairs.")
    parser.add_argument(
        "--pair_index",
        type=int,
        default=None,
        help=(
            "If set, only this 0-based CSV data row is processed (after header). "
            "Default output for that row is a single A->B clip unless --b2a / --dual_seed / --all_variants."
        ),
    )
    parser.add_argument(
        "--b2a",
        action="store_true",
        help="When used with --pair_index, interpolate B->A instead of A->B.",
    )
    parser.add_argument(
        "--dual_seed",
        action="store_true",
        help="When used with --pair_index, also render a second video with --seed2.",
    )
    parser.add_argument(
        "--all_variants",
        action="store_true",
        help="When used with --pair_index, render the same 4 videos as batch mode for that row.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=FIXED_PROMPT,
        help="Text prompt passed to DynamiCrafter interpolation.",
    )
    parser.add_argument(
        "--upscale_to_input",
        action="store_true",
        help=(
            "After generation, resize every frame to the start image's native H×W (e.g. 540×960). "
            "The model still runs at its fixed canvas (320×512 or 576×1024); this only upscales pixels — "
            "it does not recover regions removed by the model's center crop."
        ),
    )
    parser.add_argument(
        "--fit_letterbox",
        action="store_true",
        help=(
            "512-interp only: scale each keyframe to fit inside the model canvas (no cropping of image "
            "content), pad to 320×512, run DynamiCrafter, then map every output frame back to the original "
            "pixel H×W. Requires start/end images with the same dimensions. Ignores --upscale_to_input."
        ),
    )
    parser.add_argument(
        "--resize_to_canvas",
        action="store_true",
        help=(
            "512-interp only: independently stretch each keyframe to exactly 320×512 (LANCZOS) before "
            "DynamiCrafter. Aspect ratio is not preserved unless the source is already 320:512. Mutually "
            "exclusive with --fit_letterbox. Output MP4 is 320×512 unless --upscale_to_input."
        ),
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=None,
        help=(
            "FPS used when writing the final MP4 (after pinning / upscale / unletterbox). "
            "Default: DynamiCrafter's save_fps (8). The model still emits a fixed number of frames "
            "(~16 for 512 interp); lowering FPS stretches playback (e.g. ~10s at 1.6fps for 16 frames) "
            "without generating new in-between content."
        ),
    )
    parser.add_argument(
        "--target_duration_sec",
        type=float,
        default=None,
        help=(
            "Convenience: set output FPS as (expected frame count / this value) so the clip is about "
            "this many seconds long. Ignored if --output_fps is set. Example: 10 → ~1.6fps for 16 frames."
        ),
    )
    return parser.parse_args()


_PLACEHOLDER_PROMPT = re.compile(
    r"^\s*(your[\s_]*prompt[\s_]*here|prompt\s*here|todo|tbd)\s*$",
    re.IGNORECASE,
)


def _die_if_bad_prompt(prompt: str) -> None:
    p = prompt.strip()
    if not p:
        print("[error] --prompt is empty.", file=sys.stderr)
        sys.exit(1)
    if _PLACEHOLDER_PROMPT.match(p):
        print(
            "[error] --prompt looks like a placeholder (e.g. 'your prompt here'). "
            "Use text that describes your real frames (scene + desired motion).",
            file=sys.stderr,
        )
        sys.exit(1)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def sanitize_name(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def stretch_pair_to_canvas(
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    canvas_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resize each image to exactly canvas_hw (H×W) with independent scaling (may change aspect ratio)."""
    ch, cw = int(canvas_hw[0]), int(canvas_hw[1])

    def _one(rgb: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(rgb).resize((cw, ch), Image.Resampling.LANCZOS)
        return np.asarray(pil, dtype=np.uint8)

    return _one(rgb_a), _one(rgb_b)


def letterbox_pair_to_canvas(
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    canvas_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Resize both images with the same uniform scale so they fit inside canvas_hw (contain),
    center-pad to exact canvas size. No cropping of source pixels.
    """
    ch, cw = int(canvas_hw[0]), int(canvas_hw[1])
    ha, wa = rgb_a.shape[0], rgb_a.shape[1]
    hb, wb = rgb_b.shape[0], rgb_b.shape[1]
    if (ha, wa) != (hb, wb):
        raise ValueError(
            f"--fit_letterbox requires matching image sizes; got {ha}x{wa} vs {hb}x{wb}"
        )
    ho, wo = ha, wa
    scale = min(ch / ho, cw / wo)
    nh = max(1, int(round(ho * scale)))
    nw = max(1, int(round(wo * scale)))
    pt, pl = (ch - nh) // 2, (cw - nw) // 2

    def _one(rgb: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS)
        arr = np.asarray(pil, dtype=np.uint8)
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        canvas[pt : pt + nh, pl : pl + nw] = arr
        return canvas

    meta = {
        "orig_hw": (ho, wo),
        "nh": nh,
        "nw": nw,
        "pt": pt,
        "pl": pl,
        "canvas_hw": (ch, cw),
    }
    return _one(rgb_a), _one(rgb_b), meta


def unletterbox_video(mp4_path: Path, meta: dict, fps: int) -> None:
    """Map each 320×512 (letterboxed) frame back to orig_hw via crop + bicubic resize."""
    ho, wo = int(meta["orig_hw"][0]), int(meta["orig_hw"][1])
    nh, nw = int(meta["nh"]), int(meta["nw"])
    pt, pl = int(meta["pt"]), int(meta["pl"])
    video, _, _ = torchvision.io.read_video(str(mp4_path), pts_unit="sec")
    if video.numel() == 0:
        raise RuntimeError(f"no frames in {mp4_path}")
    ch, cw = int(video.shape[1]), int(video.shape[2])
    if ch != meta["canvas_hw"][0] or cw != meta["canvas_hw"][1]:
        raise RuntimeError(f"video {ch}x{cw} != letterbox canvas {meta['canvas_hw']}")

    outs = []
    for i in range(video.shape[0]):
        patch = video[i, pt : pt + nh, pl : pl + nw, :].permute(2, 0, 1).float().unsqueeze(0) / 255.0
        up = torch.nn.functional.interpolate(patch, size=(ho, wo), mode="bicubic", align_corners=False)
        outs.append((up.squeeze(0) * 255.0).round().clamp(0, 255).byte().permute(1, 2, 0))
    stacked = torch.stack(outs, dim=0)
    torchvision.io.write_video(
        str(mp4_path),
        stacked,
        fps=fps,
        video_codec="h264",
        options={"crf": "10"},
    )


def _rgb_to_model_canvas(rgb: np.ndarray, resolution_hw: tuple[int, int]) -> np.ndarray:
    """Match DynamiCrafter Image2Video (CHW float 0–255, same Resize/CenterCrop as get_image)."""
    h_t, w_t = resolution_hw
    tform = transforms.Compose(
        [transforms.Resize(min(h_t, w_t)), transforms.CenterCrop((h_t, w_t))]
    )
    chw = torch.from_numpy(rgb).permute(2, 0, 1).float()
    out = tform(chw)
    return out.round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


def upscale_video_to_hw(mp4_path: Path, target_hw: tuple[int, int], fps: int) -> None:
    """Resize each decoded frame to target (height, width); expects THWC uint8 RGB."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    video, _, _ = torchvision.io.read_video(str(mp4_path), pts_unit="sec")
    if video.numel() == 0:
        raise RuntimeError(f"no frames to upscale: {mp4_path}")
    h, w = int(video.shape[1]), int(video.shape[2])
    if h == th and w == tw:
        return
    x = video.permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False)
    out = (x * 255.0).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(
        str(mp4_path),
        out,
        fps=fps,
        video_codec="h264",
        options={"crf": "10"},
    )


def pin_i2v_first_frame(
    mp4_path: Path,
    image_start: np.ndarray,
    resolution_hw: tuple[int, int],
    fps: int = 8,
) -> None:
    """Force frame 0 to match the conditioned still (1024 single-image path has no end keyframe)."""
    video, _audio, _meta = torchvision.io.read_video(str(mp4_path), pts_unit="sec")
    if video.numel() == 0:
        raise RuntimeError(f"no frames in {mp4_path}")

    pin0 = _rgb_to_model_canvas(image_start, resolution_hw)
    if tuple(video.shape[1:3]) != (pin0.shape[0], pin0.shape[1]):
        raise RuntimeError(
            f"video spatial {tuple(video.shape[1:3])} != pinned canvas {pin0.shape[:2]} for {mp4_path}"
        )

    video = video.clone()
    video[0] = torch.from_numpy(pin0)
    torchvision.io.write_video(
        str(mp4_path),
        video,
        fps=fps,
        video_codec="h264",
        options={"crf": "10"},
    )


def pin_interpolation_endpoints(
    mp4_path: Path,
    image_start: np.ndarray,
    image_end: np.ndarray,
    resolution_hw: tuple[int, int],
    fps: int = 8,
    exact_pins: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """
    Overwrite the first and last decoded frames with the actual start/end RGB inputs
    (after the same resize/crop as the model), so the MP4 visibly begins and ends on those images.
    If exact_pins is set, use those uint8 H×W×3 arrays (already model canvas pixels) instead of resize+crop.
    """
    video, _audio, meta = torchvision.io.read_video(str(mp4_path), pts_unit="sec")
    if video.numel() == 0 or video.shape[0] < 2:
        raise RuntimeError(f"expected at least 2 frames in {mp4_path}, got shape {tuple(video.shape)}")

    if exact_pins is not None:
        pin0, pin1 = exact_pins
    else:
        pin0 = _rgb_to_model_canvas(image_start, resolution_hw)
        pin1 = _rgb_to_model_canvas(image_end, resolution_hw)
    if tuple(video.shape[1:3]) != (pin0.shape[0], pin0.shape[1]):
        raise RuntimeError(
            f"video spatial {tuple(video.shape[1:3])} != pinned canvas {pin0.shape[:2]} for {mp4_path}"
        )

    video = video.clone()
    video[0] = torch.from_numpy(pin0)
    video[-1] = torch.from_numpy(pin1)
    torchvision.io.write_video(
        str(mp4_path),
        video,
        fps=fps,
        video_codec="h264",
        options={"crf": "10"},
    )


def main() -> None:
    args = parse_args()
    _die_if_bad_prompt(args.prompt)
    args.pairs_csv = args.pairs_csv.resolve()
    args.data_root = args.data_root.resolve()
    args.out_dir = args.out_dir.resolve()

    if args.fit_letterbox and args.resolution != "320_512":
        print("[error] --fit_letterbox only applies to --resolution 320_512 (512 interp).", file=sys.stderr)
        sys.exit(1)
    if args.resize_to_canvas and args.resolution != "320_512":
        print("[error] --resize_to_canvas only applies to --resolution 320_512 (512 interp).", file=sys.stderr)
        sys.exit(1)
    if args.fit_letterbox and args.resize_to_canvas:
        print("[error] use either --fit_letterbox or --resize_to_canvas, not both.", file=sys.stderr)
        sys.exit(1)
    if args.output_fps is not None and args.output_fps <= 0:
        print("[error] --output_fps must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.target_duration_sec is not None and args.target_duration_sec <= 0:
        print("[error] --target_duration_sec must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.output_fps is not None and args.target_duration_sec is not None:
        print("[warn] both --output_fps and --target_duration_sec set; using --output_fps.", file=sys.stderr)

    if not args.pairs_csv.is_file():
        raise FileNotFoundError(f"--pairs_csv not found: {args.pairs_csv}")
    if not args.data_root.is_dir():
        raise NotADirectoryError(f"--data_root is not a directory: {args.data_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out_dir / "tmp_dyn"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # File is placed at camera_pose/Interpose/generate_dynamicrafter_from_pairs.py
    # so parent[1] is camera_pose root.
    root = Path(__file__).resolve().parents[1]
    dyn_root = root / "DynamiCrafter"
    if not dyn_root.is_dir():
        raise FileNotFoundError(f"DynamiCrafter folder not found: {dyn_root}")

    sys.path.insert(0, str(dyn_root))
    use_i2v_1024 = args.resolution == "576_1024"
    if use_i2v_1024:
        from scripts.gradio.i2v_test import Image2Video  # pylint: disable=import-outside-toplevel
    else:
        from scripts.gradio.i2v_test_application import Image2Video  # pylint: disable=import-outside-toplevel

    old_cwd = Path.cwd()
    os.chdir(dyn_root)
    try:
        i2v = Image2Video(result_dir=str(tmp_dir.resolve()), resolution=args.resolution)
        res_hw = tuple(int(x) for x in args.resolution.split("_"))  # (H, W)
        model = i2v.model_list[0]
        exp_frames = int(getattr(model, "temporal_length", 16))
        if use_i2v_1024:
            exp_frames = max(1, exp_frames - 1)  # i2v_test.get_image drops last temporal slice
        base_fps = float(getattr(i2v, "save_fps", 8))
        if args.output_fps is not None:
            video_fps = float(args.output_fps)
        elif args.target_duration_sec is not None:
            video_fps = max(1e-6, float(exp_frames) / float(args.target_duration_sec))
        else:
            video_fps = base_fps
        if args.output_fps is not None or args.target_duration_sec is not None:
            approx_dur = exp_frames / video_fps
            print(
                f"[note] model emits ~{exp_frames} frames; output_fps={video_fps:.4g} → "
                f"~{approx_dur:.2f}s playback (no extra frames synthesized).",
                file=sys.stderr,
            )
        fs = args.fs if args.fs >= 0 else (10 if use_i2v_1024 else 5)

        records = []
        with args.pairs_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"rel_a", "rel_b"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"pairs_csv must contain columns {sorted(required)}; got {reader.fieldnames}"
                )

            for pair_idx, row in enumerate(reader):
                if args.pair_index is not None and pair_idx != args.pair_index:
                    continue
                if args.pair_index is None and args.max_pairs > 0 and pair_idx >= args.max_pairs:
                    break

                rel_a = row["rel_a"]
                rel_b = row["rel_b"]
                scene = row.get("scene", "unknown_scene")

                img_a_path = args.data_root / rel_a
                img_b_path = args.data_root / rel_b
                if not img_a_path.is_file() or not img_b_path.is_file():
                    print(f"[skip] missing image file(s): {img_a_path} | {img_b_path}")
                    continue

                img_a = load_rgb(img_a_path)
                img_b = load_rgb(img_b_path)

                letter_meta: dict | None = None
                feed_a, feed_b = img_a, img_b
                if args.resize_to_canvas:
                    if use_i2v_1024:
                        print("[warn] --resize_to_canvas ignored for 576_1024.", file=sys.stderr)
                    else:
                        feed_a, feed_b = stretch_pair_to_canvas(img_a, img_b, res_hw)
                elif args.fit_letterbox:
                    if use_i2v_1024:
                        print("[warn] --fit_letterbox ignored for 576_1024.", file=sys.stderr)
                    else:
                        feed_a, feed_b, letter_meta = letterbox_pair_to_canvas(img_a, img_b, res_hw)

                pair_tag = f"pair_{pair_idx:05d}_{sanitize_name(scene)}"
                pair_out_dir = videos_dir / pair_tag
                pair_out_dir.mkdir(parents=True, exist_ok=True)

                if args.all_variants or args.pair_index is None:
                    jobs = [
                        ("A2B", args.seed1, img_a, img_b),
                        ("A2B", args.seed2, img_a, img_b),
                        ("B2A", args.seed1, img_b, img_a),
                        ("B2A", args.seed2, img_b, img_a),
                    ]
                else:
                    if args.b2a:
                        image_start, image_end = img_b, img_a
                        direction_prefix = "B2A"
                    else:
                        image_start, image_end = img_a, img_b
                        direction_prefix = "A2B"
                    jobs = [(direction_prefix, args.seed1, image_start, image_end)]
                    if args.dual_seed:
                        jobs.append((direction_prefix, args.seed2, image_start, image_end))

                use_canvas_feeds = (not use_i2v_1024) and (
                    (args.fit_letterbox and letter_meta is not None) or args.resize_to_canvas
                )
                for vid_idx, (direction, seed, image_start, image_end) in enumerate(jobs):
                    g_start = feed_a if direction.startswith("A2B") else feed_b
                    g_end = feed_b if direction.startswith("A2B") else feed_a
                    if not use_canvas_feeds:
                        g_start, g_end = image_start, image_end

                    if use_i2v_1024:
                        src_mp4 = i2v.get_image(
                            image=g_start,
                            prompt=args.prompt,
                            steps=args.steps,
                            cfg_scale=args.cfg_scale,
                            eta=args.eta,
                            fs=fs,
                            seed=seed,
                        )
                    else:
                        src_mp4 = i2v.get_image(
                            image=g_start,
                            image2=g_end,
                            prompt=args.prompt,
                            steps=args.steps,
                            cfg_scale=args.cfg_scale,
                            eta=args.eta,
                            fs=fs,
                            seed=seed,
                        )

                    out_name = f"{pair_tag}_v{vid_idx}_{direction}_seed{seed}.mp4"
                    out_path = pair_out_dir / out_name
                    shutil.copy2(src_mp4, out_path)
                    if use_i2v_1024:
                        pin_i2v_first_frame(
                            out_path,
                            g_start,
                            resolution_hw=res_hw,
                            fps=video_fps,
                        )
                    else:
                        exact = None
                        if use_canvas_feeds:
                            exact = (g_start.copy(), g_end.copy())
                        pin_interpolation_endpoints(
                            out_path,
                            image_start,
                            image_end,
                            resolution_hw=res_hw,
                            fps=video_fps,
                            exact_pins=exact,
                        )

                    in_hw = (int(image_start.shape[0]), int(image_start.shape[1]))
                    if args.fit_letterbox and letter_meta is not None and not use_i2v_1024:
                        unletterbox_video(out_path, letter_meta, fps=video_fps)
                    elif args.upscale_to_input:
                        upscale_video_to_hw(out_path, in_hw, fps=video_fps)

                    records.append(
                        {
                            "pair_idx": pair_idx,
                            "scene": scene,
                            "rel_a": rel_a,
                            "rel_b": rel_b,
                            "video_idx": vid_idx,
                            "direction": direction,
                            "seed": seed,
                            "prompt": args.prompt,
                            "resolution": args.resolution,
                            "fit_letterbox": bool(args.fit_letterbox and letter_meta is not None),
                            "resize_to_canvas": bool(args.resize_to_canvas and not use_i2v_1024),
                            "upscale_to_input": bool(args.upscale_to_input) and not (
                                args.fit_letterbox and letter_meta is not None
                            ),
                            "output_hw": list(in_hw)
                            if (
                                args.upscale_to_input
                                or (args.fit_letterbox and letter_meta is not None)
                            )
                            else None,
                            "video_path": str(out_path.relative_to(args.out_dir)),
                            "output_fps": float(video_fps),
                            "target_duration_sec": args.target_duration_sec,
                            "expected_model_frames": exp_frames,
                        }
                    )

                print(f"[ok] {pair_tag}: generated {len(jobs)} video(s)")

        if args.pair_index is not None and not records:
            print(
                f"[error] no video written for --pair_index {args.pair_index} "
                "(index past end of CSV, or missing image files for that row).",
                file=sys.stderr,
            )
            sys.exit(1)

        manifest_path = args.out_dir / "videos_manifest.jsonl"
        with manifest_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"\nDone. Wrote manifest: {manifest_path.resolve()}")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
