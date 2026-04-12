#!/usr/bin/env python3
"""
Interpose video pipeline: CSV → frame paths + prompt string → (optional) DynamiCrafter 512 interp.

Functions are split so you can delete unused ones later; keep the flow obvious.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Default prompt (override with --prompt).
DEFAULT_PROMPT = (
    "a statue on the white stage with plants and flowers surrounding in a room"
)


def repository_root() -> Path:
    """camera_pose repo root (parent of Interpose/)."""
    return Path(__file__).resolve().parents[1]


def dynamicrafter_root() -> Path:
    return repository_root() / "DynamiCrafter"


def load_pair_row(
    pairs_csv: Path,
    data_root: Path,
    pair_index: int,
) -> tuple[Path, Path, str, dict[str, str]]:
    """One CSV row: absolute paths to start/end frames, scene tag, raw row dict."""
    pairs_csv = pairs_csv.resolve()
    data_root = data_root.resolve()
    if not pairs_csv.is_file():
        raise FileNotFoundError(f"CSV not found: {pairs_csv}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {data_root}")

    with pairs_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if not {"rel_a", "rel_b"}.issubset(set(fields)):
            raise ValueError(f"CSV must contain rel_a and rel_b columns; got {fields}")

        for i, row in enumerate(reader):
            if i != pair_index:
                continue
            rel_a = (row.get("rel_a") or "").strip()
            rel_b = (row.get("rel_b") or "").strip()
            if not rel_a or not rel_b:
                raise ValueError(f"Row {pair_index}: empty rel_a or rel_b")
            scene = (row.get("scene") or "unknown_scene").strip()
            path_a = (data_root / rel_a).resolve()
            path_b = (data_root / rel_b).resolve()
            if not path_a.is_file():
                raise FileNotFoundError(f"Start frame not found: {path_a}")
            if not path_b.is_file():
                raise FileNotFoundError(f"End frame not found: {path_b}")
            return path_a, path_b, scene, {k: (v or "") for k, v in row.items()}

    raise IndexError(
        f"No data row at pair_index={pair_index} (0-based; first data row is 0)."
    )


def resolve_inputs(
    pairs_csv: Path,
    data_root: Path,
    pair_index: int,
    prompt: str,
) -> dict[str, Any]:
    """Bundle passed to staging / inference: paths, scene, row, prompt, index."""
    path_a, path_b, scene, row = load_pair_row(pairs_csv, data_root, pair_index)
    p = prompt.strip()
    if not p:
        raise ValueError("prompt is empty after strip()")
    return {
        "path_start": path_a,
        "path_end": path_b,
        "scene": scene,
        "csv_row": row,
        "prompt": p,
        "pair_index": pair_index,
    }


def save_pre_model_inputs(
    dest_dir: Path,
    path_start: Path,
    path_end: Path,
    prompt: str,
    *,
    height: int,
    width: int,
) -> Path:
    """
    Write exactly what inference.py uses as pixel conditioning (same Resize/CenterCrop/Normalize
    as load_data_prompts), as human-readable PNGs plus prompt.txt.

    PNGs are the tensors after Normalize, denormed back to sRGB uint8 — same spatial crop the
    VAE sees at each timestep slot before encoding.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "prompt.txt").write_text(prompt.strip() + "\n", encoding="utf-8")

    video_size = (height, width)
    transform = T.Compose(
        [
            T.Resize(min(video_size)),
            T.CenterCrop(video_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    def _denorm_to_png(tensor_chw: torch.Tensor, out_path: Path) -> None:
        t = tensor_chw.detach().float().clamp(-1.0, 1.0)
        t = (t * 0.5 + 0.5) * 255.0
        arr = t.byte().cpu().permute(1, 2, 0).numpy()
        Image.fromarray(np.asarray(arr)).save(out_path)

    im0 = Image.open(path_start).convert("RGB")
    im1 = Image.open(path_end).convert("RGB")
    _denorm_to_png(transform(im0), dest_dir / "00_start_pre_model.png")
    _denorm_to_png(transform(im1), dest_dir / "01_end_pre_model.png")
    (dest_dir / "meta.txt").write_text(
        f"height={height} width={width}\n"
        "Same transform chain as DynamiCrafter/scripts/evaluation/inference.py load_data_prompts.\n",
        encoding="utf-8",
    )
    return dest_dir.resolve()


def stage_prompt_pack_interp(
    pack_dir: Path,
    path_start: Path,
    path_end: Path,
    prompt: str,
) -> Path:
    """
    Layout expected by DynamiCrafter scripts/evaluation/inference.py (interp):
    one *.txt (single line) and two images sorted as 00_*, 01_* for glob order.
    """
    pack_dir.mkdir(parents=True, exist_ok=True)
    dst0 = pack_dir / "00_start.png"
    dst1 = pack_dir / "01_end.png"
    shutil.copy2(path_start, dst0)
    shutil.copy2(path_end, dst1)
    (pack_dir / "prompt.txt").write_text(prompt.strip() + "\n", encoding="utf-8")
    return pack_dir.resolve()


def build_inference_args_512_interp(
    savedir: Path,
    prompt_dir: Path,
    *,
    seed: int = 12306,
    ddim_steps: int = 50,
    frame_stride: int = 5,
    cfg_scale: float = 7.5,
) -> SimpleNamespace:
    """Arguments compatible with scripts/evaluation/inference.py run_inference (512 interp)."""
    return SimpleNamespace(
        savedir=str(savedir.resolve()),
        ckpt_path="checkpoints/dynamicrafter_512_interp_v1/model.ckpt",
        config="configs/inference_512_v1.0.yaml",
        prompt_dir=str(prompt_dir.resolve()),
        n_samples=1,
        ddim_steps=ddim_steps,
        ddim_eta=1.0,
        bs=1,
        height=320,
        width=512,
        frame_stride=frame_stride,
        unconditional_guidance_scale=cfg_scale,
        seed=seed,
        video_length=16,
        negative_prompt=False,
        text_input=True,
        multiple_cond_cfg=False,
        cfg_img=None,
        timestep_spacing="uniform_trailing",
        guidance_rescale=0.7,
        perframe_ae=True,
        loop=False,
        interp=True,
    )


def run_dynamiCrafter_interp_inference(
    bundle: dict[str, Any],
    out_dir: Path,
    *,
    seed: int,
    ddim_steps: int,
    frame_stride: int,
    cfg_scale: float,
) -> Path:
    """
    Stage prompt_dir under out_dir, chdir to DynamiCrafter, call inference.run_inference.
    Returns savedir (same as out_dir).
    """
    dyn = dynamicrafter_root()
    if not dyn.is_dir():
        raise FileNotFoundError(f"DynamiCrafter not found: {dyn}")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = out_dir / "prompt_pack"
    inf_args = build_inference_args_512_interp(
        out_dir,
        pack_dir,
        seed=seed,
        ddim_steps=ddim_steps,
        frame_stride=frame_stride,
        cfg_scale=cfg_scale,
    )
    save_pre_model_inputs(
        out_dir / "pre_model_inputs",
        bundle["path_start"],
        bundle["path_end"],
        bundle["prompt"],
        height=int(inf_args.height),
        width=int(inf_args.width),
    )
    stage_prompt_pack_interp(
        pack_dir,
        bundle["path_start"],
        bundle["path_end"],
        bundle["prompt"],
    )

    old = Path.cwd()
    sys.path.insert(0, str(dyn))
    os.chdir(dyn)
    try:
        from pytorch_lightning import seed_everything  # pylint: disable=import-outside-toplevel
        from scripts.evaluation import inference as dyn_inference  # pylint: disable=import-outside-toplevel

        seed_everything(seed if seed >= 0 else 12306)
        dyn_inference.run_inference(inf_args, gpu_num=1, gpu_no=0)
    finally:
        os.chdir(old)
    return out_dir


def print_bundle_summary(bundle: dict[str, Any]) -> None:
    print("Resolved inputs:")
    print(f"  pair_index:  {bundle['pair_index']}")
    print(f"  scene:       {bundle['scene']}")
    print(f"  start image: {bundle['path_start']}")
    print(f"  end image:   {bundle['path_end']}")
    print(f"  prompt:      {bundle['prompt']!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CSV pair + prompt → optional DynamiCrafter 512 interp via official inference.py."
    )
    p.add_argument("--pairs_csv", type=Path, required=True)
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--pair_index", type=int, default=0)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory when using --run (default: Interpose/out_video_gen/<pair_tag>).",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="Stage prompt_pack and run DynamiCrafter inference.py (GPU).",
    )
    p.add_argument("--seed", type=int, default=12306)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument(
        "--dump-pre-model",
        action="store_true",
        help="Only write pre_model_inputs/ (prompt + resized PNGs); no GPU inference. Uses --out_dir or default tag.",
    )
    return p.parse_args()


def default_out_dir(bundle: dict[str, Any], override: Path | None) -> Path:
    tag = f"pair_{bundle['pair_index']:05d}_{bundle['scene'].replace('/', '_')}"
    return (override or (repository_root() / "Interpose" / "out_video_gen" / tag)).resolve()


def main() -> None:
    args = parse_args()
    try:
        bundle = resolve_inputs(args.pairs_csv, args.data_root, args.pair_index, args.prompt)
    except (OSError, ValueError, IndexError) as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    print_bundle_summary(bundle)

    if args.dump_pre_model and not args.run:
        out_dir = default_out_dir(bundle, args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inf_args = build_inference_args_512_interp(
            out_dir,
            out_dir / "prompt_pack",
            seed=args.seed,
            ddim_steps=args.ddim_steps,
            frame_stride=args.frame_stride,
            cfg_scale=args.cfg_scale,
        )
        save_pre_model_inputs(
            out_dir / "pre_model_inputs",
            bundle["path_start"],
            bundle["path_end"],
            bundle["prompt"],
            height=int(inf_args.height),
            width=int(inf_args.width),
        )
        print(f"\nWrote pre_model_inputs only: {out_dir / 'pre_model_inputs'}")
        return

    if not args.run:
        print("\nDry run only. Pass --run to stage prompt_pack and call inference.py.")
        print("Pass --dump-pre-model to save pre_model_inputs/ without GPU.")
        return

    out_dir = default_out_dir(bundle, args.out_dir)
    try:
        run_dynamiCrafter_interp_inference(
            bundle,
            out_dir,
            seed=args.seed,
            ddim_steps=args.ddim_steps,
            frame_stride=args.frame_stride,
            cfg_scale=args.cfg_scale,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[error] inference failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Outputs: {out_dir / 'samples_separate'}")
    print(f"Pre-model dump:  {out_dir / 'pre_model_inputs'}")


if __name__ == "__main__":
    main()
