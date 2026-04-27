#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VGGT4D_ROOT = _REPO_ROOT / "VGGT4D"
if str(_VGGT4D_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT4D_ROOT))

from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt4d.models.vggt4d import VGGTFor4D  # noqa: E402
from vggt4d.utils.model_utils import inference  # noqa: E402
from vggt4d.utils.store import save_intrinsic_txt, save_tum_poses  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run VGGT4D inference with provided dynamic masks and export camera poses."
    )
    p.add_argument("--video_path", type=Path, required=True, help="Input video file.")
    p.add_argument(
        "--mask_dir",
        type=Path,
        required=True,
        help="Directory containing dynamic_mask_XXXX.png files.",
    )
    p.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_VGGT4D_ROOT / "ckpts" / "model_tracker_fixed_e20.pt",
        help="Path to VGGT4D checkpoint (.pt).",
    )
    p.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu.")
    return p.parse_args()


def _extract_video_frames_to_temp(video_path: Path, temp_dir: Path) -> list[Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    frames, _, _ = read_video(str(video_path), pts_unit="sec")  # [T,H,W,C] uint8
    if frames.numel() == 0:
        raise ValueError(f"No frames read from video: {video_path}")
    frame_paths: list[Path] = []
    for i in range(frames.shape[0]):
        frame_path = temp_dir / f"{i:04d}.png"
        bgr = cv2.cvtColor(frames[i].cpu().numpy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), bgr)
        frame_paths.append(frame_path)
    return frame_paths


def _load_binary_masks(mask_dir: Path, n_frames: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    mask_paths = sorted(mask_dir.glob("dynamic_mask_*.png"))
    if len(mask_paths) != n_frames:
        raise ValueError(
            f"Mask/frame count mismatch: masks={len(mask_paths)} frames={n_frames}. "
            f"Expected one dynamic_mask_XXXX.png per frame."
        )
    masks = []
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Failed to read mask: {p}")
        m = (m > 0).astype(np.float32)
        m_t = torch.from_numpy(m)[None, None]  # [1,1,H,W]
        m_t = F.interpolate(m_t, size=(h, w), mode="nearest")[0, 0]  # [H,W]
        masks.append(m_t)
    dyn_masks = torch.stack(masks, dim=0).to(device) > 0.5  # [S,H,W] bool
    # Do not mask endpoints (first/last frame).
    if dyn_masks.shape[0] >= 1:
        dyn_masks[0] = False
    if dyn_masks.shape[0] >= 2:
        dyn_masks[-1] = False
    return dyn_masks


def _load_model(checkpoint: Path, device: torch.device) -> VGGTFor4D:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = VGGTFor4D()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    model.eval()
    return model.to(device)


def main() -> None:
    args = parse_args()
    video_path = args.video_path.resolve()
    mask_dir = args.mask_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint.resolve(), device)

    tmp_frame_dir = out_dir / "_tmp_frames"
    frame_paths = _extract_video_frames_to_temp(video_path, tmp_frame_dir)
    images = load_and_preprocess_images([str(p) for p in frame_paths]).to(device)  # [S,3,H,W]
    n_frames, _, h_img, w_img = images.shape
    if args.no_mask:
        dyn_masks = torch.zeros((n_frames, h_img, w_img), dtype=torch.bool, device=device)
    else:
        dyn_masks = _load_binary_masks(mask_dir, n_frames, h_img, w_img, device)

    print(f"[info] Running masked inference: frames={n_frames}, size=({h_img},{w_img}), device={device}")
    predictions, _, _, _ = inference(model, images, dyn_masks=dyn_masks)

    save_intrinsic_txt(out_dir, predictions["intrinsic"])
    save_tum_poses(out_dir, predictions["cam2world"])
    np.save(out_dir / "cam2world.npy", predictions["cam2world"])
    np.save(out_dir / "extrinsic.npy", predictions["extrinsic"])
    # Save explicit R|t per frame as text: one line = 12 values (row-major 3x4).
    extr = np.asarray(predictions["extrinsic"], dtype=np.float64)
    extr_rt = extr[:, :3, :4].reshape(extr.shape[0], 12)
    np.savetxt(out_dir / "extrinsic_rt.txt", extr_rt, fmt="%.8f")
    print(f"[done] Saved poses to {out_dir / 'pred_traj.txt'}")


if __name__ == "__main__":
    main()

