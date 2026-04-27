#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VGGT_ROOT = _REPO_ROOT / "vggt"
_VGGT4D_ROOT = _REPO_ROOT / "VGGT4D"
for _p in (str(_VGGT_ROOT), str(_VGGT4D_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from vggt4d.utils.store import save_intrinsic_txt, save_tum_poses  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run VGGT3D masked inference using VGGT4D-generated masks on early attention layers."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_path", type=Path, help="Input video path.")
    src.add_argument(
        "--subset_dir",
        type=Path,
        help="Directory containing subset frames (.png/.jpg), e.g. subset_00_random.",
    )
    p.add_argument(
        "--mask_dir",
        type=Path,
        default=None,
        help="Directory with dynamic_mask_XXXX.png masks (pixel-space).",
    )
    p.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    p.add_argument(
        "--mask_layers",
        type=int,
        default=5,
        help="Apply mask on attention layers [0, mask_layers).",
    )
    p.add_argument("--no_mask", action="store_true", help="Disable mask application.")
    p.add_argument(
        "--allow_endpoint_mask",
        action="store_true",
        help="If set, do not force first/last frame masks to False.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local VGGT checkpoint (.pt). If omitted, download official VGGT-1B weights.",
    )
    p.add_argument("--device", type=str, default="cuda", help="Torch device.")
    return p.parse_args()


def _extract_video_frames_to_temp(video_path: Path, temp_dir: Path) -> list[Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    frames, _, _ = read_video(str(video_path), pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"No frames read from video: {video_path}")
    frame_paths: list[Path] = []
    for i in range(frames.shape[0]):
        frame_path = temp_dir / f"{i:04d}.png"
        bgr = cv2.cvtColor(frames[i].cpu().numpy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), bgr)
        frame_paths.append(frame_path)
    return frame_paths


def _collect_subset_frames(subset_dir: Path) -> list[Path]:
    image_paths = sorted(list(subset_dir.glob("*.png")) + list(subset_dir.glob("*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No .png/.jpg frames found in subset_dir: {subset_dir}")
    return image_paths


def _load_binary_masks(mask_dir: Path, n_frames: int, clear_endpoints: bool = True) -> torch.Tensor:
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
        masks.append(m > 0)
    dyn_masks = np.stack(masks, axis=0)  # [S,H,W] bool
    if clear_endpoints:
        if dyn_masks.shape[0] >= 1:
            dyn_masks[0] = False
        if dyn_masks.shape[0] >= 2:
            dyn_masks[-1] = False
    return torch.from_numpy(dyn_masks)


def _to_patch_token_mask(pixel_mask: torch.Tensor, h_img: int, w_img: int, patch_size: int) -> torch.Tensor:
    # pixel_mask: [S,H,W] bool -> [S,HW_patch] bool
    s = pixel_mask.shape[0]
    mask_f = pixel_mask.float().unsqueeze(1)  # [S,1,H,W]
    hp = h_img // patch_size
    wp = w_img // patch_size
    mask_patch = F.interpolate(mask_f, size=(hp, wp), mode="nearest").squeeze(1) > 0.5  # [S,hp,wp]
    return mask_patch.reshape(s, hp * wp)


def _load_model(checkpoint: Path | None, device: torch.device) -> VGGT:
    model = VGGT()
    if checkpoint is None:
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def _cam2world_from_extrinsic(extrinsic: torch.Tensor) -> torch.Tensor:
    # extrinsic: [B,S,3,4] (camera-from-world) -> [B,S,4,4] world-from-camera
    b, s = extrinsic.shape[:2]
    eye = torch.eye(4, device=extrinsic.device, dtype=extrinsic.dtype).view(1, 1, 4, 4).repeat(b, s, 1, 1)
    eye[..., :3, :4] = extrinsic
    return torch.linalg.inv(eye)


def _install_attention_mask_hooks(model: VGGT, dyn_mask_tokens: torch.Tensor | None, mask_layers: int) -> None:
    # dyn_mask_tokens: [S, HW_patch] bool
    def _make_forward(attn_module, is_frame_attn: bool, layer_id: int):
        def _forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
            bsz, ntok, dim = x.shape
            qkv = self.qkv(x).reshape(bsz, ntok, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)

            use_mask = dyn_mask_tokens is not None and layer_id < mask_layers
            if use_mask:
                s_frames, hw_tokens = dyn_mask_tokens.shape
                pad = torch.zeros((s_frames, 5), dtype=torch.bool, device=x.device)
                frame_token_mask = torch.cat([pad, dyn_mask_tokens.to(x.device)], dim=1)  # [S, 5+HW]

                out = torch.empty_like(v)
                if is_frame_attn:
                    if bsz % s_frames != 0:
                        raise ValueError(f"Frame-attn batch {bsz} not divisible by num_frames {s_frames}")
                    full_mask = frame_token_mask.repeat(bsz // s_frames, 1)  # [B*S, N]
                else:
                    global_mask = frame_token_mask.reshape(-1)  # [S*(5+HW)]
                    if global_mask.numel() != ntok:
                        raise ValueError(
                            f"Global token mismatch: mask tokens={global_mask.numel()} model tokens={ntok}"
                        )
                    full_mask = global_mask.unsqueeze(0).repeat(bsz, 1)  # [B, N]

                for bi in range(bsz):
                    keep_idx = (~full_mask[bi]).nonzero(as_tuple=True)[0]
                    k_keep = k[bi : bi + 1, :, keep_idx, :].contiguous()
                    v_keep = v[bi : bi + 1, :, keep_idx, :].contiguous()
                    out[bi : bi + 1] = F.scaled_dot_product_attention(q[bi : bi + 1], k_keep, v_keep)
                x_out = out
            else:
                x_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

            x_out = x_out.transpose(1, 2).reshape(bsz, ntok, dim)
            x_out = self.proj(x_out)
            x_out = self.proj_drop(x_out)
            return x_out

        return types.MethodType(_forward, attn_module)

    for layer_id, blk in enumerate(model.aggregator.frame_blocks):
        blk.attn.forward = _make_forward(blk.attn, is_frame_attn=True, layer_id=layer_id)
    for layer_id, blk in enumerate(model.aggregator.global_blocks):
        blk.attn.forward = _make_forward(blk.attn, is_frame_attn=False, layer_id=layer_id)


def main() -> None:
    args = parse_args()
    video_path = args.video_path.resolve() if args.video_path is not None else None
    subset_dir = args.subset_dir.resolve() if args.subset_dir is not None else None
    mask_dir = args.mask_dir.resolve() if args.mask_dir is not None else None
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if video_path is not None and not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if subset_dir is not None and not subset_dir.is_dir():
        raise FileNotFoundError(f"subset_dir not found: {subset_dir}")
    if not args.no_mask and (mask_dir is None or not mask_dir.is_dir()):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint.resolve() if args.checkpoint is not None else None, device)

    if subset_dir is not None:
        frame_paths = _collect_subset_frames(subset_dir)
        source = f"subset_dir={subset_dir}"
    else:
        assert video_path is not None
        tmp_frame_dir = out_dir / "_tmp_frames"
        frame_paths = _extract_video_frames_to_temp(video_path, tmp_frame_dir)
        source = f"video_path={video_path}"
    images = load_and_preprocess_images([str(p) for p in frame_paths]).to(device)  # [S,3,H,W]
    n_frames, _, h_img, w_img = images.shape

    dyn_mask_tokens = None
    if not args.no_mask:
        assert mask_dir is not None
        pixel_masks = _load_binary_masks(
            mask_dir,
            n_frames,
            clear_endpoints=not args.allow_endpoint_mask,
        ).to(device)  # [S,H,W]
        dyn_mask_tokens = _to_patch_token_mask(
            pixel_masks, h_img=h_img, w_img=w_img, patch_size=model.aggregator.patch_size
        )  # [S,HW_patch]

    _install_attention_mask_hooks(model, dyn_mask_tokens=dyn_mask_tokens, mask_layers=args.mask_layers)

    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(
        f"[info] Running VGGT3D masked inference: frames={n_frames}, size=({h_img},{w_img}), "
        f"mask_layers={args.mask_layers}, no_mask={args.no_mask}, "
        f"allow_endpoint_mask={args.allow_endpoint_mask}, device={device}, source={source}"
    )
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=dtype):
        aggregated_tokens_list, _ = model.aggregator(images.unsqueeze(0))  # [B=1,S,3,H,W]
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        cam2world = _cam2world_from_extrinsic(extrinsic)

    extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()
    cam2world_np = cam2world.squeeze(0).detach().cpu().numpy()

    save_intrinsic_txt(out_dir, intrinsic_np)
    save_tum_poses(out_dir, cam2world_np)
    np.save(out_dir / "cam2world.npy", cam2world_np)
    np.save(out_dir / "extrinsic.npy", extrinsic_np)
    extr_rt = extrinsic_np[:, :3, :4].reshape(extrinsic_np.shape[0], 12)
    np.savetxt(out_dir / "extrinsic_rt.txt", extr_rt, fmt="%.8f")
    print(f"[done] Saved poses to {out_dir / 'pred_traj.txt'}")


if __name__ == "__main__":
    main()
