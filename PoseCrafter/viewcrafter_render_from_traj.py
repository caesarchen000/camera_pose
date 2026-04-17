#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "PoseCrafter Step 4: render/generate 25-frame videos in ViewCrafter using "
            "interpolated camera trajectories from Step 3."
        )
    )
    p.add_argument(
        "--traj_manifest",
        type=Path,
        default=Path("PoseCrafter/outputs/interpolated_camera_poses.jsonl"),
        help="Step-3 interpolated trajectory manifest.",
    )
    p.add_argument(
        "--posecrafter_out_root",
        type=Path,
        default=Path("PoseCrafter/outputs"),
        help="PoseCrafter output root that contains <pair>/four_frames.",
    )
    p.add_argument("--viewcrafter_root", type=Path, default=Path("ViewCrafter"))
    p.add_argument(
        "--viewcrafter_ckpt",
        type=Path,
        default=Path("ViewCrafter/checkpoints/model_sparse.ckpt"),
        help="ViewCrafter diffusion checkpoint (.ckpt).",
    )
    p.add_argument(
        "--dust3r_ckpt",
        type=Path,
        default=Path("ViewCrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
        help="DUSt3R checkpoint path used by ViewCrafter internals.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("ViewCrafter/configs/inference_pvd_1024.yaml"),
        help="ViewCrafter inference config yaml.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--multi_gpu_devices",
        type=str,
        default="",
        help=(
            "Optional comma-separated CUDA device ids for DataParallel on diffusion UNet, "
            'e.g. "0,1". Leave empty to use single GPU.'
        ),
    )
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--frame_stride", type=int, default=10)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--prompt", type=str, default="Rotating view of a scene")
    p.add_argument(
        "--pair_index",
        type=int,
        default=None,
        help="Optional pair index filter (pair_XXXXX_*).",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=Path("PoseCrafter/outputs/viewcrafter"),
        help="Output root for step-4 render/diffusion videos.",
    )
    return p.parse_args()


def pair_idx_from_pair_dir_name(pair_dir_name: str) -> int:
    # pair_00000_xxxxx
    return int(pair_dir_name.split("_")[1])


def stage_numeric_viewcrafter_inputs(src_four_frames_dir: Path, dst_dir: Path) -> Path:
    """
    ViewCrafter's load_initial_dir sorts with int(stem), so we must provide
    numeric filenames like 0.png, 1.png, 2.png, 3.png.
    """
    imgs = sorted(src_four_frames_dir.glob("*.png"))
    if len(imgs) < 4:
        raise FileNotFoundError(f"Need 4 frame PNGs in {src_four_frames_dir}, found {len(imgs)}")

    def key_fn(p: Path) -> int:
        # e.g. 00_start_f000.png, 01_f001.png, 02_f014.png, 03_end_f015.png
        token = p.stem.split("_")[0]
        return int(token)

    ordered = sorted(imgs, key=key_fn)[:4]
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(ordered):
        shutil.copy2(p, dst_dir / f"{i}.png")
    return dst_dir


def interpolate_seq(anchor_vals: torch.Tensor, num_out: int) -> torch.Tensor:
    # anchor_vals: [N, D], returns [num_out, D]
    n, d = anchor_vals.shape
    if n == num_out:
        return anchor_vals
    x_anchor = np.linspace(0.0, 1.0, n, dtype=np.float64)
    x_out = np.linspace(0.0, 1.0, num_out, dtype=np.float64)
    out = np.zeros((num_out, d), dtype=np.float64)
    src = anchor_vals.detach().float().cpu().numpy()
    for j in range(d):
        out[:, j] = np.interp(x_out, x_anchor, src[:, j])
    return torch.from_numpy(out).to(anchor_vals.device, dtype=anchor_vals.dtype)


def c2w_to_p3d_cameras(c2ws: torch.Tensor, H: int, W: int, focals: torch.Tensor, principal: torch.Tensor):
    from pytorch3d.renderer import PerspectiveCameras

    R = c2ws[:, :3, :3]
    T = c2ws[:, :3, 3:]
    # DUSt3R RDF -> PyTorch3D LUF
    R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], 2)
    new_c2w = torch.cat([R, T], 2)
    eye = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=c2ws.device, dtype=c2ws.dtype).repeat(c2ws.shape[0], 1, 1)
    w2c = torch.linalg.inv(torch.cat((new_c2w, eye), 1))
    R_new = w2c[:, :3, :3].permute(0, 2, 1)
    T_new = w2c[:, :3, 3]
    image_size = ((H, W),)
    return PerspectiveCameras(
        focal_length=focals,
        principal_point=principal,
        in_ndc=False,
        image_size=image_size,
        R=R_new,
        T=T_new,
        device=c2ws.device,
    )


def main() -> None:
    args = parse_args()
    traj_manifest = args.traj_manifest.resolve()
    posecrafter_out_root = args.posecrafter_out_root.resolve()
    viewcrafter_root = args.viewcrafter_root.resolve()
    viewcrafter_ckpt = args.viewcrafter_ckpt.resolve()
    dust3r_ckpt = args.dust3r_ckpt.resolve()
    config_path = args.config.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # ViewCrafter uses relative imports/paths (e.g. ./extern/dust3r).
    # Make execution robust no matter where this wrapper is launched from.
    os.chdir(viewcrafter_root)
    if str(viewcrafter_root) not in sys.path:
        sys.path.insert(0, str(viewcrafter_root))
    vc_dust3r = viewcrafter_root / "extern" / "dust3r"
    if str(vc_dust3r) not in sys.path:
        sys.path.insert(0, str(vc_dust3r))

    # Import from ViewCrafter codebase.
    from configs.infer_config import get_parser  # pylint: disable=import-outside-toplevel
    try:
        from utils.pvd_utils import save_video  # pylint: disable=import-outside-toplevel
        from viewcrafter import ViewCrafter  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as e:
        if "pytorch3d" in str(e):
            raise RuntimeError(
                "ViewCrafter step4 requires pytorch3d, but it is missing in the current environment. "
                "Install pytorch3d in this env (or run this script from a ViewCrafter-ready env) and retry."
            ) from e
        raise

    rows = [json.loads(l) for l in traj_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"No rows in trajectory manifest: {traj_manifest}")

    parser = get_parser()
    summary_rows = []

    for rec in rows:
        if rec.get("status") != "ok":
            continue
        pair_tag = Path(rec["pair_dir"]).name
        if args.pair_index is not None and pair_idx_from_pair_dir_name(pair_tag) != args.pair_index:
            continue
        variant = rec["variant"]
        frames_dir = posecrafter_out_root / pair_tag / "four_frames"
        if not frames_dir.is_dir():
            print(f"[skip] missing four_frames: {frames_dir}")
            continue

        exp_name = f"{pair_tag}_{variant}"
        save_dir = out_root / exp_name
        save_dir.mkdir(parents=True, exist_ok=True)
        numeric_frames_dir = stage_numeric_viewcrafter_inputs(frames_dir, save_dir / "input_frames_numeric")

        # Build opts from ViewCrafter parser defaults, then override.
        opts = parser.parse_args([])
        opts.image_dir = str(numeric_frames_dir)
        opts.out_dir = str(out_root)
        opts.exp_name = exp_name
        opts.save_dir = str(save_dir)
        opts.mode = "sparse_view_interp"
        opts.device = args.device
        opts.ckpt_path = str(viewcrafter_ckpt)
        opts.model_path = str(dust3r_ckpt)
        opts.config = str(config_path)
        opts.ddim_steps = args.ddim_steps
        opts.frame_stride = args.frame_stride
        opts.unconditional_guidance_scale = args.cfg_scale
        opts.seed = args.seed
        opts.prompt = args.prompt
        opts.video_length = int(rec["num_output_poses"])

        print(f"[run] {exp_name}")
        pvd = ViewCrafter(opts)

        # Optional multi-GPU execution for diffusion UNet.
        # This keeps the pipeline identical while splitting UNet forward passes across GPUs.
        if args.multi_gpu_devices.strip():
            if not torch.cuda.is_available():
                raise RuntimeError("--multi_gpu_devices requested, but CUDA is not available.")
            device_ids = [int(x.strip()) for x in args.multi_gpu_devices.split(",") if x.strip()]
            if len(device_ids) < 2:
                raise ValueError("--multi_gpu_devices must contain at least 2 GPU ids, e.g. '0,1'.")
            primary_device = f"cuda:{device_ids[0]}"
            pvd.diffusion = pvd.diffusion.to(primary_device)
            pvd.diffusion.model.diffusion_model = torch.nn.DataParallel(
                pvd.diffusion.model.diffusion_model,
                device_ids=device_ids,
                output_device=device_ids[0],
            )
            pvd.device = primary_device
            opts.device = primary_device
            print(f"[multi-gpu] Enabled DataParallel on diffusion UNet with devices: {device_ids}")

        # Use our interpolated trajectory directly.
        c2ws = torch.tensor(rec["trajectory_c2w_poses"], dtype=torch.float32, device=opts.device)
        num_views = int(c2ws.shape[0])

        # Collect scene data from ViewCrafter's DUSt3R result on four selected frames.
        shape = pvd.images[0]["true_shape"]
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in pvd.scene.get_pts3d(clip_thred=opts.dpt_trd)]
        imgs = np.array(pvd.scene.imgs)
        masks = None

        # Interpolate intrinsics from anchor frames to full trajectory length.
        focals_anchor = pvd.scene.get_focals().detach()  # [4,2] usually
        principal_anchor = pvd.scene.get_principal_points().detach()  # [4,2]
        focals = interpolate_seq(focals_anchor, num_views)
        principal = interpolate_seq(principal_anchor, num_views)
        cameras = c2w_to_p3d_cameras(c2ws, H, W, focals, principal)

        render_results, _ = pvd.run_render(pcd, imgs, masks, H, W, cameras, num_views)
        render_results = F.interpolate(
            render_results.permute(0, 3, 1, 2), size=(576, 1024), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        render_results[0] = pvd.img_ori[0]
        render_results[-1] = pvd.img_ori[-1]

        save_video(render_results, str(save_dir / "render_from_traj.mp4"))
        diffusion_results = pvd.run_diffusion(render_results)
        save_video((diffusion_results + 1.0) / 2.0, str(save_dir / "diffusion_from_traj.mp4"))

        summary_rows.append(
            {
                "pair_dir": rec["pair_dir"],
                "variant": variant,
                "frames_dir": str(frames_dir),
                "numeric_frames_dir": str(numeric_frames_dir),
                "save_dir": str(save_dir),
                "num_views": num_views,
                "render_video": str(save_dir / "render_from_traj.mp4"),
                "diffusion_video": str(save_dir / "diffusion_from_traj.mp4"),
            }
        )

    summary_path = out_root / "viewcrafter_step4_manifest.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for r in summary_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote step4 manifest: {summary_path}")


if __name__ == "__main__":
    main()
