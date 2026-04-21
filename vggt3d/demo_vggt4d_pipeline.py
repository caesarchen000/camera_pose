#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VGGT4D_ROOT = _REPO_ROOT / "VGGT4D"
if str(_VGGT4D_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT4D_ROOT))

from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt4d.masks.dynamic_mask import (  # noqa: E402
    adaptive_multiotsu_variance,
    cluster_attention_maps,
    extract_dyn_map,
)
from vggt4d.masks.refine_dyn_mask import RefineDynMask  # noqa: E402
from vggt4d.models.vggt4d import VGGTFor4D  # noqa: E402
from vggt4d.utils.model_utils import inference, organize_qk_dict  # noqa: E402
from vggt4d.utils.store import (  # noqa: E402
    save_depth,
    save_depth_conf,
    save_dynamic_masks,
    save_intrinsic_txt,
    save_rgb,
    save_tum_poses,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Official VGGT4D 3-stage dynamic-mask pipeline from vggt3d."
    )
    p.add_argument("--input_dir", type=Path, required=True, help="Directory of scene folders with .jpg/.png frames.")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory for predictions and masks.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_VGGT4D_ROOT / "ckpts" / "model_tracker_fixed_e20.pt",
        help="Path to VGGT4D checkpoint (.pt).",
    )
    p.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu.")
    return p.parse_args()


def _load_model(checkpoint: Path, device: torch.device) -> VGGTFor4D:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Download it with:\n"
            f"  mkdir -p {_VGGT4D_ROOT / 'ckpts'}\n"
            f"  wget -c \"https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt?download=true\" "
            f"-O \"{_VGGT4D_ROOT / 'ckpts' / 'model_tracker_fixed_e20.pt'}\""
        )
    model = VGGTFor4D()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    model.eval()
    model = model.to(device)
    return model


def process_scene(scene_dir: Path, output_dir: Path, model: VGGTFor4D, device: torch.device) -> None:
    image_paths = sorted(list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.png")))
    if not image_paths:
        print(f"[warn] No images found in {scene_dir}, skipping.")
        return

    print(f"[scene] {scene_dir.name} ({len(image_paths)} frames)")
    images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)
    n_img, _, h_img, w_img = images.shape
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: dynamic cue extraction from attention
    print("  Stage 1: predict depth and coarse dynamic mask")
    predictions1, qk_dict, enc_feat, agg_tokens_list = inference(model, images)
    del agg_tokens_list
    qk_dict = organize_qk_dict(qk_dict, n_img)
    dyn_maps = extract_dyn_map(qk_dict, images)

    h_tok, w_tok = h_img // 14, w_img // 14
    feat_map = rearrange(enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)
    norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)
    upsampled_map = F.interpolate(
        rearrange(norm_dyn_map, "n_img h w -> n_img 1 h w"),
        size=(h_img, w_img),
        mode="bilinear",
        align_corners=False,
    )
    upsampled_map = rearrange(upsampled_map, "n_img 1 h w -> n_img h w")
    thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
    dyn_masks = upsampled_map > thres

    # Stage 2: rerun with early-stage masking
    print("  Stage 2: refine extrinsics with dynamic mask")
    del enc_feat, feat_map
    torch.cuda.empty_cache()
    predictions2, _, _, _ = inference(model, images, dyn_masks.to(device))

    final_prediction = {**predictions1}
    final_prediction["extrinsic"] = predictions2["extrinsic"]
    final_prediction["cam2world"] = predictions2["cam2world"]

    # Stage 3: mask refinement (SOR + clustering + reprojection/photometric losses)
    print("  Stage 3: refine dynamic mask")
    torch.cuda.empty_cache()
    pred_intrinsic = final_prediction["intrinsic"]
    pred_cam2world = final_prediction["cam2world"]
    pred_depths = final_prediction["depth"]
    pred_conf = final_prediction["depth_conf"]

    refiner = RefineDynMask(
        images,
        torch.tensor(pred_depths).to(device),
        dyn_masks.to(device),
        torch.tensor(pred_cam2world).float().to(device),
        torch.tensor(pred_intrinsic).to(device),
        device,
    )
    refined_mask = refiner.refine_masks()
    del refiner

    save_intrinsic_txt(output_dir, pred_intrinsic)
    save_rgb(output_dir, images)
    save_depth(output_dir, pred_depths)
    save_depth_conf(output_dir, pred_conf)
    save_tum_poses(output_dir, final_prediction["cam2world"])
    save_dynamic_masks(output_dir, refined_mask)
    print(f"  Saved outputs -> {output_dir}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint.resolve(), device)

    scene_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found in {input_dir}")
    print(f"[info] scenes={len(scene_dirs)} device={device}")
    for scene_dir in scene_dirs:
        process_scene(scene_dir, output_dir / scene_dir.name, model, device)
    print(f"[done] Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()

