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
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input_dir",
        type=Path,
        help="Directory of scene folders with .jpg/.png frames (legacy mode).",
    )
    src.add_argument(
        "--subset_root",
        type=Path,
        help="Root directory to recursively find subset_* folders and process each subset directly.",
    )
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory for predictions and masks.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_VGGT4D_ROOT / "ckpts" / "model_tracker_fixed_e20.pt",
        help="Path to VGGT4D checkpoint (.pt).",
    )
    p.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu.")
    p.add_argument(
        "--mask_threshold",
        type=float,
        default=None,
        help=(
            "Optional manual threshold for dynamic score map binarization. "
            "If omitted, use adaptive_multiotsu_variance."
        ),
    )
    p.add_argument(
        "--allow_endpoint_mask",
        action="store_true",
        help="If set, do not force first/last frame masks to False.",
    )
    return p.parse_args()


def _discover_subset_dirs(subset_root: Path) -> list[Path]:
    dirs = sorted([p for p in subset_root.glob("**/subset_*") if p.is_dir()])
    if dirs:
        return dirs
    # Also support passing a single subset_* directory directly.
    if subset_root.is_dir() and subset_root.name.startswith("subset_"):
        has_images = bool(list(subset_root.glob("*.png")) or list(subset_root.glob("*.jpg")))
        if has_images:
            return [subset_root]
    return []


def _infer_subset_rel_base(subset_root: Path, scene_dirs: list[Path]) -> Path:
    # For a single subset dir ".../pair_xxx/subset/<variant>/subset_yy", use base
    # at ".../out_root" so output names stay stable as
    # "pair_xxx__subset__<variant>__subset_yy".
    if len(scene_dirs) == 1 and subset_root.name.startswith("subset_"):
        p = subset_root
        if "subset" in p.parts:
            i = p.parts.index("subset")
            # subset_root: .../<pair_dir>/subset/<variant>/subset_xx
            # We want rel path to include <pair_dir>/subset/<variant>/subset_xx,
            # so base should be parent of <pair_dir>.
            if i >= 1:
                return Path(*p.parts[: i - 1])
            return Path(*p.parts[:i])
    return subset_root


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


def _clear_endpoint_masks(dyn_masks: torch.Tensor) -> torch.Tensor:
    # Keep endpoints fully visible to anchor pose estimation to real inputs.
    if dyn_masks.shape[0] >= 1:
        dyn_masks[0] = False
    if dyn_masks.shape[0] >= 2:
        dyn_masks[-1] = False
    return dyn_masks


def process_scene(
    scene_dir: Path,
    output_dir: Path,
    model: VGGTFor4D,
    device: torch.device,
    mask_threshold: float | None = None,
    clear_endpoints: bool = True,
) -> None:
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
    if mask_threshold is None:
        thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
        print(f"  Mask threshold (adaptive): {float(thres):.6f}")
    else:
        thres = float(mask_threshold)
        print(f"  Mask threshold (manual): {float(thres):.6f}")
    dyn_masks = upsampled_map > thres
    if clear_endpoints:
        dyn_masks = _clear_endpoint_masks(dyn_masks)

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
    input_dir = args.input_dir.resolve() if args.input_dir is not None else None
    subset_root = args.subset_root.resolve() if args.subset_root is not None else None
    output_dir = args.output_dir.resolve()
    if input_dir is not None and not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if subset_root is not None and not subset_root.is_dir():
        raise FileNotFoundError(f"subset_root not found: {subset_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint.resolve(), device)

    if subset_root is not None:
        scene_dirs = _discover_subset_dirs(subset_root)
        rel_base = _infer_subset_rel_base(subset_root, scene_dirs)
    else:
        assert input_dir is not None
        scene_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        rel_base = input_dir
    if not scene_dirs:
        raise RuntimeError("No scene directories found for the provided input source.")
    print(f"[info] scenes={len(scene_dirs)} device={device}")
    for scene_dir in scene_dirs:
        rel_name = scene_dir.relative_to(rel_base).as_posix().replace("/", "__")
        process_scene(
            scene_dir,
            output_dir / rel_name,
            model,
            device,
            mask_threshold=args.mask_threshold,
            clear_endpoints=not args.allow_endpoint_mask,
        )
    print(f"[done] Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()

