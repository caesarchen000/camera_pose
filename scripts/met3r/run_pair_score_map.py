import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MET3R_REPO = REPO_ROOT / "MEt3R"
if str(MET3R_REPO) not in sys.path:
    sys.path.insert(0, str(MET3R_REPO))

from met3r import MEt3R


def load_rgb(path: str, image_size: int | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def save_score_map(score_map: torch.Tensor, out_dir: str, prefix: str = "met3r") -> None:
    score = score_map[0].detach().cpu().numpy()
    np.save(os.path.join(out_dir, f"{prefix}_score_map.npy"), score)

    smin = float(score.min())
    smax = float(score.max())
    denom = max(smax - smin, 1e-8)
    norm = (score - smin) / denom
    img = (norm * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(os.path.join(out_dir, f"{prefix}_score_map.png"))


def main():
    parser = argparse.ArgumentParser(description="Run MEt3R on a single image pair.")
    parser.add_argument("--img1", type=str, required=True, help="First image path")
    parser.add_argument("--img2", type=str, required=True, help="Second image path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--img_size", type=int, default=256, help="Input size for MEt3R")
    parser.add_argument(
        "--backbone",
        type=str,
        default="mast3r",
        choices=["mast3r", "dust3r", "raft"],
        help="Warping backbone for MEt3R",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="cosine",
        choices=["cosine", "lpips", "rmse", "psnr", "mse", "ssim"],
        help="Distance function used by MEt3R",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metric = MEt3R(
        img_size=args.img_size,
        backbone=args.backbone,
        distance=args.distance,
        freeze=True,
    ).to(device)
    metric.eval()

    img1 = load_rgb(args.img1, args.img_size)
    img2 = load_rgb(args.img2, args.img_size)
    batch = torch.stack([img1, img2], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        score, overlap_mask, score_map = metric(
            images=batch,
            return_overlap_mask=True,
            return_score_map=True,
            return_projections=False,
        )

    np.save(os.path.join(args.output_dir, "met3r_score.npy"), score.detach().cpu().numpy())
    np.save(
        os.path.join(args.output_dir, "met3r_overlap_mask.npy"),
        overlap_mask[0].detach().cpu().numpy(),
    )
    save_score_map(score_map, args.output_dir, prefix="met3r")

    print("Saved outputs in:", args.output_dir)
    print("-", os.path.join(args.output_dir, "met3r_score.npy"))
    print("-", os.path.join(args.output_dir, "met3r_overlap_mask.npy"))
    print("-", os.path.join(args.output_dir, "met3r_score_map.npy"))
    print("-", os.path.join(args.output_dir, "met3r_score_map.png"))


if __name__ == "__main__":
    main()
