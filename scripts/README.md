# DL3DV Evaluation Scripts

This folder contains the DUSt3R evaluation scripts for DL3DV benchmark pairs.

## Scripts

- `scripts/DL3DV/eval_dust3r_dl3dv.py`
  - Selects DL3DV pairs by geometry filters.
  - Runs DUSt3R on selected pairs.
  - Reports relative-pose metrics (rotation and translation-direction errors).
  - Saves selected pairs and per-pair metrics as CSV.

- `scripts/DL3DV/visualize_dust3r_dl3dv.py`
  - Loads a pair CSV (or metrics CSV with `rel_a/rel_b` columns).
  - Runs DUSt3R per pair.
  - Saves visualization images (input pair, camera poses, point cloud).

## 1) Run evaluation

From repo root (`/home/caesar/camera_pose`):

```bash
python scripts/DL3DV/eval_dust3r_dl3dv.py \
  --dl3dv_root /home/caesar/camera_pose/DL3DV-Benchmark/DL3DV-10K-Benchmark \
  --checkpoint /home/caesar/camera_pose/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --dust3r_root /home/caesar/camera_pose/dust3r \
  --yaw_min 50 --yaw_max 90 \
  --min_forward_angle 15 \
  --min_baseline 0.5 \
  --max_overlap_ratio -1 \
  --max_pairs 300 \
  --seed 0 \
  --device cuda \
  --save_pairs_csv /home/caesar/camera_pose/outputs/dl3dv_pairs_relaxed.csv \
  --save_metrics_csv /home/caesar/camera_pose/outputs/dl3dv_metrics_relaxed.csv
```

Notes:
- `--image_size` defaults to `512` (DUSt3R input size).
- Use `--max_pairs 0` to evaluate all selected pairs.

Outputs:
- Console summary:
  - number of loaded frames/scenes
  - number of selected/evaluated pairs
  - aggregate metrics (rotation/translation direction accuracy and AUC-style scores)
- Optional pair list CSV (`--save_pairs_csv`), for example:
  - `outputs/dl3dv_pairs_relaxed.csv`
  - contains selected pair metadata (pair indices, frame paths, geometric filters)
- Optional per-pair metrics CSV (`--save_metrics_csv`), for example:
  - `outputs/dl3dv_metrics_relaxed.csv`
  - contains per-pair errors such as `rot_err_deg` and `trans_dir_err_deg` with `rel_a/rel_b`

## 2) Visualize results

Use the saved metrics CSV (it already contains `rel_a` and `rel_b`):

```bash
python scripts/DL3DV/visualize_dust3r_dl3dv.py \
  --dl3dv_root /home/caesar/camera_pose/DL3DV-Benchmark/DL3DV-10K-Benchmark \
  --checkpoint /home/caesar/camera_pose/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --dust3r_root /home/caesar/camera_pose/dust3r \
  --pairs_csv /home/caesar/camera_pose/outputs/dl3dv_metrics_relaxed.csv \
  --viz_dir /home/caesar/camera_pose/outputs/viz_dl3dv_relaxed \
  --num_pairs 20 \
  --image_size 512 \
  --device cuda
```

Outputs:
- One PNG per visualized pair in `--viz_dir`, for example:
  - `outputs/viz_dl3dv_relaxed/pair_000_<scene>_<frameA>_<frameB>.png`
- Each PNG includes:
  - input image pair
  - predicted + GT camera pose view
  - reconstructed point cloud view

## 3) (Optional) Download missing DL3DV benchmark scenes

If benchmark data is incomplete:

```bash
python DL3DV-Benchmark/download.py \
  --odir /home/caesar/camera_pose/DL3DV-Benchmark/DL3DV-10K-Benchmark \
  --subset full \
  --only_level4 \
  --format nerfstudio \
  --hf-repo-id DL3DV/DL3DV-10K-Benchmark
```

This downloader supports resume behavior and skips files already present locally.

Outputs:
- Downloaded benchmark tree under:
  - `/home/caesar/camera_pose/DL3DV-Benchmark/DL3DV-10K-Benchmark`
- Scene folders named by hash plus cache metadata under `.cache`.
