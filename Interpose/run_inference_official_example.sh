#!/usr/bin/env bash
# Full inference.py --interp command (matches scripts/run_application.sh).
# Set PROMPT_DIR to your staged folder (must contain prompt.txt + 00_start.png + 01_end.png).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROMPT_DIR="${PROMPT_DIR:-$ROOT/Interpose/out_inference/pair_00000_f71ac346cd0f/prompt_pack}"
SAVEDIR="${SAVEDIR:-$ROOT/Interpose/out_inference/pair_00000_f71ac346cd0f_from_shell}"

if [[ ! -d "$PROMPT_DIR" ]]; then
  echo "Missing prompt_dir: $PROMPT_DIR" >&2
  echo "Run: python3 Interpose/run_inference_interp_pair.py ... --dry_run" >&2
  exit 1
fi

cd "$ROOT/DynamiCrafter"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 scripts/evaluation/inference.py \
  --seed 12306 \
  --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt \
  --config configs/inference_512_v1.0.yaml \
  --savedir "$SAVEDIR" \
  --n_samples 1 \
  --bs 1 --height 320 --width 512 \
  --unconditional_guidance_scale 7.5 \
  --ddim_steps 50 \
  --ddim_eta 1.0 \
  --prompt_dir "$PROMPT_DIR" \
  --text_input \
  --video_length 16 \
  --frame_stride 5 \
  --timestep_spacing uniform_trailing \
  --guidance_rescale 0.7 \
  --perframe_ae \
  --interp

echo "Outputs: $SAVEDIR/samples_separate/"
