#!/usr/bin/env bash
set -euo pipefail

# Minimal runner for the real Wan 2.2 pipeline.
#
# Defaults:
#   WAN_REPO_DIR=/teamspace/studios/this_studio/Wan2.2
#   WAN_PRESET=ti2v-5b
#
# For better motion quality on chemistry / physics clips, try:
#   WAN_PRESET=i2v-a14b
#   (requires a single GPU with 80GB VRAM per Wan docs)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${PIPELINE_WORKDIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WAN_REPO_DIR="${WAN_REPO_DIR:-$WORKDIR/Wan2.2}"
WAN_PRESET="${WAN_PRESET:-ti2v-5b}"

cd "$WORKDIR"

if [ ! -f "$WAN_REPO_DIR/generate.py" ]; then
  echo "Wan repo not found at: $WAN_REPO_DIR" >&2
  exit 1
fi

if [ ! -d "$WAN_REPO_DIR/Wan2.2-TI2V-5B" ] && [ "$WAN_PRESET" = "ti2v-5b" ]; then
  echo "TI2V-5B model dir not found at: $WAN_REPO_DIR/Wan2.2-TI2V-5B" >&2
  exit 1
fi

if [ ! -d "$WAN_REPO_DIR/Wan2.2-I2V-A14B" ] && [ "$WAN_PRESET" = "i2v-a14b" ]; then
  echo "I2V-A14B model dir not found at: $WAN_REPO_DIR/Wan2.2-I2V-A14B" >&2
  exit 1
fi

python3 image_to_video_pipeline/main.py \
  --knowledge-file image_to_video_pipeline/knowledge/newtons_cradle.json \
  --generator-script image_to_video_pipeline/wan_local_wrapper.py \
  --wan-repo-dir "$WAN_REPO_DIR" \
  --wan-model-preset "$WAN_PRESET"
