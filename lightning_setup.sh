#!/usr/bin/env bash
set -euo pipefail

# Lightning AI bootstrap script for the first Wan2.2 5B test.
#
# Edit the variables below before running:
#   LIGHTNING_WORKDIR - local working directory on the machine
#   WAN_REPO_DIR      - path where Wan2.2 will be cloned
#   MODEL_DIR         - path where Wan checkpoints will be downloaded
#
# This script is intentionally conservative:
#   - it uses the 5B preset first
#   - it does not assume prompt extension
#   - it does not require audio or other modules

LIGHTNING_WORKDIR="${LIGHTNING_WORKDIR:-$HOME/work}"
WAN_REPO_DIR="${WAN_REPO_DIR:-$LIGHTNING_WORKDIR/Wan2.2}"
MODEL_DIR="${MODEL_DIR:-$LIGHTNING_WORKDIR/models}"

mkdir -p "$LIGHTNING_WORKDIR" "$MODEL_DIR"

if [ ! -d "$WAN_REPO_DIR/.git" ]; then
  git clone https://github.com/Wan-Video/Wan2.2.git "$WAN_REPO_DIR"
fi

cd "$WAN_REPO_DIR"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -d "$MODEL_DIR/Wan2.2-TI2V-5B" ]; then
  python -m pip install "huggingface_hub[cli]"
  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$MODEL_DIR/Wan2.2-TI2V-5B"
fi

echo
echo "Setup complete. Next step: place your input image at:"
echo "$LIGHTNING_WORKDIR/image_to_video_pipeline/outputs/newton_s_cradle/input/input_image.png"
echo
echo "To run the first test from this workspace, use:"
echo "/usr/bin/python3 $LIGHTNING_WORKDIR/image_to_video_pipeline/main.py \
  --knowledge-file $LIGHTNING_WORKDIR/image_to_video_pipeline/knowledge/newtons_cradle.json \
  --generator-script $LIGHTNING_WORKDIR/image_to_video_pipeline/wan_local_wrapper.py \
  --wan-repo-dir $WAN_REPO_DIR \
  --wan-model-preset ti2v-5b"