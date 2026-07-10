#!/usr/bin/env python3
"""Simple local generator for testing the pipeline without Wan/GPU.

Creates a static or lightly scaled video from an input image using ffmpeg.
Accepts the same basic CLI args as the Wan wrapper so it can be swapped in.
"""

from __future__ import annotations

import argparse
import subprocess
try:
    from imageio_ffmpeg import get_ffmpeg_exe
except Exception:
    get_ffmpeg_exe = None
from pathlib import Path
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dummy local image->video generator")
    p.add_argument("--input-image", required=True)
    p.add_argument("--output-video", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--clip-duration", default="1.0")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--clip-name", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    # allow passthrough of wrapper args (e.g. --wan-repo-dir, --model-preset)
    args, _unknown = parser.parse_known_args(argv)

    inp = Path(args.input_image)
    out = Path(args.output_video)
    duration = float(args.clip_duration)

    if not inp.exists():
        print(f"Input image not found: {inp}", file=sys.stderr)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg to create a static video of the requested duration
    ffmpeg_exe = get_ffmpeg_exe() if get_ffmpeg_exe is not None else "ffmpeg"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-loop",
        "1",
        "-i",
        str(inp),
        "-c:v",
        "libx264",
        "-t",
        str(duration),
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=1280:704",
        "-r",
        "25",
        str(out),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:", e.stderr, file=sys.stderr)
        return 3

    print(f"Dummy generation complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
