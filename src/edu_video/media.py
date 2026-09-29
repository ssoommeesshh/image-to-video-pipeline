"""CPU FFmpeg helpers. Decode every frame when checking a generated artifact."""
import subprocess
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe, read_frames
from PIL import Image


def ffmpeg(*args):
    result = subprocess.run([get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                            capture_output=True, text=True, errors="replace", timeout=180)
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-4000:]}")


def inspect_video(path, last_frame=None):
    reader = read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        count, last = 0, None
        for frame in reader:
            count += 1
            last = frame
        if count == 0 or metadata["fps"] <= 0:
            raise ValueError(f"Video has no decodable frames: {path}")
        if last_frame:
            Image.frombytes("RGB", metadata["size"], last).save(last_frame)
        return {"frame_count": count, "fps": metadata["fps"], "size": list(metadata["size"]),
                "duration_seconds": count / metadata["fps"]}
    finally:
        reader.close()


def stitch(paths, destination):
    # Paths are generated internally. Escape quotes for FFmpeg's concat parser.
    listing = Path(destination).with_suffix(".concat.txt")
    listing.write_text("\n".join("file '" + str(Path(p).resolve().as_posix()).replace("'", "'\\''") + "'" for p in paths) + "\n", encoding="utf-8")
    try:
        ffmpeg("-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", destination)
    finally:
        listing.unlink(missing_ok=True)
