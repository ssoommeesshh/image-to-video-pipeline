#!/usr/bin/env python3
"""Generate the ammonium-chloride pilot one clip at a time on Kaggle.

Each child worker encodes only one motion/negative prompt pair. Completed
steps are reused on a restart; an interrupted step gets a new attempt folder
so the pipeline cannot silently reuse a truncated MP4.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE / "knowledge" / "ammonium_chloride_resident_pilot.json"
EXPECTED_CLIPS = [f"clip_{number}" for number in range(1, 6)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wan-repo-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/kaggle/tmp/wan22-int8-cache"))
    parser.add_argument("--reuse-first-pilot", type=Path,
                        help="Existing clip-1 experiment folder, including final_video.mp4")
    parser.add_argument("--dummy-generator", action="store_true",
                        help="Use static CPU clips to validate the orchestration locally")
    return parser


def _groups(data: dict) -> list[list[dict]]:
    groups: list[list[dict]] = []
    pending_cards: list[dict] = []
    for clip in data["clips"]:
        if clip.get("metadata", {}).get("is_title"):
            pending_cards.append(clip)
        else:
            groups.append([*pending_cards, clip])
            pending_cards = []
    if pending_cards:
        raise ValueError("Trailing title cards need a following generated clip")
    names = [group[-1]["name"] for group in groups]
    if names != EXPECTED_CLIPS:
        raise ValueError(f"Expected {EXPECTED_CLIPS}, got {names}")
    return groups


def _image_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else HERE / path


def _materialize(data: dict, group: list[dict], frames: dict[str, Path]) -> tuple[dict, Path]:
    group = copy.deepcopy(group)
    generated = group[-1]
    reference = generated.pop("reference_clip", None)
    if reference:
        if reference not in frames:
            raise ValueError(f"{generated['name']}: missing frame from {reference}")
        generated["input_frame_path"] = str(frames[reference].resolve())
    image = _image_path(generated.get("input_frame_path", ""))
    if not image.is_file():
        raise FileNotFoundError(f"{generated['name']}: missing starting image {image}")
    generated["continuity_required"] = False
    generated.setdefault("metadata", {})["new_scene"] = True
    mini = {key: copy.deepcopy(value) for key, value in data.items() if key != "clips"}
    mini["clips"] = group
    return mini, image


def _signature(mini: dict, image: Path) -> str:
    digest = hashlib.sha256(json.dumps(mini, sort_keys=True).encode("utf-8"))
    with image.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_video(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    return result.returncode == 0


def _finished(experiment_dir: Path, name: str, signature: str | None = None) -> bool:
    if signature is not None:
        marker = experiment_dir.parent / "signature.txt"
        if not marker.is_file() or marker.read_text().strip() != signature:
            return False
    frame = experiment_dir / name / "last_frame.png"
    if not frame.is_file() or frame.stat().st_size == 0:
        return False
    return _valid_video(experiment_dir / f"{name}.mp4") and _valid_video(experiment_dir / "final_video.mp4")


def _run_step(args: argparse.Namespace, mini: dict, image: Path, name: str) -> Path:
    step_root = args.output_dir / "steps" / name
    step_root.mkdir(parents=True, exist_ok=True)
    signature = _signature(mini, image)
    attempts = sorted(step_root.glob("attempt_*"))
    for attempt in attempts:
        completed = attempt / "segment"
        if _finished(completed, name, signature):
            print(f"Reusing completed {name}: {completed}", flush=True)
            return completed

    attempt = step_root / f"attempt_{len(attempts) + 1:02d}"
    attempt.mkdir()
    fixture = attempt / "experiment.json"
    fixture.write_text(json.dumps(mini, indent=2) + "\n", encoding="utf-8")
    generator = HERE / ("local_dummy_generator.py" if args.dummy_generator else "kaggle_wan_resident.py")
    command = [
        sys.executable, str(HERE / "main.py"),
        "--knowledge-file", str(fixture),
        "--output-dir", str(attempt),
        "--experiment-name", "segment",
        "--generator-script", str(generator),
        "--generator-executable", sys.executable,
        "--generator-working-dir", str(HERE),
        "--continuity-mode", "chain",
    ]
    if not args.dummy_generator:
        if args.wan_repo_dir is None:
            raise ValueError("--wan-repo-dir is required for Wan generation")
        command.extend([
            "--wan-repo-dir", str(args.wan_repo_dir.resolve()),
            "--wan-model-preset", "i2v-a14b",
            "--require-resident-daemon",
            "--generator-arg=--lightning",
            f"--generator-arg=--knowledge-file={fixture}",
            f"--generator-arg=--cache-dir={args.cache_dir.resolve()}",
            "--generator-arg=--gpu0-memory-gib=6",
            "--generator-arg=--gpu1-memory-gib=10",
            "--generator-arg=--memory-telemetry",
        ])
    print(f"Generating {name} in {attempt}", flush=True)
    subprocess.run(command, cwd=HERE, check=True)
    completed = attempt / "segment"
    if not _finished(completed, name):
        raise RuntimeError(f"{name} did not produce a valid clip, frame and segment video")
    (attempt / "signature.txt").write_text(signature + "\n", encoding="ascii")
    print(f"Saved {name}: {completed / 'final_video.mp4'}", flush=True)
    return completed


def main() -> None:
    args = _parser().parse_args()
    data = json.loads(args.fixture.read_text(encoding="utf-8"))
    groups = _groups(data)
    args.output_dir = args.output_dir.resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, Path] = {}
    segments: list[Path] = []
    for group in groups:
        name = group[-1]["name"]
        mini, image = _materialize(data, group, frames)
        if name == "clip_1" and args.reuse_first_pilot is not None:
            completed = args.reuse_first_pilot.resolve()
            if not _finished(completed, name):
                raise ValueError(f"First pilot is incomplete or invalid: {completed}")
            print(f"Reusing existing first pilot: {completed}", flush=True)
        else:
            completed = _run_step(args, mini, image, name)
        frames[name] = completed / name / "last_frame.png"
        segments.append(completed / "final_video.mp4")

    from stitcher import VideoStitcher
    final_video = args.output_dir / "final_video.mp4"
    VideoStitcher().stitch_clips(segments, final_video)
    if not _valid_video(final_video):
        raise RuntimeError(f"Stitched video is invalid: {final_video}")
    print(f"Completed ammonium-chloride video: {final_video}", flush=True)


if __name__ == "__main__":
    main()
