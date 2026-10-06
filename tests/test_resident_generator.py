"""The strict clip runner must keep one worker and refuse silent fallback."""

import sys
from pathlib import Path

import pytest

from models import PromptBundle
from video_generator import LocalVideoGenerator, VideoGenerationError, VideoGenerationRequest


def worker_script(path: Path, handshake: str) -> None:
    path.write_text(
        "import json, pathlib, sys\n"
        "assert '--daemon' in sys.argv\n"
        "pathlib.Path(sys.argv[1]).open('a').write('start\\n')\n"
        f"print({handshake!r}, flush=True)\n"
        "for line in sys.stdin:\n"
        "    task = json.loads(line)\n"
        "    if task.get('action') == 'exit': break\n"
        "    pathlib.Path(task['output_video']).write_bytes(b'clip')\n"
        "    print(json.dumps({'status': 'success'}), flush=True)\n",
        encoding="utf-8",
    )


def request(image: Path, destination: Path, name: str) -> VideoGenerationRequest:
    return VideoGenerationRequest(
        image, destination, PromptBundle(motion_prompt="one small motion"), name,
    )


def test_resident_worker_handles_two_clips_in_one_process(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    launch_log = tmp_path / "launch.log"
    script = tmp_path / "worker.py"
    worker_script(script, "READY resident")
    generator = LocalVideoGenerator(
        executable=sys.executable,
        script_path=str(script),
        base_arguments=[str(launch_log)],
        require_resident_daemon=True,
    )
    try:
        generator.generate_clip_video(request(image, tmp_path / "one.mp4", "one"))
        generator.generate_clip_video(request(image, tmp_path / "two.mp4", "two"))
    finally:
        generator.stop_daemon()
    assert launch_log.read_text(encoding="utf-8").splitlines() == ["start"]


def test_strict_mode_rejects_nonresident_worker(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    script = tmp_path / "worker.py"
    worker_script(script, "READY")
    generator = LocalVideoGenerator(
        executable=sys.executable,
        script_path=str(script),
        base_arguments=[str(tmp_path / "launch.log")],
        require_resident_daemon=True,
    )
    with pytest.raises(VideoGenerationError, match="Resident generator"):
        generator.generate_clip_video(request(image, tmp_path / "clip.mp4", "clip"))
    assert not (tmp_path / "clip.mp4").exists()
