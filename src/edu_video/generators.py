"""CPU dummy and explicit local-command adapters using the existing request shape."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from .models import PromptBundle
from .media import ffmpeg


@dataclass
class VideoGenerationRequest:
    input_image_path: Path
    output_video_path: Path
    prompt_bundle: PromptBundle
    clip_name: str = ""


class DummyGenerator:
    identity = {"name": "dummy", "version": 1, "size": [320, 240], "fps": 12}

    def generate_clip_video(self, request):
        ffmpeg("-loop", "1", "-i", request.input_image_path, "-an", "-vf",
               "scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2",
               "-r", "12", "-t", request.prompt_bundle.clip_duration_seconds,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "1", request.output_video_path)
        return request.output_video_path


class LocalGenerator:
    def __init__(self, script, timeout=180):
        from .pipeline import file_hash
        self.script = Path(script).expanduser().resolve()
        self.timeout = timeout
        self.identity = {"name": "local", "script": str(self.script), "sha256": file_hash(self.script),
                         "python": sys.executable, "version": 1}

    def generate_clip_video(self, request):
        subprocess.run([sys.executable, str(self.script), "--input-image", str(request.input_image_path),
                        "--output-video", str(request.output_video_path), "--prompt", request.prompt_bundle.motion_prompt,
                        "--negative-prompt", request.prompt_bundle.negative_prompt,
                        "--clip-duration", str(request.prompt_bundle.clip_duration_seconds),
                        "--clip-name", request.clip_name], check=True, capture_output=True, text=True,
                       timeout=self.timeout)
        return request.output_video_path
