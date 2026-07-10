"""Generate clip videos locally from an input image and motion prompt.

This module intentionally does not embed Wan 2.2 specifics. Instead, it wraps a
configurable local command so the pipeline can run inside Lightning AI while the
underlying generation script remains swappable.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from models import PromptBundle


class VideoGenerationError(RuntimeError):
	"""Raised when a local video generation command fails."""


@dataclass(slots=True)
class VideoGenerationRequest:
	"""Inputs needed to generate a single clip video."""

	input_image_path: str | Path
	output_video_path: str | Path
	prompt_bundle: PromptBundle
	clip_name: str = ""
	metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LocalVideoGenerator:
	"""Runs a configurable local generation command and validates its output."""

	executable: str = "python"
	script_path: str | None = None
	base_arguments: list[str] = field(default_factory=list)
	working_directory: str | Path | None = None

	def generate_clip_video(self, request: VideoGenerationRequest) -> Path:
		"""Run the configured local command to produce one clip video."""

		input_image_path = Path(request.input_image_path)
		output_video_path = Path(request.output_video_path)

		if not input_image_path.exists():
			raise VideoGenerationError(f"Input image does not exist: {input_image_path}")

		output_video_path.parent.mkdir(parents=True, exist_ok=True)

		command = self._build_command(request, input_image_path, output_video_path)

		try:
			subprocess.run(
				command,
				check=True,
				capture_output=True,
				text=True,
				cwd=str(self.working_directory) if self.working_directory is not None else None,
			)
		except FileNotFoundError as error:
			raise VideoGenerationError(
				f"Could not start video generation command: {command[0]}"
			) from error
		except subprocess.CalledProcessError as error:
			raise VideoGenerationError(
				f"Video generation failed for {request.clip_name or output_video_path.name}: {error.stderr.strip()}"
			) from error

		if not output_video_path.exists():
			raise VideoGenerationError(
				f"Video generation completed but no file was written: {output_video_path}"
			)

		return output_video_path

	def _build_command(
		self,
		request: VideoGenerationRequest,
		input_image_path: Path,
		output_video_path: Path,
	) -> list[str]:
		if self.script_path is None:
			raise VideoGenerationError("script_path must be set to run local video generation")

		motion_prompt = request.prompt_bundle.motion_prompt or self._compose_motion_prompt(request.prompt_bundle)

		command = [self.executable, self.script_path, *self.base_arguments]
		command.extend([
			"--input-image",
			str(input_image_path),
			"--output-video",
			str(output_video_path),
			"--prompt",
			motion_prompt,
			"--clip-duration",
			str(request.prompt_bundle.clip_duration_seconds),
		])

		if request.prompt_bundle.negative_prompt:
			command.extend(["--negative-prompt", request.prompt_bundle.negative_prompt])

		if request.clip_name:
			command.extend(["--clip-name", request.clip_name])

		for key, value in request.metadata.items():
			command.extend([f"--{key.replace('_', '-')}", value])

		return command

	@staticmethod
	def _compose_motion_prompt(prompt_bundle: PromptBundle) -> str:
		parts = [
			prompt_bundle.motion,
			f"moving object: {prompt_bundle.moving_object}" if prompt_bundle.moving_object else "",
			f"direction: {prompt_bundle.direction}" if prompt_bundle.direction else "",
			f"constraints: {prompt_bundle.constraints}" if prompt_bundle.constraints else "",
			f"stop condition: {prompt_bundle.stop_condition}" if prompt_bundle.stop_condition else "",
		]
		return ", ".join(part for part in parts if part)


def generate_clip_video(
	input_image_path: str | Path,
	output_video_path: str | Path,
	prompt_bundle: PromptBundle,
	*,
	clip_name: str = "",
	executable: str = "python",
	script_path: str | None = None,
	base_arguments: Sequence[str] | None = None,
	working_directory: str | Path | None = None,
	metadata: dict[str, str] | None = None,
) -> Path:
	"""Convenience wrapper for the default local generator."""

	generator = LocalVideoGenerator(
		executable=executable,
		script_path=script_path,
		base_arguments=list(base_arguments or []),
		working_directory=working_directory,
	)
	request = VideoGenerationRequest(
		input_image_path=input_image_path,
		output_video_path=output_video_path,
		prompt_bundle=prompt_bundle,
		clip_name=clip_name,
		metadata=dict(metadata or {}),
	)
	return generator.generate_clip_video(request)
