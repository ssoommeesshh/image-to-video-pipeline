"""Pipeline orchestration for clip-by-clip educational video generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config import PipelineConfig
import sys
from frame_extractor import FrameExtractor
from models import Clip, Experiment
from prompt_builder import PromptBuilder
from stitcher import VideoStitcher
from video_generator import LocalVideoGenerator, VideoGenerationRequest


class PipelineError(RuntimeError):
	"""Raised when the pipeline cannot complete."""


class ManualImageRequiredError(PipelineError):
	"""Raised when a manual image is required before generation can continue."""

	def __init__(
		self,
		clip_name: str,
		expected_image_path: Path,
		image_prompt_path: Path,
		message: str,
	) -> None:
		super().__init__(message)
		self.clip_name = clip_name
		self.expected_image_path = expected_image_path
		self.image_prompt_path = image_prompt_path


@dataclass(slots=True)
class ExperimentPipeline:
	"""Coordinates clip-by-clip generation from structured experiment data."""

	config: PipelineConfig = field(default_factory=PipelineConfig)
	prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)
	video_generator: LocalVideoGenerator = field(default_factory=LocalVideoGenerator)
	frame_extractor: FrameExtractor = field(default_factory=FrameExtractor)
	stitcher: VideoStitcher = field(default_factory=VideoStitcher)

	def run(
		self,
		experiment: Experiment,
		*,
		initial_image_path: str | Path | None = None,
	) -> Path:
		"""Run the full clip loop and return the final stitched video path."""

		self._prepare_directories()

		clip_video_paths: list[Path] = []
		previous_frame_path = Path(initial_image_path) if initial_image_path else None

		for index, clip in enumerate(experiment.clips):
			clip_name = clip.name or f"clip_{index + 1}"
			clip.name = clip_name
			self._prepare_clip_directory(clip_name)

			prompt_bundle = self.prompt_builder.build_prompt_bundle(experiment, clip)
			image_prompt_path = self._write_clip_prompts(clip_name, prompt_bundle.image_prompt, prompt_bundle.motion_prompt)

			clip_input_image_path = self._resolve_clip_input_image(
				clip=clip,
				clip_name=clip_name,
				previous_frame_path=previous_frame_path,
				image_prompt_path=image_prompt_path,
			)

			output_video_path = Path(self.config.get_clip_video_file(clip_name))
			generated_video_path = self.video_generator.generate_clip_video(
				VideoGenerationRequest(
					input_image_path=clip_input_image_path,
					output_video_path=output_video_path,
					prompt_bundle=prompt_bundle,
					clip_name=clip_name,
					metadata={},
				)
			)

			last_frame_path = self.frame_extractor.extract_last_frame(
				generated_video_path,
				self.config.get_clip_frame_file(clip_name),
			)

			clip.generated_image_path = str(clip_input_image_path)
			clip.output_clip_path = str(generated_video_path)
			clip.extracted_frame_path = str(last_frame_path)
			previous_frame_path = last_frame_path
			clip_video_paths.append(generated_video_path)

		if not clip_video_paths:
			raise PipelineError("Experiment has no clips to generate")

		final_video_path = self.stitcher.stitch_clips(clip_video_paths, self.config.stitched_video_file)
		return final_video_path

	def _prepare_directories(self) -> None:
		Path(self.config.experiment_output_dir).mkdir(parents=True, exist_ok=True)
		Path(self.config.motion_prompts_dir).mkdir(parents=True, exist_ok=True)
		Path(self.config.input_image_dir).mkdir(parents=True, exist_ok=True)

	def _prepare_clip_directory(self, clip_name: str) -> None:
		Path(self.config.get_clip_dir(clip_name)).mkdir(parents=True, exist_ok=True)

	def _write_clip_prompts(self, clip_name: str, image_prompt: str, motion_prompt: str) -> Path:
		image_prompt_path = Path(self.config.get_clip_dir(clip_name)) / "image_prompt.txt"
		motion_prompt_path = Path(self.config.get_clip_motion_prompt_file(clip_name))

		image_prompt_path.write_text(image_prompt.strip() + "\n", encoding="utf-8")
		motion_prompt_path.parent.mkdir(parents=True, exist_ok=True)
		motion_prompt_path.write_text(motion_prompt.strip() + "\n", encoding="utf-8")

		return image_prompt_path

	def _resolve_clip_input_image(
		self,
		*,
		clip: Clip,
		clip_name: str,
		previous_frame_path: Path | None,
		image_prompt_path: Path,
	) -> Path:
		if clip.generated_image_path:
			provided_image_path = Path(clip.generated_image_path)
			if provided_image_path.exists():
				return provided_image_path

		if previous_frame_path and previous_frame_path.exists():
			return previous_frame_path

		# Repository root (two levels up from this file)
		repo_root = Path(__file__).resolve().parent.parent
		package_root = Path(__file__).resolve().parent
		print(f"[debug] repo_root={repo_root}", file=sys.stderr)

		# Helper to check both cwd-relative and repo-root-relative locations
		def _exists_and_resolve(p: Path) -> Path | None:
			if p.exists():
				return p
			if not p.is_absolute():
				# check package-local path first (image_to_video_pipeline/<p>)
				pkg_alt = package_root / p
				if pkg_alt.exists():
					return pkg_alt
				# then check repo-root-relative path
				repo_alt = repo_root / p
				if repo_alt.exists():
					return repo_alt
			return None

		# Candidate: workspace-level input folders (e.g. "input/" or "inputs/")
		workspace_candidates = [
			Path(self.config.input_dir) / f"{self.config.experiment_name}" / f"input_image.{self.config.default_image_extension}",
			Path(self.config.input_dir) / f"input_image.{self.config.default_image_extension}",
		]
		for p in workspace_candidates:
			resolved = _exists_and_resolve(p)
			print(f"[debug] checking workspace candidate: {p} -> resolved={resolved}", file=sys.stderr)
			if resolved:
				return resolved

		# Candidate: experiment-level shared input (outputs/<experiment>/input/input_image.png)
		generic_input_image_path = Path(self.config.get_input_image_file())
		resolved = _exists_and_resolve(generic_input_image_path)
		if not generic_input_image_path.is_absolute():
			print(f"[debug] generic alt path: {repo_root / generic_input_image_path} -> exists={(repo_root / generic_input_image_path).exists()}", file=sys.stderr)
		print(f"[debug] checking generic input image: {generic_input_image_path} -> resolved={resolved}", file=sys.stderr)
		if resolved:
			return resolved

		# Candidate: clip-specific input inside the clip directory
		expected_image_path = Path(self.config.get_clip_image_file(clip_name))
		resolved = _exists_and_resolve(expected_image_path)
		if not expected_image_path.is_absolute():
			print(f"[debug] clip alt path: {repo_root / expected_image_path} -> exists={(repo_root / expected_image_path).exists()}", file=sys.stderr)
		print(f"[debug] checking clip-specific image: {expected_image_path} -> resolved={resolved}", file=sys.stderr)
		if resolved:
			return resolved

		raise ManualImageRequiredError(
			clip_name=clip_name,
			expected_image_path=expected_image_path,
			image_prompt_path=image_prompt_path,
			message=(
				f"Manual image required for {clip_name}. "
				f"Generate an image from prompt file '{image_prompt_path}' and place it at either "
				f"'{generic_input_image_path}' or '{expected_image_path}'."
			),
		)
