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

		try:
			for index, clip in enumerate(experiment.clips):
				clip_name = clip.name or f"clip_{index + 1}"
				clip.name = clip_name

				# Check if this is a static title clip
				is_title = clip.metadata.get("is_title", False)
				if is_title:
					title_text = clip.metadata.get("title_text", "Title Card")
					clip_duration = float(clip.metadata.get("clip_duration", 3.0))
					output_video_path = Path(self.config.get_clip_video_file(clip_name))
					
					# Generate static video if it doesn't exist
					if not output_video_path.exists():
						print(f"[pipeline] Generating static title card video for '{clip_name}' (text: {title_text.replace('\n', ' ')})...", file=sys.stderr)
						self._generate_title_card_video(title_text, clip_duration, output_video_path)
					
					clip_video_paths.append(output_video_path)
					# Reset continuity so the next scene starts fresh
					previous_frame_path = None
					continue

				# Reset previous frame path if this clip transitions to a new scene
				if clip.metadata.get("new_scene") or clip.metadata.get("is_new_scene"):
					print(f"[pipeline] Clip '{clip_name}' starts a new scene. Resetting continuity.", file=sys.stderr)
					previous_frame_path = None

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
				if output_video_path.exists():
					print(f"Clip video {output_video_path} already exists. Skipping generation.")
					generated_video_path = output_video_path
				else:
					generated_video_path = self.video_generator.generate_clip_video(
						VideoGenerationRequest(
							input_image_path=clip_input_image_path,
							output_video_path=output_video_path,
							prompt_bundle=prompt_bundle,
							clip_name=clip_name,
							metadata={},
						)
					)

				# Evaluate visual continuity between the starting image and first frame of the generated clip
				try:
					from evaluator import frame_similarity, extract_frames
					from PIL import Image
					clip_frames = extract_frames(generated_video_path, num_frames=1)
					if clip_frames and clip_input_image_path and Path(clip_input_image_path).exists():
						input_img = Image.open(clip_input_image_path)
						similarity = frame_similarity(input_img, clip_frames[0])
						print(f"[evaluator] Clip '{clip_name}' visual continuity score: {similarity:.4f}", file=sys.stderr)
						if similarity < 0.55:
							print(f"⚠️ WARNING: Low visual continuity detected ({similarity:.4f}). The video generator may have warped the starting frame.", file=sys.stderr)
				except Exception as e:
					print(f"[evaluator warning] Failed to compute continuity metrics: {e}", file=sys.stderr)

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
		finally:
			if hasattr(self.video_generator, "stop_daemon"):
				self.video_generator.stop_daemon()

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

		# Helper to resolve check with common image extensions (.png, .jpg, .jpeg, etc.)
		def _resolve_with_extensions(base_path_with_ext: Path) -> Path | None:
			resolved = _exists_and_resolve(base_path_with_ext)
			if resolved:
				return resolved
			stem = base_path_with_ext.stem
			parent = base_path_with_ext.parent
			for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
				alt_path = parent / f"{stem}.{ext}"
				resolved = _exists_and_resolve(alt_path)
				if resolved:
					return resolved
			return None

		# 1. Previous frame stitched image (takes precedence for video continuity)
		if previous_frame_path and previous_frame_path.exists():
			return previous_frame_path

		# 2. Candidate: clip-specific input inside the clip directory (e.g. manual image for the clip)
		expected_image_path = Path(self.config.get_clip_image_file(clip_name))
		resolved = _resolve_with_extensions(expected_image_path)
		if resolved:
			print(f"[debug] checking clip-specific image: {expected_image_path} -> resolved={resolved}", file=sys.stderr)
			return resolved

		# 3. Candidate: workspace-level input folders (e.g. "input/" or "inputs/")
		workspace_candidates = [
			Path(self.config.input_dir) / f"{self.config.experiment_name}" / f"input_image.{self.config.default_image_extension}",
			Path(self.config.input_dir) / f"input_image.{self.config.default_image_extension}",
		]
		for p in workspace_candidates:
			resolved = _resolve_with_extensions(p)
			print(f"[debug] checking workspace candidate: {p} -> resolved={resolved}", file=sys.stderr)
			if resolved:
				return resolved

		# 4. Candidate: experiment-level shared input (outputs/<experiment>/input/input_image.png)
		generic_input_image_path = Path(self.config.get_input_image_file())
		resolved = _resolve_with_extensions(generic_input_image_path)
		if not generic_input_image_path.is_absolute():
			print(f"[debug] generic alt path: {repo_root / generic_input_image_path} -> exists={(repo_root / generic_input_image_path).exists()}", file=sys.stderr)
		print(f"[debug] checking generic input image: {generic_input_image_path} -> resolved={resolved}", file=sys.stderr)
		if resolved:
			return resolved

		# Try generating the image with FLUX
		if image_prompt_path.exists():
			image_prompt = image_prompt_path.read_text(encoding="utf-8").strip()
			if image_prompt:
				try:
					print(f"No initial image found. Attempting to generate starting image using FLUX.1-dev...", file=sys.stderr)
					from image_generator import generate_image_from_prompt
					generate_image_from_prompt(image_prompt, expected_image_path)
					if expected_image_path.exists():
						print(f"Successfully generated starting image and saved to: {expected_image_path}", file=sys.stderr)
						return expected_image_path
				except Exception as e:
					print(f"Warning: FLUX image generation failed: {e}. Falling back to manual image request.", file=sys.stderr)

		# If all candidates fail, enter an interactive pause loop to wait for manual placement
		print(f"\n==========================================", file=sys.stderr)
		print(f"⏸️  PIPELINE PAUSED: Manual image input required", file=sys.stderr)
		print(f"Clip: {clip_name}", file=sys.stderr)
		print(f"Image prompt file: {image_prompt_path}", file=sys.stderr)
		print(f"Please place your generated image at: {expected_image_path}", file=sys.stderr)
		print(f"==========================================\n", file=sys.stderr)

		while True:
			resolved = _resolve_with_extensions(expected_image_path)
			if resolved:
				print(f"--> Found starting image: {resolved}. Resuming generation...", file=sys.stderr)
				return resolved
			
			try:
				input("Press [Enter] once you have placed the image to verify, or Ctrl+C to abort... ")
			except (KeyboardInterrupt, EOFError):
				print("\nAborting pipeline run...", file=sys.stderr)
				raise ManualImageRequiredError(
					clip_name=clip_name,
					expected_image_path=expected_image_path,
					image_prompt_path=image_prompt_path,
					message="Pipeline aborted by user during manual image wait.",
				)

	def _generate_title_card_video(self, text: str, duration: float, output_path: Path) -> None:
		"""Generates a static title card video with centered text on a dark background."""
		from PIL import Image, ImageDraw, ImageFont
		import numpy as np
		import imageio.v3 as iio

		# Create dark background image matching Wan 2.2 preset dimensions (832x480)
		width, height = 832, 480
		img = Image.new("RGB", (width, height), "#0d0d11")
		draw = ImageDraw.Draw(img)

		# Add a premium, glowing blue-indigo border line at the top
		draw.rectangle([(0, 0), (width, 8)], fill="#6366f1")

		# Try to load a bold sans-serif system font
		font = None
		font_paths = [
			"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
			"/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
			"/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
		]
		for p in font_paths:
			if Path(p).exists():
				try:
					font = ImageFont.truetype(p, 36)
					break
				except Exception:
					pass
		if font is None:
			font = ImageFont.load_default()

		# Draw multiline text centered on the canvas
		lines = text.split("\n")
		# Calculate line height
		if hasattr(font, "getbbox"):
			line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + 16
		else:
			line_height = 24
		
		total_height = len(lines) * line_height
		start_y = (height - total_height) // 2

		for idx, line in enumerate(lines):
			if hasattr(font, "getbbox"):
				bbox = font.getbbox(line)
				w = bbox[2] - bbox[0]
			else:
				w = len(line) * 8
			x = (width - w) // 2
			y = start_y + idx * line_height
			draw.text((x, y), line, fill="#ffffff", font=font)

		# Convert PIL image to stack of frames representing the duration at 16 FPS
		fps = 16
		num_frames = int(duration * fps)
		frame = np.array(img)
		video_frames = np.repeat(frame[np.newaxis, :, :, :], num_frames, axis=0)

		# Write static video
		output_path.parent.mkdir(parents=True, exist_ok=True)
		iio.imwrite(str(output_path), video_frames, fps=fps, codec="libx264")
