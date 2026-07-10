"""Build text prompts from the shared experiment models.

This module is responsible for turning an `Experiment` and one of its `Clip`
definitions into human-readable prompt text that downstream image and video
generation steps can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models import Clip, Experiment, PromptBundle, Scene


def _clean_parts(parts: list[str]) -> list[str]:
	return [part.strip() for part in parts if part and part.strip()]


def _join_sentences(parts: list[str]) -> str:
	return ". ".join(_clean_parts(parts))


def _join_phrases(parts: list[str]) -> str:
	return ", ".join(_clean_parts(parts))


@dataclass(slots=True)
class PromptBuilder:
	"""Constructs image and motion prompts for a single clip."""

	image_suffix: str = "high quality educational science illustration, realistic composition"
	motion_suffix: str = "keep motion physically plausible and visually clear"
	metadata: dict[str, str] = field(default_factory=dict)

	def build_image_prompt(self, experiment: Experiment, scene: Scene, clip: Clip) -> str:
		parts = [
			f"Create a {experiment.name} scene",
			experiment.description,
			scene.scene,
			f"Camera: {scene.camera}" if scene.camera else "",
			f"Lighting: {scene.lighting}" if scene.lighting else "",
			f"Objects: {_join_phrases(scene.objects)}" if scene.objects else "",
			f"Background: {scene.background}" if scene.background else "",
			self.metadata.get("image_style", ""),
			self.image_suffix,
		]
		prompt = _join_sentences(parts)

		if clip.motion_prompt.image_prompt:
			prompt = _join_sentences([prompt, clip.motion_prompt.image_prompt])

		return prompt

	def build_motion_prompt(self, experiment: Experiment, clip: Clip) -> str:
		motion = clip.motion_prompt
		
		# If a full motion_prompt is provided, use it directly
		if motion.motion_prompt:
			return motion.motion_prompt
		
		# Otherwise assemble from components
		parts = [
			f"Animate the next step of {experiment.name}",
			motion.moving_object,
			motion.motion,
			f"Direction: {motion.direction}" if motion.direction else "",
			f"Constraints: {motion.constraints}" if motion.constraints else "",
			f"Stop when: {motion.stop_condition}" if motion.stop_condition else "",
			self.metadata.get("motion_style", ""),
			self.motion_suffix,
		]
		return _join_sentences(parts)

	def build_prompt_bundle(self, experiment: Experiment, clip: Clip) -> PromptBundle:
		"""Create a normalized prompt bundle for downstream modules."""

		image_prompt = self.build_image_prompt(experiment, clip.image_state, clip)
		motion_prompt = self.build_motion_prompt(experiment, clip)

		return PromptBundle(
			image_prompt=image_prompt,
			motion_prompt=motion_prompt,
			moving_object=clip.motion_prompt.moving_object,
			direction=clip.motion_prompt.direction,
			motion=clip.motion_prompt.motion,
			constraints=clip.motion_prompt.constraints,
			stop_condition=clip.motion_prompt.stop_condition,
			clip_duration_seconds=clip.motion_prompt.clip_duration_seconds,
			negative_prompt=clip.motion_prompt.negative_prompt,
			metadata={
				**clip.motion_prompt.metadata,
				**clip.metadata,
				**self.metadata,
			},
		)


def build_prompt_bundle(experiment: Experiment, clip: Clip) -> PromptBundle:
	"""Convenience wrapper for the default prompt builder."""

	return PromptBuilder().build_prompt_bundle(experiment, clip)
