"""Pipeline-wide configuration and filesystem layout helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PipelineConfig:
    """Shared configuration for the research video pipeline."""

    base_output_dir: str = "outputs"
    knowledge_dir: str = "knowledge"
    input_dir: str = "inputs"
    experiment_name: str = "newtons_cradle"
    default_clip_duration_seconds: float = 1.0
    default_image_extension: str = "png"
    default_video_extension: str = "mp4"

    @property
    def experiment_output_dir(self) -> str:
        return f"{self.base_output_dir}/{self.experiment_name}"

    @property
    def image_prompt_file(self) -> str:
        return f"{self.experiment_output_dir}/image_prompt.txt"

    @property
    def motion_prompts_dir(self) -> str:
        return f"{self.experiment_output_dir}/motion_prompts"

    @property
    def stitched_video_file(self) -> str:
        return f"{self.experiment_output_dir}/final_video.{self.default_video_extension}"

    @property
    def input_image_dir(self) -> str:
        return f"{self.experiment_output_dir}/input"

    def get_clip_dir(self, clip_name: str) -> str:
        return f"{self.experiment_output_dir}/{clip_name}"

    def get_clip_motion_prompt_file(self, clip_name: str) -> str:
        return f"{self.motion_prompts_dir}/{clip_name}_motion.txt"

    def get_clip_video_file(self, clip_name: str) -> str:
        return f"{self.experiment_output_dir}/{clip_name}.{self.default_video_extension}"

    def get_clip_image_file(self, clip_name: str) -> str:
        return f"{self.get_clip_dir(clip_name)}/input_image.{self.default_image_extension}"

    def get_clip_frame_file(self, clip_name: str) -> str:
        return f"{self.get_clip_dir(clip_name)}/last_frame.{self.default_image_extension}"

    def get_input_image_file(self) -> str:
        return f"{self.input_image_dir}/input_image.{self.default_image_extension}"