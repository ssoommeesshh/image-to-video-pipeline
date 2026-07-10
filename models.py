"""Core shared data structures for the image-to-video pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class Scene:
    """Static visual state for a single experiment step."""

    scene: str
    camera: str
    lighting: str
    objects: list[str] = field(default_factory=list)
    background: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptBundle:
    """Structured prompt data for generating a clip."""

    image_prompt: str = ""
    motion_prompt: str = ""
    moving_object: str = ""
    direction: str = ""
    motion: str = ""
    constraints: str = ""
    stop_condition: str = ""
    clip_duration_seconds: float = 1.0
    negative_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Clip:
    """One generated segment in the full experiment sequence."""

    image_state: Scene
    motion_prompt: PromptBundle
    name: str = ""
    output_clip_path: Optional[str] = None
    input_frame_path: Optional[str] = None
    generated_image_path: Optional[str] = None
    extracted_frame_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Experiment:
    """Top-level experiment definition and clip sequence."""

    name: str
    description: str = ""
    clips: list[Clip] = field(default_factory=list)
    subject: str = ""
    educational_goal: str = ""
    knowledge_source: str = "manual_json"
    metadata: dict[str, Any] = field(default_factory=dict)


ImageState = Scene
MotionPrompt = PromptBundle


__all__ = [
    "Scene",
    "PromptBundle",
    "Clip",
    "Experiment",
    "ImageState",
    "MotionPrompt",
]