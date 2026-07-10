"""Load experiment knowledge from manual JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from models import Clip, Experiment, PromptBundle, Scene


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _parse_scene(scene_data: Mapping[str, Any]) -> Scene:
    return Scene(
        scene=str(scene_data.get("scene", "")),
        camera=str(scene_data.get("camera", "")),
        lighting=str(scene_data.get("lighting", "")),
        objects=list(scene_data.get("objects", [])),
        background=str(scene_data.get("background", "")),
        metadata=_as_dict(scene_data.get("metadata", {})),
    )


def _parse_prompt_bundle(prompt_data: Mapping[str, Any]) -> PromptBundle:
    clip_duration = prompt_data.get("clip_duration_seconds", prompt_data.get("clip_duration", 1.0))

    return PromptBundle(
        image_prompt=str(prompt_data.get("image_prompt", "")),
        motion_prompt=str(prompt_data.get("motion_prompt", "")),
        moving_object=str(prompt_data.get("moving_object", "")),
        direction=str(prompt_data.get("direction", "")),
        motion=str(prompt_data.get("motion", "")),
        constraints=str(prompt_data.get("constraints", "")),
        stop_condition=str(prompt_data.get("stop_condition", "")),
        clip_duration_seconds=float(clip_duration),
        negative_prompt=str(prompt_data.get("negative_prompt", "")),
        metadata=_as_dict(prompt_data.get("metadata", {})),
    )


def _parse_clip(clip_data: Mapping[str, Any], index: int) -> Clip:
    scene_data = _as_dict(clip_data.get("scene", clip_data.get("image_state", {})))
    prompt_data = _as_dict(clip_data.get("prompt_bundle", clip_data.get("motion_prompt", {})))

    return Clip(
        name=str(clip_data.get("name", f"clip_{index + 1}")),
        image_state=_parse_scene(scene_data),
        motion_prompt=_parse_prompt_bundle(prompt_data),
        output_clip_path=clip_data.get("output_clip_path"),
        input_frame_path=clip_data.get("input_frame_path"),
        generated_image_path=clip_data.get("generated_image_path"),
        extracted_frame_path=clip_data.get("extracted_frame_path"),
        metadata=_as_dict(clip_data.get("metadata", {})),
    )


def load_experiment_from_json(file_path: str) -> Experiment:
    """Load a manual experiment definition from JSON."""

    data_path = Path(file_path)
    with data_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    clips = [
        _parse_clip(_as_dict(clip_data), index)
        for index, clip_data in enumerate(data.get("clips", []))
    ]

    return Experiment(
        name=str(data.get("name", "Unnamed Experiment")),
        description=str(data.get("description", "")),
        clips=clips,
        subject=str(data.get("subject", data.get("topic", ""))),
        educational_goal=str(data.get("educational_goal", data.get("learning_goal", ""))),
        knowledge_source=str(data.get("knowledge_source", data.get("knowledge_base_source", "manual_json"))),
        metadata=_as_dict(data.get("metadata", {})),
    )