"""Smoke test helper for the educational image-to-video pipeline.

This script verifies the earliest safe test boundary:
1. load the manual knowledge JSON
2. write the prompt files for the first clip
3. resolve the first clip input image without starting video generation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import PipelineConfig
from knowledge_loader import load_experiment_from_json
from pipeline import ExperimentPipeline, ManualImageRequiredError


def _slugify(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "experiment"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a pipeline smoke test without Wan 2.2.")
    parser.add_argument(
        "--knowledge-file",
        default=str(Path(__file__).resolve().parent / "knowledge" / "newtons_cradle.json"),
        help="Path to the manual experiment JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Base output directory used for smoke-test artifacts.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional override for the experiment output folder name.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    knowledge_path = Path(args.knowledge_file)
    if not knowledge_path.exists():
        print(f"FAIL: knowledge file not found: {knowledge_path}")
        return 1

    experiment = load_experiment_from_json(str(knowledge_path))
    experiment_name = args.experiment_name or _slugify(experiment.name)

    config = PipelineConfig(
        base_output_dir=args.output_dir,
        experiment_name=experiment_name,
    )
    pipeline = ExperimentPipeline(config=config)
    pipeline._prepare_directories()

    if not experiment.clips:
        print("FAIL: no clips were found in the knowledge file.")
        return 1

    first_clip = experiment.clips[0]
    clip_name = first_clip.name or "clip_1"
    first_clip.name = clip_name
    pipeline._prepare_clip_directory(clip_name)

    prompt_bundle = pipeline.prompt_builder.build_prompt_bundle(experiment, first_clip)
    image_prompt_path = pipeline._write_clip_prompts(
        clip_name,
        prompt_bundle.image_prompt,
        prompt_bundle.motion_prompt,
    )

    try:
        resolved_input_image_path = pipeline._resolve_clip_input_image(
            clip=first_clip,
            clip_name=clip_name,
            previous_frame_path=None,
            image_prompt_path=image_prompt_path,
        )
    except ManualImageRequiredError as error:
        image_prompt_exists = error.image_prompt_path.exists()
        expected_image_parent_exists = error.expected_image_path.parent.exists()

        print("PASS: smoke test reached the manual-image pause point.")
        print(f"Experiment: {experiment.name}")
        print(f"Clip: {error.clip_name}")
        print(f"Prompt file: {error.image_prompt_path} ({'found' if image_prompt_exists else 'missing'})")
        print(
            f"Expected manual image path: {error.expected_image_path} "
            f"({'parent exists' if expected_image_parent_exists else 'parent missing'})"
        )
        print("Next step: place the manual image at one of the accepted locations, then rerun this smoke test or use main.py.")

        return 0 if image_prompt_exists else 2

    print("PASS: smoke test reached the image-resolution checkpoint without starting video generation.")
    print(f"Experiment: {experiment.name}")
    print(f"Clip: {clip_name}")
    print(f"Prompt file: {image_prompt_path} (found)")
    print(f"Resolved input image: {resolved_input_image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())