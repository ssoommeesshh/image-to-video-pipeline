"""CLI entrypoint for the educational image-to-video pipeline."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from config import PipelineConfig
from knowledge_loader import load_experiment_from_json
from pipeline import ExperimentPipeline, ManualImageRequiredError, PipelineError
from video_generator import LocalVideoGenerator


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "experiment"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the clip-by-clip educational image-to-video pipeline."
    )
    parser.add_argument(
        "--knowledge-file",
        default=None,
        help="Path to the manual experiment knowledge JSON file.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Natural language query to search or generate experiment steps (RAG).",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional experiment output folder name. Defaults to a slug from knowledge JSON name.",
    )
    parser.add_argument(
        "--initial-image",
        default=None,
        help="Optional initial image file for the first clip.",
    )
    pipeline_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--output-dir",
        default=str(pipeline_dir / "outputs"),
        help="Base output directory for generated clips and final video.",
    )
    parser.add_argument(
        "--knowledge-dir",
        default=str(pipeline_dir / "knowledge"),
        help="Knowledge directory label used in configuration.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(pipeline_dir / "inputs"),
        help="Input directory label used in configuration.",
    )
    parser.add_argument(
        "--generator-executable",
        default="python",
        help="Command executable used to run the local Wan generation script.",
    )
    parser.add_argument(
        "--generator-script",
        required=True,
        help="Path to the local Wan generation script in Lightning AI.",
    )
    parser.add_argument(
        "--generator-working-dir",
        default=None,
        help="Optional working directory for generator execution.",
    )
    parser.add_argument(
        "--generator-arg",
        action="append",
        default=[],
        help="Extra argument for the local generator command. Repeat for multiple arguments.",
    )
    parser.add_argument(
        "--wan-repo-dir",
        default=None,
        help="Path to the local Wan 2.2 repository checkout. Used by the local wrapper.",
    )
    parser.add_argument(
        "--wan-model-preset",
        default="ti2v-5b",
        choices=["ti2v-5b", "i2v-a14b"],
        help="Wan model preset used by the local wrapper.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.knowledge_file and not args.query:
        print("Error: Either --knowledge-file or --query must be provided.", file=sys.stderr)
        return 1

    if args.query:
        print(f"Running RAG lookup/generation for query: '{args.query}'...", file=sys.stderr)
        from rag_interface import lookup_or_generate, store
        try:
            experiment_dict = lookup_or_generate(args.query)
            knowledge_path = store(experiment_dict)
        except Exception as e:
            print(f"Error performing RAG query/generation: {e}", file=sys.stderr)
            return 1
    else:
        knowledge_path = Path(args.knowledge_file)

    if not knowledge_path.exists():
        print(f"Knowledge file not found: {knowledge_path}", file=sys.stderr)
        return 1

    experiment = load_experiment_from_json(str(knowledge_path))
    experiment_name = args.experiment_name or _slugify(experiment.name)

    config = PipelineConfig(
        base_output_dir=args.output_dir,
        knowledge_dir=args.knowledge_dir,
        input_dir=args.input_dir,
        experiment_name=experiment_name,
    )

    video_generator = LocalVideoGenerator(
        executable=args.generator_executable,
        script_path=args.generator_script,
        base_arguments=[
            *list(args.generator_arg),
            *( [f"--wan-repo-dir={args.wan_repo_dir}"] if args.wan_repo_dir else [] ),
            f"--model-preset={args.wan_model_preset}",
        ],
        working_directory=args.generator_working_dir,
    )

    pipeline = ExperimentPipeline(config=config, video_generator=video_generator)

    try:
        final_video_path = pipeline.run(
            experiment,
            initial_image_path=args.initial_image,
        )
    except ManualImageRequiredError as error:
        print("Pipeline paused: manual image input is required.", file=sys.stderr)
        print(f"Clip: {error.clip_name}", file=sys.stderr)
        print(f"Image prompt file: {error.image_prompt_path}", file=sys.stderr)
        print(f"Place generated image at: {error.expected_image_path}", file=sys.stderr)
        print("After placing the image, rerun the same command.", file=sys.stderr)
        return 2
    except PipelineError as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1

    print(f"Pipeline completed successfully. Final video: {final_video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
