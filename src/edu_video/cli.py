"""CLI equivalent of the notebook API."""
import argparse
import json
from pathlib import Path

from .pipeline import VideoPipeline


def main():
    parser = argparse.ArgumentParser(description="Run a selected educational clip plan")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", default="runs/demo")
    parser.add_argument("--generator", choices=["dummy", "local"], default="dummy")
    parser.add_argument("--generator-script")
    parser.add_argument("--initial-image")
    parser.add_argument("--scene-images", help="JSON map of scene IDs to paths or image records; paths relative to this JSON file")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    try:
        images = {}
        if args.scene_images:
            source = Path(args.scene_images).resolve()
            images = json.loads(source.read_text(encoding="utf-8"))
            for key, value in images.items():
                entry = dict(value) if isinstance(value, dict) else {"path": value}
                entry["path"] = str((source.parent / entry["path"]).resolve())
                images[key] = entry
        pipeline = VideoPipeline(args.generator, args.output_dir, generator_script=args.generator_script)
        if args.preflight_only:
            result = pipeline.preflight(args.plan, scene_images=images, initial_image=args.initial_image)
            print(json.dumps({"status": "ready", "images": result["images"]}, indent=2))
        else:
            result = pipeline.run(args.plan, scene_images=images, initial_image=args.initial_image, resume=not args.no_resume)
            print(json.dumps({"video_path": str(result.video_path), "manifest_path": str(result.manifest_path)}))
    except Exception as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
