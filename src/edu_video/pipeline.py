"""Noninteractive execution of a selected plan with reproducible CPU artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from PIL import Image

from .adapter import to_experiment
from .contract import load_plan
from .generators import DummyGenerator, LocalGenerator, VideoGenerationRequest
from .media import inspect_video, stitch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


@dataclass(frozen=True)
class RunResult:
    video_path: Path
    manifest_path: Path
    manifest: dict


class VideoPipeline:
    def __init__(self, generator="dummy", output_dir="runs/demo", *, generator_script=None):
        if generator == "dummy":
            self.generator = DummyGenerator()
        elif generator == "local" and generator_script:
            self.generator = LocalGenerator(generator_script)
        else:
            raise ValueError("Use generator='dummy', or generator='local' with generator_script")
        self.output_dir = Path(output_dir).expanduser().resolve()

    def preflight(self, plan, *, scene_images=None, initial_image=None):
        data, base = load_plan(plan)
        scene_images = scene_images or {}
        required = {c["scene_id"] for c in data["clips"] if c["new_scene"]}
        errors, images = [], {}
        unknown = set(scene_images) - required
        if unknown:
            errors.append(f"Unknown scene image keys: {sorted(unknown)}")
        for index, clip in enumerate(data["clips"]):
            if not clip["new_scene"]:
                continue
            # Caller overrides are cwd-relative; embedded paths are plan-relative.
            entry = scene_images.get(clip["scene_id"])
            origin = Path.cwd()
            if entry is None and index == 0 and initial_image is not None:
                entry = initial_image
            if entry is None:
                entry, origin = clip["starting_image"], base
            if entry is None:
                errors.append(f"Missing image for scene {clip['scene_id']} ({clip['clip_id']})")
                continue
            entry = dict(entry) if isinstance(entry, dict) else {"path": str(entry), "provenance": "user_supplied", "review_status": "not_reviewed"}
            try:
                path = Path(entry["path"]).expanduser()
                path = (origin / path).resolve() if not path.is_absolute() else path.resolve()
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                images[clip["clip_id"]] = {**entry, "path": str(path), "sha256": file_hash(path)}
            except (OSError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"Invalid image for {clip['scene_id']}: {exc}")
        if errors:
            raise ValueError("Preflight failed:\n" + "\n".join(errors))
        if data["review_status"] == "integration_fixture" and self.generator.identity["name"] != "dummy":
            raise ValueError("Integration fixtures may only run with the dummy generator")
        return {"plan": data, "images": images, "generator": self.generator.identity}

    def run(self, plan, *, scene_images=None, initial_image=None, resume=True):
        preflight = self.preflight(plan, scene_images=scene_images, initial_image=initial_image)
        data, images = preflight["plan"], preflight["images"]
        experiment = to_experiment(data)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock = self.output_dir / ".run.lock"
        # A unique directory per simultaneous job; never race shared artifacts.
        try:
            handle = lock.open("x")
        except FileExistsError as exc:
            raise RuntimeError(f"Output directory is already in use: {self.output_dir}") from exc
        handle.close()
        try:
            return self._run(data, images, experiment, resume)
        finally:
            lock.unlink(missing_ok=True)

    def _run(self, data, images, experiment, resume):
        manifest_path = self.output_dir / "run_manifest.json"
        old = {}
        if resume and manifest_path.exists():
            try:
                old = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        cached = {c["clip_id"]: c for c in old.get("clips", [])}
        manifest = {"status": "running", "experiment_id": data["experiment_id"], "plan_sha256": object_hash(data),
                    "dataset_revision": data["dataset_revision"], "generator": self.generator.identity,
                    "review_status": data["review_status"], "visual_review_status": "not_reviewed",
                    "output_kind": "integration_output" if self.generator.identity["name"] == "dummy" else "unreviewed_video",
                    "clips": []}
        write_json(self.output_dir / "clip_plan.json", data)
        write_json(manifest_path, manifest)
        previous, paths = None, []
        active_clip = None
        try:
            for spec, clip in zip(data["clips"], experiment.clips):
                active_clip = spec["clip_id"]
                folder = self.output_dir / active_clip
                folder.mkdir(exist_ok=True)
                input_record = images[active_clip] if spec["new_scene"] else {
                    "path": str(previous), "sha256": file_hash(previous), "provenance": "previous_clip_last_frame",
                    "review_status": "not_reviewed"}
                input_path = Path(input_record["path"])
                request_hash = object_hash({"clip": spec, "input_sha256": input_record["sha256"],
                                            "generator": self.generator.identity, "runner_version": 1})
                video, last_frame = folder / "clip.mp4", folder / "last_frame.png"
                old_clip = cached.get(active_clip, {})
                reuse = (resume and video.is_file() and old_clip.get("request_sha256") == request_hash
                         and old_clip.get("video_sha256") == file_hash(video))
                if not reuse:
                    temporary = folder / "generating.mp4"
                    temporary.unlink(missing_ok=True)
                    self.generator.generate_clip_video(VideoGenerationRequest(input_path, temporary, clip.motion_prompt, active_clip))
                    info = inspect_video(temporary, last_frame)
                    temporary.replace(video)
                else:
                    info = inspect_video(video, last_frame)
                if abs(info["duration_seconds"] - spec["duration_seconds"]) > max(0.1, 1.0 / info["fps"]):
                    raise ValueError(f"{active_clip}: actual duration differs from requested duration")
                if manifest["clips"] and (info["size"], info["fps"]) != (manifest["clips"][0]["media"]["size"], manifest["clips"][0]["media"]["fps"]):
                    raise ValueError("Clips must have matching resolution and frame rate before stitching")
                manifest["clips"].append({"clip_id": active_clip, "scene_id": spec["scene_id"], "step_ids": spec["step_ids"],
                    "request_sha256": request_hash, "input_image": input_record, "prompts": {
                        "image": spec["image_prompt"], "motion": spec["motion_prompt"], "negative": spec["negative_prompt"]},
                    "source_passage_ids": spec["source_passage_ids"], "review_status": spec["review_status"],
                    "requested_duration_seconds": spec["duration_seconds"], "generated_duration_seconds": info["duration_seconds"],
                    "retained_duration_seconds": info["duration_seconds"], "duration_status": spec["duration_status"],
                    "video_path": str(video), "video_sha256": file_hash(video), "last_frame_path": str(last_frame),
                    "last_frame_sha256": file_hash(last_frame), "reused": bool(reuse), "media": info})
                write_json(manifest_path, manifest)
                previous = last_frame
                paths.append(video)
            final = self.output_dir / "final_video.mp4"
            temporary_final = self.output_dir / "stitching.mp4"
            stitch(paths, temporary_final)
            final_info = inspect_video(temporary_final)
            if final_info["frame_count"] != sum(c["media"]["frame_count"] for c in manifest["clips"]):
                raise ValueError("Stitched frame count does not match ordered input clips")
            temporary_final.replace(final)
            manifest.update(status="complete", video_path=str(final), video_sha256=file_hash(final), media=final_info)
            write_json(manifest_path, manifest)
            return RunResult(final, manifest_path, manifest)
        except Exception as exc:
            manifest.update(status="failed", failed_clip_id=active_clip, error=str(exc))
            write_json(manifest_path, manifest)
            raise
