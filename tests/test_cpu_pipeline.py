import copy
import json
from importlib.resources import files
from pathlib import Path

from PIL import Image
import pytest

from edu_video import VideoPipeline, validate_plan
from edu_video.media import inspect_video


@pytest.fixture
def plan():
    return json.loads(files("edu_video").joinpath("fixtures/cpu_plan.json").read_text())


@pytest.fixture
def images(tmp_path):
    result = {}
    for scene, color in [("scene_1", "red"), ("scene_2", "blue")]:
        path = tmp_path / f"{scene}.png"
        Image.new("RGB", (160, 120), color).save(path)
        result[scene] = path
    return result


def test_order_continuity_scene_reset_and_cache(plan, images, tmp_path):
    pipeline = VideoPipeline(output_dir=tmp_path / "output with space's")
    result = pipeline.run(plan, scene_images=images)
    assert result.manifest["output_kind"] == "integration_output"
    clips = result.manifest["clips"]
    assert [c["clip_id"] for c in clips] == [c["clip_id"] for c in plan["clips"]]
    assert clips[0]["input_image"]["path"] == str(images["scene_1"])
    assert clips[1]["input_image"]["sha256"] == clips[0]["last_frame_sha256"]
    assert clips[2]["input_image"]["path"] == str(images["scene_2"])
    assert result.manifest["media"]["frame_count"] == 18
    with Image.open(clips[0]["last_frame_path"]) as im:
        assert im.getpixel((10, 10))[0] > 200
    with Image.open(clips[2]["last_frame_path"]) as im:
        assert im.getpixel((10, 10))[2] > 200
    assert all(c["reused"] for c in pipeline.run(plan, scene_images=images).manifest["clips"])
    plan["clips"][0]["motion_prompt"] += " Changed request."
    changed = pipeline.run(plan, scene_images=images)
    assert not changed.manifest["clips"][0]["reused"]
    # Image mutation invalidates the first request and the dependent frame chain.
    Image.new("RGB", (160, 120), "green").save(images["scene_1"])
    changed = pipeline.run(plan, scene_images=images)
    assert not changed.manifest["clips"][0]["reused"]
    assert not changed.manifest["clips"][1]["reused"]
    assert changed.manifest["clips"][2]["reused"]
    Path(changed.manifest["clips"][2]["video_path"]).write_bytes(b"corrupt")
    assert not pipeline.run(plan, scene_images=images).manifest["clips"][2]["reused"]


def test_initial_image_survives_first_scene_reset(plan, images, tmp_path):
    plan["clips"] = plan["clips"][:2]
    result = VideoPipeline(output_dir=tmp_path / "run").run(plan, initial_image=images["scene_1"])
    assert result.manifest["clips"][0]["input_image"]["path"] == str(images["scene_1"])


def test_missing_images_report_all_before_generation(plan, tmp_path, monkeypatch):
    pipeline = VideoPipeline(output_dir=tmp_path / "run")
    monkeypatch.setattr(pipeline.generator, "generate_clip_video", lambda request: pytest.fail("generator called"))
    with pytest.raises(ValueError, match="scene_1[\\s\\S]*scene_2"):
        pipeline.run(plan)
    assert not pipeline.output_dir.exists()


@pytest.mark.parametrize("fault", ["missing_scene", "removed_file", "non_image"])
def test_one_bad_scene_stops_entire_run(plan, images, tmp_path, monkeypatch, fault):
    pipeline = VideoPipeline(output_dir=tmp_path / "run")
    monkeypatch.setattr(pipeline.generator, "generate_clip_video", lambda request: pytest.fail("generator called"))
    if fault == "missing_scene":
        del images["scene_2"]
    elif fault == "removed_file":
        images["scene_2"].unlink()
    else:
        text = tmp_path / "not_an_image.txt"
        text.write_text("Not a scene image")
        images["scene_2"] = text
    with pytest.raises(ValueError, match="scene_2"):
        pipeline.run(plan, scene_images=images)
    assert not pipeline.output_dir.exists()


@pytest.mark.parametrize("decision", ["blocked", "out_of_scope", "selection_required", "not_in_catalog"])
def test_rejected_decision_never_calls_generator(decision, tmp_path, monkeypatch):
    pipeline = VideoPipeline(output_dir=tmp_path / "run")
    monkeypatch.setattr(pipeline.generator, "generate_clip_video", lambda request: pytest.fail("generator called"))
    with pytest.raises(Exception):
        pipeline.run({"status": decision, "clips": []})
    assert not pipeline.output_dir.exists()


def test_sequence_and_evidence_validation(plan):
    invalid = copy.deepcopy(plan)
    invalid["clips"][2]["new_scene"] = False
    with pytest.raises(ValueError, match="scene transition"):
        validate_plan(invalid)
    invalid = copy.deepcopy(plan)
    invalid["clips"][1]["clip_id"] = invalid["clips"][0]["clip_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_plan(invalid)
    invalid = copy.deepcopy(plan)
    invalid["review_status"] = "evidence_gate_passed"
    with pytest.raises(ValueError, match="passage IDs"):
        validate_plan(invalid)
    invalid = copy.deepcopy(plan)
    invalid["clips"][0]["clip_id"] = "../escape"
    with pytest.raises(Exception):
        validate_plan(invalid)


def test_plan_relative_image_and_invalid_image(plan, images, tmp_path):
    plan["clips"] = plan["clips"][:1]
    plan["clips"][0]["starting_image"] = {"path": "scene_1.png", "provenance": "test", "review_status": "integration_fixture"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    pipeline = VideoPipeline(output_dir=tmp_path / "run")
    assert pipeline.preflight(path)["images"]["fixture_clip_1"]["path"] == str(images["scene_1"])
    images["scene_1"].write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Invalid image"):
        pipeline.preflight(path)


def test_failed_run_manifest_and_resume(plan, images, tmp_path, monkeypatch):
    pipeline = VideoPipeline(output_dir=tmp_path / "run")
    original = pipeline.generator.generate_clip_video
    def fail_second(request):
        if request.clip_name == "fixture_clip_2":
            raise RuntimeError("simulated provider failure")
        return original(request)
    monkeypatch.setattr(pipeline.generator, "generate_clip_video", fail_second)
    with pytest.raises(RuntimeError, match="simulated"):
        pipeline.run(plan, scene_images=images)
    failed = json.loads((pipeline.output_dir / "run_manifest.json").read_text())
    assert failed["status"] == "failed"
    assert failed["failed_clip_id"] == "fixture_clip_2"
    monkeypatch.setattr(pipeline.generator, "generate_clip_video", original)
    result = pipeline.run(plan, scene_images=images)
    assert result.manifest["clips"][0]["reused"]
    assert not result.manifest["clips"][1]["reused"]


def test_last_frame_is_actual_final_frame(tmp_path):
    from edu_video.media import ffmpeg
    video, frame = tmp_path / "changing.mp4", tmp_path / "last.png"
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=160x120:r=12:d=0.5", "-f", "lavfi", "-i",
           "color=c=blue:s=160x120:r=12:d=0.5", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
           "-map", "[v]", "-c:v", "libx264", video)
    assert inspect_video(video, frame)["frame_count"] == 12
    with Image.open(frame) as im:
        assert im.getpixel((10, 10))[2] > 200
