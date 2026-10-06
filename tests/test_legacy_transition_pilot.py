"""The legacy CLI path must honor scene references and render cards without Wan."""

from pathlib import Path

from PIL import Image

from config import PipelineConfig
from knowledge_loader import load_experiment_from_json
from pipeline import ExperimentPipeline
from edu_video.media import ffmpeg, inspect_video


class StaticClipGenerator:
    def __init__(self):
        self.requests = []

    def generate_clip_video(self, request):
        self.requests.append(request)
        path = Path(request.output_video_path)
        ffmpeg(
            "-f", "lavfi", "-i", "color=c=green:s=160x96:r=16:d=2",
            "-frames:v", "32", "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
        )
        return path

    def stop_daemon(self):
        pass


def test_two_clip_pilot_uses_reference_after_card_and_matches_card_geometry(tmp_path):
    image = tmp_path / "start.png"
    Image.new("RGB", (160, 96), "white").save(image)
    knowledge = Path(__file__).parents[1] / "knowledge" / "chloride_two_clip_resident_pilot.json"
    experiment = load_experiment_from_json(str(knowledge))
    generator = StaticClipGenerator()
    pipeline = ExperimentPipeline(
        config=PipelineConfig(base_output_dir=str(tmp_path / "out"), experiment_name="pilot"),
        video_generator=generator,
    )
    final = pipeline.run(experiment, initial_image_path=image)
    assert final.is_file()
    assert inspect_video(final)["frame_count"] == 176
    assert len(generator.requests) == 2
    assert Path(generator.requests[0].input_image_path) == image
    assert Path(generator.requests[1].input_image_path).name == "last_frame.png"
    assert "silver_nitrate" in str(generator.requests[1].input_image_path)
    for title in ("intro", "observation_card", "result_card"):
        assert inspect_video(tmp_path / "out" / "pilot" / f"{title}.mp4")["size"] == [160, 96]
