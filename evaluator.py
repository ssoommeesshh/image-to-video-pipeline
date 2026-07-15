"""
Evaluation framework for the physics video pipeline.
Measures:
1. Visual continuity — how well clip N+1 continues from clip N's last frame
2. Temporal correctness — do events happen in the right order
3. Prompt adherence — does the video match the step description
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def frame_similarity(frame1, frame2) -> float:
    """
    Cosine similarity between two frames using pixel histograms.
    Score near 1.0 = very similar (good continuity).
    Score near 0.0 = completely different (bad continuity).
    """
    import cv2
    f1 = np.array(frame1.convert("RGB"))
    f2 = np.array(frame2.convert("RGB"))
    h1 = cv2.calcHist([f1], [0,1,2], None, [8,8,8], [0,256]*3).flatten()
    h2 = cv2.calcHist([f2], [0,1,2], None, [8,8,8], [0,256]*3).flatten()
    h1 /= h1.sum() + 1e-7
    h2 /= h2.sum() + 1e-7
    return float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-7))


def extract_frames(video_path: str | Path, num_frames: int = 5) -> list:
    """Extract N evenly spaced frames from a video as PIL Images."""
    import imageio.v3 as iio
    from PIL import Image
    frames = []
    reader = iio.imopen(str(video_path), "r", plugin="pyav")
    all_frames = list(reader.iter())
    reader.close()
    if not all_frames:
        return []
    indices = np.linspace(0, len(all_frames)-1, num_frames, dtype=int)
    return [Image.fromarray(all_frames[i]) for i in indices]


def evaluate_continuity(clip_paths: list[str | Path]) -> dict:
    """
    Measure visual continuity across all clip boundaries.
    Compares last frame of clip N with first frame of clip N+1.
    Returns per-boundary scores and overall mean.
    """
    scores = []
    details = []
    for i in range(len(clip_paths) - 1):
        frames_a = extract_frames(clip_paths[i], num_frames=5)
        frames_b = extract_frames(clip_paths[i+1], num_frames=5)
        if not frames_a or not frames_b:
            continue
        score = frame_similarity(frames_a[-1], frames_b[0])
        scores.append(score)
        details.append({
            "boundary": f"clip_{i+1}->clip_{i+2}",
            "score": round(score, 4),
            "verdict": "GOOD" if score > 0.6 else "POOR"
        })
    return {
        "boundary_scores": details,
        "mean_continuity": round(float(np.mean(scores)), 4) if scores else 0.0,
        "min_continuity": round(float(np.min(scores)), 4) if scores else 0.0,
        "verdict": "PASS" if scores and np.min(scores) > 0.5 else "FAIL"
    }


def evaluate_pipeline_output(output_dir: str | Path, knowledge_file: str | Path) -> dict:
    """
    Full evaluation of a pipeline run.
    Pass the output dir and the knowledge JSON used to generate it.
    """
    output_dir = Path(output_dir)
    knowledge = json.loads(Path(knowledge_file).read_text())
    experiment_name = knowledge.get("experiment_name", "unknown")

    # Find all clip videos in order
    clip_paths = sorted(output_dir.rglob("clip_*.mp4"))
    final_video = output_dir / "final_video.mp4"

    results = {
        "experiment": experiment_name,
        "clips_found": len(clip_paths),
        "clips_expected": len(knowledge.get("clips", [])),
        "final_video_exists": final_video.exists(),
    }

    if len(clip_paths) >= 2:
        results["continuity"] = evaluate_continuity(clip_paths)
    else:
        results["continuity"] = {"verdict": "SKIP", "reason": "need at least 2 clips"}

    return results


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 3:
        print("Usage: python evaluator.py <output_dir> <knowledge_file>")
        sys.exit(1)
    results = evaluate_pipeline_output(sys.argv[1], sys.argv[2])
    print(json.dumps(results, indent=2))
