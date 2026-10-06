# Kaggle two-T4 resident Wan handoff

This path combines **ssoommeesshh/image-to-video-pipeline** (experiment JSON,
scene continuity, static explanation cards and stitching) with
**abhinavk0006/Wan2.2** (the tested two-T4 INT8/Lightning model code).
Use the original pipeline repository for `main.py` and
`kaggle_wan_resident.py`. The teammate's image-to-video fork is useful as the
working baseline and comparison, but its `kaggle_wan_wrapper.py` still starts
`kaggle_generate_video.py` separately for every clip.

## What this changes

- One worker initializes `WanI2V` and the VAE once per experiment.
- It encodes every motion and negative prompt from the experiment JSON with
  T5 once, then releases T5 before loading a DiT expert.
- The first high-noise and low-noise expert loads merge Lightning and convert
  the FP8 weights to INT8. Trusted, locally created INT8 copies are stored in
  `/kaggle/temp/wan22-int8-cache`. Later clips load those converted copies,
  avoiding another Hugging Face download and FP8/LoRA conversion.
- The experts still move through GPU memory one at a time. Two 14B experts
  plus activations do not fit on the tested T4 pair together, so each clip
  still incurs expert disk-to-GPU loading and denoising time. This is a
  **reduced reload cost**, not a zero-load claim.
- `--require-resident-daemon` refuses a silent fallback to a fresh child
  process when the resident worker cannot start.

The worker follows the API at Wan fork commit `78fb5ce`. It has passed local
Python syntax and CPU pipeline tests. **A two-T4 Kaggle GPU run is still
required to validate quantized model serialization, cache loading, and memory
headroom.** Run the two-clip pilot before an overnight batch.

## Kaggle setup

Select the **two T4 GPU** accelerator and enable internet. The working
notebook asserts two CUDA devices; the 6/10 GiB placement means 6 GiB planned
model storage on GPU 0 and 10 GiB on GPU 1, not one pooled 16 GiB GPU.

In a fresh Kaggle notebook:

```python
from pathlib import Path
import subprocess, sys

ROOT = Path("/kaggle/working")
PIPELINE = ROOT / "image-to-video-pipeline"
WAN = ROOT / "Wan2.2"
DATA = ROOT / "Edu-video-gen-dataset"

for url, folder in [
    ("https://github.com/ssoommeesshh/image-to-video-pipeline.git", PIPELINE),
    ("https://github.com/abhinavk0006/Wan2.2.git", WAN),
    ("https://github.com/abhinavk0006/Edu-video-gen-dataset.git", DATA),
]:
    if not folder.exists():
        subprocess.run(["git", "clone", url, str(folder)], check=True)
subprocess.run(["git", "-C", str(PIPELINE), "checkout", "feat/cpu-notebook-module"], check=True)
subprocess.run(["git", "-C", str(WAN), "checkout", "--detach", "78fb5ce"], check=True)
print("Pipeline:", subprocess.check_output(["git", "-C", str(PIPELINE), "rev-parse", "--short", "HEAD"], text=True).strip())
print("Wan:", subprocess.check_output(["git", "-C", str(WAN), "rev-parse", "--short", "HEAD"], text=True).strip())
```

Use the same dependencies as the teammate's successful salt-analysis notebook:

```python
%pip install -q -r /kaggle/working/Edu-video-gen-dataset/requirements.txt imageio-ffmpeg
%pip install -q 'diffusers>=0.31,<0.33' 'transformers>=4.49,<=4.51.3' 'accelerate>=1.1.1' easydict safetensors ftfy huggingface_hub imageio
```

Restart the kernel if pip replaced an already imported package. Confirm
`torch.cuda.device_count() >= 2`.

## Starting image and pilot

Add a Kaggle input image named `chloride_test_start.jpg`. It should show **one
clear test tube of salt solution already acidified with dilute nitric acid and
one silver-nitrate dropper directly above it**. The supplied broad rack photo
is suitable only if it visibly matches this exact first state. Keep one
camera angle. The second generated clip uses the first clip's decoded final
frame, even though an explanation card appears between them. The cards need
no extra input image or Wan call.

Run the pilot from a fresh output directory:

```python
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
images = list(Path("/kaggle/input").rglob("chloride_test_start.jpg"))
assert len(images) == 1, f"Expected one starting image, found {len(images)}"
image = images[0]
fixture = PIPELINE / "knowledge" / "chloride_two_clip_resident_pilot.json"
output = ROOT / "chloride_resident_run_01"
assert fixture.is_file() and not output.exists()

command = [
    sys.executable, str(PIPELINE / "main.py"),
    "--knowledge-file", str(fixture),
    "--initial-image", str(image),
    "--output-dir", str(output),
    "--experiment-name", "chloride_pilot",
    "--generator-script", str(PIPELINE / "kaggle_wan_resident.py"),
    "--generator-executable", sys.executable,
    "--generator-working-dir", str(PIPELINE),
    "--wan-repo-dir", str(WAN),
    "--wan-model-preset", "i2v-a14b",
    "--continuity-mode", "chain",
    "--require-resident-daemon",
    "--generator-arg=--lightning",
    f"--generator-arg=--knowledge-file={fixture}",
    "--generator-arg=--gpu0-memory-gib=6",
    "--generator-arg=--gpu1-memory-gib=10",
    "--generator-arg=--memory-telemetry",
]
subprocess.run(command, cwd=PIPELINE, check=True)
print(output / "chloride_pilot" / "final_video.mp4")
```

Expected files are `silver_nitrate.mp4`, `ammonia_dissolves.mp4`, three
short static card MP4s, and `final_video.mp4`. The first generated clip
should log `Converting high_noise_model` and `Converting low_noise_model`;
the next should log `INT8 cache hit` for both. `Wan session ready` should
appear only once. This proves the intended reuse path was exercised; record
actual per-clip elapsed time and inspect the MP4 for scientific and visual
accuracy.

If cache space is insufficient, the worker fails with the needed/free GiB
before it starts a clip without caching. Use a larger trusted temp volume
through `--generator-arg=--cache-dir=/path/to/cache` and a fresh output
directory. The cache contains Python-serialized model objects, so only use
files created by this worker; do not point it at an uploaded/untrusted cache.

The legacy pipeline skips existing clip MP4s by path. For a changed image,
prompt, model setting or worker version, always choose a new `--output-dir`.
The separate packaged `edu-video` runner has hash-based resume but does
not provide this Kaggle Wan worker path.

## Extending beyond this pilot

For the teammate's six-segment salt analysis, copy its experiment JSON into
the original pipeline and inspect every clip's `continuity_required`,
`reference_clip`, and `input_frame_path`. This original pipeline now reads
those fields. Give each independent chemical test a scene-specific starting
image. A continuation clip may point to an earlier clip via
`reference_clip`; static explanation cards do not disturb that reference.
Every generated clip should stay at or below the default 2-second cap. Put
longer explanations in `is_title` cards, which are rendered on CPU.

The current two-clip pilot follows the visible chloride test described in the
[NCERT Class XI chemistry laboratory manual, Unit 7](https://www.ncert.nic.in/pdf/publication/sciencelaboratorymanuals/classXI/chemistry/kelm207.pdf).
Have a chemistry reviewer approve the exact initial image, observations and
manual-linked wording before presenting the output as a verified lesson.
