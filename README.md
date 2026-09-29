# Educational video module: CPU workflow

Phases 1?4 provide a versioned clip-plan boundary, installable Python package, noninteractive CPU runner, and an end-to-end notebook. Start here for dataset integration. The older Wan/GPU commands below remain a separate legacy entry point.

## Install

Python 3.10+ is required. Python 3.12 is the tested baseline. Clone both repositories alongside each other:

```bash
git clone --branch feat/cpu-notebook-module https://github.com/ssoommeesshh/image-to-video-pipeline.git
git clone --branch feat/pdf-grounded-rag-review https://github.com/ssoommeesshh/Edu-video-gen-dataset.git chemistry-dataset
cd image-to-video-pipeline
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[rag,notebook]"
python -m ipykernel install --user --name edu-video --display-name "Edu video CPU"
```

Open `notebooks/cpu_workflow.ipynb` in Jupyter, VS Code, or a hosted notebook and select the installed environment. Set `EDU_DATASET_DIR` if the dataset checkout is elsewhere. The notebook uses `chem_12_101` as an explicit selection, exports its existing evidence status, assigns synthetic test cards, generates three static clips, and displays a stitched MP4. It also checks blocked-record rejection and cache reuse.

For a hosted notebook, clone the repositories into its filesystem and use `%pip install -e "/path/to/image-to-video-pipeline[rag,notebook]"`. No CUDA, PyTorch, model downloads, cloud credentials, or paid calls are required. FFmpeg is supplied by `imageio-ffmpeg` on supported platforms. Installation needs network access; execution with local assets does not.

## Python API

```python
from edu_video import Catalog, VideoPipeline

catalog = Catalog("../chemistry-dataset")
candidates = catalog.search("boiling point organic compound")
plan = catalog.plan("chem_12_101")  # Explicit choice; evidence gate still applies.

pipeline = VideoPipeline(generator="dummy", output_dir="runs/boiling_point")
result = pipeline.run(plan, scene_images={
    "chem_12_101_scene": {
        "path": "images/setup.png",
        "provenance": "User-supplied setup photograph",
        "review_status": "not_reviewed",
    }
})
print(result.video_path, result.manifest_path)
```

`Catalog` calls the dataset planner using the current Python interpreter. It requires the `rag` extra and a compatible dataset checkout. A video-only consumer needs only `pip install .` and an exported plan plus images; it does not need the catalog, PDFs, or retrieval index.

## CLI

```bash
python ../chemistry-dataset/scripts/plan_experiment.py --query "boiling point organic compound"
python ../chemistry-dataset/scripts/plan_experiment.py --experiment-id chem_12_101 --output runs/plan.json
edu-video --plan runs/plan.json --initial-image images/setup.png --preflight-only
edu-video --plan runs/plan.json --initial-image images/setup.png --output-dir runs/demo
```

For multiple scenes, pass `--scene-images scene_images.json`, a map of scene IDs to image paths or image records. CLI image-map paths are relative to that JSON file; embedded plan image paths are relative to the plan file; Python API overrides and `--initial-image` are relative to the caller's working directory. New scenes require an image; continuing clips always use the actual last decoded frame. Missing or invalid images are reported together before any generation. No automatic image generation or interactive wait occurs.

`--generator local --generator-script /path/to/script.py` accepts the existing local generator CLI arguments. The default is always `dummy`. Local adapters must emit matching video dimensions/frame rates and honor requested duration. Hosted adapters and deployment infrastructure are a later phase.

## Contract and reproducibility

`src/edu_video/schemas/clip_plan.schema.json` is the distributed copy of the dataset-owned v1 schema. Both copies must change together. Stable clip IDs, ordered steps, scene transitions, passage IDs, source references, content hashes, and review states travel in the plan. Query-only responses and blocked records are decision documents, not executable plans. The video runner validates the exported contract; it does not independently re-verify PDF passages.

Per-clip duration defaults to one second labeled `provisional_cpu_test`. Override durations with the dataset planner's `--durations durations.json` (map clip IDs to seconds). These values are integration settings, not calibrated motion timings. The manifest separately records requested, generated, and retained durations; v1 retains the full clip and rejects duration mismatches instead of silently trimming. Prompt/video tuning and provider-specific durations remain future work.

Every run stores `clip_plan.json`, `run_manifest.json`, individual MP4s, actual final frames, and `final_video.mp4`. Resume requires matching request and video hashes; the request includes prompts, input image content, clip configuration, and generator identity. A corrupted or changed artifact is regenerated. A failed run records the error and completed clips. Use a separate output directory for simultaneous jobs. An abnormal process kill may leave `.run.lock`; remove it only after confirming no process owns that output directory.

Dummy videos are labeled `integration_output`; they establish software behavior only. Source status, image review, and video visual review remain separate. The included `cpu_plan.json` fixture has no scientific source approval and runs only with the dummy provider. Complex physics setups can use supplied stock/reference images with provenance; actual apparatus and motion accuracy need later visual testing.

## Tests

```bash
python -m pip install -e ".[test,rag]"
python -m pytest tests -q
python -m pytest ../chemistry-dataset/tests -q
python -m build
```

Tests cover selection/rejection, evidence gating, schema invariants, image preflight, scene continuity, actual last-frame extraction, stitched frame count, cache invalidation, corruption, and failure recovery. The module does not promise scientific correctness from a passing CPU test.

---

## Legacy GPU pipeline

# Scientific Image-to-Video Generation Pipeline

An orchestrator pipeline that generates multi-clip videos from static initial images and structured scientific experiment JSON logs. It ensures visual continuity across transitions by feeding the final frame of the preceding clip as the initial frame of the next.

Designed to run in resource-constrained cloud environments (e.g., 62GB RAM / 46GB L40S GPU) using memory-efficient execution strategies.

---

## 🚀 Key Features

* **Multi-Clip Visual Continuity**: Decodes and extracts the exact final frame of each video clip to use as the starting frame of the next.
* **Resume Support**: Interrupted runs automatically skip already generated clips, allowing seamless pipeline recovery.
* **Persistent Daemon Architecture**: Keeps the heavy generative model resident in a background JSON-RPC daemon process to avoid reloading model parameters between sequential clips.
* **Robust Frame Extraction**: Employs an `ffmpeg` seek-to-end strategy with frame overwriting (`-update 1`) to guarantee pixel-perfect extraction of the absolute last frame.
* **Scientific Prompt Builder**: Dynamically constructs image and motion prompts using structured scientific variables (states, constraints, and stop conditions).

---

## 🛠️ Installation & Setup

Set up the workspace and download dependencies using the provided configuration script:

```bash
# Run minimal installation script
chmod +x lightning_setup_minimal.sh
./lightning_setup_minimal.sh
```

---

## 🏃 Execution

Run the pipeline by providing the path to your scientific experiment JSON log, the generator wrapper script, the Wan 2.2 model directory, and the initial seed image:

```bash
python main.py \
  --knowledge-file knowledge/acid_base_titration.json \
  --generator-script wan_local_wrapper.py \
  --wan-repo-dir /teamspace/studios/this_studio/Wan2.2 \
  --wan-model-preset i2v-a14b \
  --initial-image outputs/acid_base_titration/input/input_image.jpg
```

### Parameters:
* `--knowledge-file`: Absolute path to the scientific experiment JSON configuration.
* `--generator-script`: Script wrapping the underlying generator (defaults to `wan_local_wrapper.py`).
* `--wan-repo-dir`: Path to the cloned `Wan2.2` repository containing your model code.
* `--wan-model-preset`: The model preset to run (`i2v-a14b` or `ti2v-5b`).
* `--initial-image`: Starting image for the very first clip.

---

## 🧠 Memory Optimizations

To run the **14B Image-to-Video** model without crashing under standard hardware limits:
1. **Lazy Weights Loading**: High-noise and low-noise models are loaded sequentially on demand rather than all at startup.
2. **CPU-to-GPU Memory Unloading**: The inactive model is explicitly offloaded back to the CPU and garbage collected before the active model is loaded, keeping memory overhead within physical RAM boundaries.
3. **Low-Precision Execution**: Models are converted to `bfloat16` and run with `offload_model=True` to minimize VRAM footprint.
