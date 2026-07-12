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
