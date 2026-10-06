"""Resident JSON-lines worker for the pinned two-T4 Wan 2.2 Kaggle fork.

The worker keeps WanI2V and its VAE alive for an experiment, encodes all clip
prompts with T5 once, and caches each LoRA-merged INT8 expert in /kaggle/temp.
The two experts are still staged one at a time: they cannot both fit alongside
activations on the tested pair of T4s. Only locally created cache files are
deserialized. See docs/kaggle_resident_handoff.md before a GPU run.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import MethodType


def duration_to_frames(seconds: float, fps: int = 16) -> int:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("clip duration must be finite and positive")
    return max(17, 4 * math.ceil((seconds * fps - 1) / 4) + 1)


def experiment_prompts(knowledge_file: Path) -> set[str]:
    from knowledge_loader import load_experiment_from_json
    from prompt_builder import PromptBuilder

    experiment = load_experiment_from_json(str(knowledge_file))
    builder = PromptBuilder()
    prompts = set()
    for clip in experiment.clips:
        if clip.metadata.get("is_title"):
            continue
        bundle = builder.build_prompt_bundle(experiment, clip)
        if not bundle.motion_prompt.strip():
            raise ValueError(f"{clip.name}: empty motion prompt")
        prompts.add(bundle.motion_prompt)
        if bundle.negative_prompt:
            prompts.add(bundle.negative_prompt)
    if not prompts:
        raise ValueError("The experiment has no generated clips")
    return prompts


class CachedTextEncoder:
    """Serve the text embeddings computed before the large T5 is released."""

    def __init__(self, values):
        self.values = values

    def __call__(self, prompts, device):
        if len(prompts) != 1:
            raise ValueError("Expected one Wan prompt at a time")
        try:
            value = self.values[prompts[0]]
        except KeyError as exc:
            raise ValueError("Prompt differs from the prepared experiment; restart the worker") from exc
        return [value]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--wan-repo-dir", type=Path, required=True)
    parser.add_argument("--knowledge-file", type=Path, required=True)
    parser.add_argument("--model-preset", choices=["i2v-a14b"], default="i2v-a14b")
    parser.add_argument("--lightning", action="store_true", required=True)
    parser.add_argument("--gpu0-memory-gib", type=float, default=6.0)
    parser.add_argument("--gpu1-memory-gib", type=float, default=10.0)
    parser.add_argument("--max-area", type=int, default=480 * 832)
    parser.add_argument("--max-clip-seconds", type=float, default=2.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("/kaggle/temp/wan22-int8-cache"))
    parser.add_argument("--memory-telemetry", action="store_true")
    args = parser.parse_args()
    if not args.daemon:
        parser.error("This worker requires --daemon and --require-resident-daemon in main.py")
    if not args.knowledge_file.is_file():
        parser.error(f"Missing experiment JSON: {args.knowledge_file}")
    if not (args.wan_repo_dir / "quantization" / "kaggle_generate_video.py").is_file():
        parser.error(f"Missing teammate Wan runner: {args.wan_repo_dir}")
    if args.max_area <= 0 or args.max_clip_seconds <= 0:
        parser.error("max-area and max-clip-seconds must be positive")
    return args


class ResidentWanSession:
    def __init__(self, args):
        import torch
        from PIL import Image

        self.torch = torch
        self.Image = Image
        self.args = args
        self.repo = args.wan_repo_dir.resolve()
        sys.path.insert(0, str(self.repo))
        sys.path.insert(0, str(self.repo / "quantization"))
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("This pinned quantized runner needs Kaggle's two-T4 accelerator")
        if args.gpu0_memory_gib <= 0 or args.gpu1_memory_gib <= 0:
            raise ValueError("Both GPU memory budgets must be positive")

        import kaggle_generate_video as source
        from kaggle_single_expert import (
            dispatch_wan_model_across_gpus,
            load_fp8_scaled_wan_checkpoint,
            tensor_storage_bytes,
        )
        from wan.configs import WAN_CONFIGS
        from wan.image2video import WanI2V
        from wan.utils.utils import save_video
        import copy

        self.source = source
        self.dispatch = dispatch_wan_model_across_gpus
        self.convert = load_fp8_scaled_wan_checkpoint
        self.storage_bytes = tensor_storage_bytes
        self.save_video = save_video
        self.config = copy.deepcopy(WAN_CONFIGS["i2v-A14B"])
        self.config.param_dtype = torch.float16
        self.config.t5_dtype = torch.bfloat16
        self.config.sample_shift = 5.0
        self.config.sample_steps = 4
        self.config.sample_guide_scale = (1.0, 1.0)

        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        worker_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        self.cache_dir = args.cache_dir / f"{revision[:12]}-{worker_hash}-lightning"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info("INT8 expert cache: %s", self.cache_dir)

        torch.cuda.set_device(0)
        if args.memory_telemetry:
            source.log_cuda_memory("startup")
        shared = [
            source.download(source.HF_I2V_REPO, self.config.t5_checkpoint, source.MODELS_DIR),
            source.download(source.HF_I2V_REPO, self.config.vae_checkpoint, source.MODELS_DIR),
        ]
        self.model = WanI2V(
            self.config, str(source.MODELS_DIR), device_id=0, t5_cpu=True,
            init_on_cpu=True, convert_model_dtype=False,
        )
        prompts = experiment_prompts(args.knowledge_file)
        prompts.add(self.model.sample_neg_prompt)
        encoder = self.model.text_encoder
        encoded = {}
        for prompt in sorted(prompts):
            encoded[prompt] = encoder([prompt], torch.device("cpu"))[0].detach().cpu()
        self.model.text_encoder = CachedTextEncoder(encoded)
        del encoder
        gc.collect()
        for path in shared:
            path.unlink(missing_ok=True)
        self.model.release_t5_after_encode = False
        self.model._staged_expert_loading = True
        self.model._prepare_model_for_timestep = MethodType(self.prepare_expert, self.model)
        logging.info("Wan session ready: T5 encoded %d prompts once; VAE resident", len(encoded))
        if args.memory_telemetry:
            source.log_cuda_memory("session-ready")

    def prepare_expert(self, model, timestep, boundary, offload_model):
        del offload_model
        torch = self.torch
        expert_name = (
            "high_noise_model" if timestep.item() >= boundary else "low_noise_model"
        )
        other_name = (
            "low_noise_model" if expert_name == "high_noise_model" else "high_noise_model"
        )
        expert = getattr(model, expert_name)
        if expert is not None:
            return expert
        other = getattr(model, other_name)
        if other is not None:
            setattr(model, other_name, None)
            del other
            gc.collect()
            torch.cuda.empty_cache()

        cache = self.cache_dir / f"{expert_name}.pt"
        if cache.is_file():
            logging.info("INT8 cache hit: %s", expert_name)
            # This file is created by this worker in its own Kaggle temp directory.
            expert = torch.load(cache, map_location="cpu", weights_only=False)
        else:
            checkpoint = self.source.download(
                self.source.HF_COMFY_REPO,
                self.source.COMFY_PREFIX + self.source.EXPERT_FILES[expert_name],
                self.source.MODELS_DIR,
            )
            adapter = self.source.download(
                self.source.HF_LIGHTNING_REPO,
                f"{self.source.LIGHTNING_ADAPTER_DIR}/{expert_name}.safetensors",
                self.source.MODELS_DIR / "lightning",
            )
            try:
                logging.info("Converting %s to LoRA-merged INT8 once", expert_name)
                expert, _ = self.convert(checkpoint, lora_file=adapter)
                needed = int(self.storage_bytes(expert) * 1.1) + 2**30
                free = shutil.disk_usage(self.cache_dir).free
                if free < needed:
                    raise RuntimeError(
                        f"Not enough cache space for {expert_name}: "
                        f"need {needed / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB"
                    )
                temporary = cache.with_suffix(".tmp")
                torch.save(expert, temporary)
                temporary.replace(cache)
                logging.info("INT8 cache written: %s (%.1f GiB)", cache, cache.stat().st_size / 2**30)
            finally:
                checkpoint.unlink(missing_ok=True)
                adapter.unlink(missing_ok=True)
                cache.with_suffix(".tmp").unlink(missing_ok=True)
        expert, device_map = self.dispatch(
            expert, [0, 1], max_memory_gib=8.0,
            max_memory_gib_by_device={
                0: self.args.gpu0_memory_gib, 1: self.args.gpu1_memory_gib,
            },
        )
        logging.info("Expert %s placement: %s", expert_name, device_map)
        if self.args.memory_telemetry:
            self.source.log_cuda_memory(f"after-{expert_name}")
        setattr(model, expert_name, expert)
        return expert

    def run_clip(self, task):
        torch = self.torch
        prompt = str(task["prompt"])
        negative = str(task.get("negative_prompt", ""))
        if prompt not in self.model.text_encoder.values:
            raise ValueError("Unprepared motion prompt; restart with matching knowledge JSON")
        if negative and negative not in self.model.text_encoder.values:
            raise ValueError("Unprepared negative prompt; restart with matching knowledge JSON")
        image_path = Path(task["input_image"]).resolve()
        output_path = Path(task["output_video"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        seconds = float(task["clip_duration"])
        if seconds > self.args.max_clip_seconds:
            raise ValueError(
                f"{seconds:g}s exceeds the configured {self.args.max_clip_seconds:g}s clip cap"
            )
        frames = duration_to_frames(seconds)
        clip_name = str(task.get("clip_name") or output_path.name)
        seed = int.from_bytes(hashlib.sha256(clip_name.encode()).digest()[:4], "big")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Generating %s: %d frames, input %s", clip_name, frames, image_path)
        with self.Image.open(image_path) as image:
            video = self.model.generate(
                input_prompt=prompt, n_prompt=negative,
                img=image.convert("RGB"), max_area=self.args.max_area,
                frame_num=frames, sample_solver="euler", sampling_steps=4,
                guide_scale=(1.0, 1.0), seed=seed, shift=5.0,
                offload_model=False,
            )
        self.save_video(
            tensor=video[None], save_file=str(output_path),
            fps=self.config.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
        )
        del video
        gc.collect()
        torch.cuda.empty_cache()
        if self.args.memory_telemetry:
            self.source.log_cuda_memory(f"after-{clip_name}")
        if not output_path.is_file() or not output_path.stat().st_size:
            raise RuntimeError(f"No video written at {output_path}")
        logging.info("Completed %s: %s", clip_name, output_path)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Stdout is reserved for the pipeline's READY and one JSON reply per clip.
    with redirect_stdout(sys.stderr):
        worker = ResidentWanSession(args)
    print("READY resident", flush=True)
    for line in sys.stdin:
        try:
            task = json.loads(line)
            if task.get("action") == "exit":
                return
            with redirect_stdout(sys.stderr):
                worker.run_clip(task)
            print(json.dumps({"status": "success"}), flush=True)
        except Exception as exc:
            logging.exception("Resident Wan clip failed")
            print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
            return


if __name__ == "__main__":
    main()
