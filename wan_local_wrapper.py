"""Local wrapper around the Wan 2.2 generation script.

This script is the bridge between the pipeline's generic generator interface and
Wan's repo-specific `generate.py` command.

For the first practical test, use the 5B model preset. It is much easier to run
on a smaller Lightning AI GPU than the A14B image-to-video model.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in the runtime
    torch = None


class WanWrapperError(RuntimeError):
    """Raised when the Wan wrapper cannot build or run the local command."""


@dataclass(slots=True)
class WanPreset:
    """Convenience settings for a Wan generation mode."""

    task: str
    size: str
    model_dir_name: str
    extra_flags: list[str]
    sample_fps: int = 16
    output_flag: str = "--save_file"


WAN_PRESETS: dict[str, WanPreset] = {
    "i2v-a14b": WanPreset(
        task="i2v-A14B",
        size="832*480",
        model_dir_name="Wan2.2-I2V-A14B",
        sample_fps=16,
        extra_flags=["--offload_model", "True", "--convert_model_dtype", "--t5_cpu"],
    ),
    "ti2v-5b": WanPreset(
        task="ti2v-5B",
        size="1280*704",
        model_dir_name="Wan2.2-TI2V-5B",
        sample_fps=24,
        extra_flags=["--offload_model", "True", "--convert_model_dtype", "--t5_cpu"],
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Wan 2.2 locally through a stable wrapper.")
    parser.add_argument("--input-image", default=None, help="Path to the input image.")
    parser.add_argument("--output-video", default=None, help="Path to write the generated video.")
    parser.add_argument("--prompt", default=None, help="Motion prompt passed to Wan.")
    parser.add_argument("--clip-duration", default="1.0", help="Clip duration in seconds.")
    parser.add_argument("--negative-prompt", default="", help="Optional negative prompt.")
    parser.add_argument("--clip-name", default="", help="Optional clip name for logging.")
    parser.add_argument("--size", default=None, help="Override the default resolution/size for the model preset.")
    parser.add_argument("--daemon", action="store_true", default=False, help="Run in daemon mode to keep weights loaded across generations.")

    parser.add_argument(
        "--wan-repo-dir",
        default=None,
        help="Path to the local Wan 2.2 repository checkout. If omitted, set WAN_REPO_DIR.",
    )
    parser.add_argument(
        "--model-preset",
        choices=sorted(WAN_PRESETS.keys()),
        default="ti2v-5b",
        help="Model preset to run. Use ti2v-5b for the first test.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter to use for the underlying Wan command.",
    )
    parser.add_argument(
        "--extra-wan-arg",
        action="append",
        default=[],
        help="Extra argument passed through to Wan's generate.py. Repeat as needed.",
    )
    return parser


def _build_generate_command(args: argparse.Namespace) -> list[str]:
    preset = WAN_PRESETS[args.model_preset]

    wan_repo_dir_value = args.wan_repo_dir or os.environ.get("WAN_REPO_DIR")
    if not wan_repo_dir_value:
        raise WanWrapperError("Provide --wan-repo-dir or set WAN_REPO_DIR")

    wan_repo_dir = Path(wan_repo_dir_value)
    generate_py = wan_repo_dir / "generate.py"
    if not generate_py.exists():
        raise WanWrapperError(f"Could not find generate.py at: {generate_py}")

    model_dir = wan_repo_dir / preset.model_dir_name
    if not model_dir.exists():
        raise WanWrapperError(f"Could not find model directory at: {model_dir}")

    size = args.size or preset.size

    # Map clip duration to frame number: frame_num should be of the form 4n + 1
    try:
        duration = float(args.clip_duration)
        raw_frames = duration * preset.sample_fps
        n = round((raw_frames - 1) / 4)
        if n < 1:
            n = 1
        frame_num = 4 * n + 1
    except (ValueError, TypeError):
        frame_num = 81

    command = [
        args.python_executable,
        str(generate_py),
        "--task",
        preset.task,
        "--size",
        size,
        "--frame_num",
        str(frame_num),
        "--ckpt_dir",
        str(model_dir),
    ]

    if not args.daemon:
        command.extend([
            "--image",
            str(Path(args.input_image).resolve()),
            "--prompt",
            args.prompt,
        ])
    else:
        command.append("--daemon")

    command.extend(preset.extra_flags)

    if args.negative_prompt:
        command.extend(["--n_prompt", args.negative_prompt])

    command.extend(args.extra_wan_arg)

    if not args.daemon:
        command.extend([preset.output_flag, str(Path(args.output_video).resolve())])
    return command


def _preflight_gpu_check(model_preset: str) -> None:
    """Fail fast when the selected GPU cannot reasonably run the preset."""

    if torch is None:
        raise WanWrapperError("PyTorch is not available in this environment")

    if not torch.cuda.is_available():
        raise WanWrapperError(
            "CUDA GPU is required for Wan inference. Select a GPU runtime before running this preset."
        )

    device_index = torch.cuda.current_device()
    total_memory_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)

    if model_preset == "i2v-a14b" and total_memory_gb < 38:
        raise WanWrapperError(
            f"Wan I2V-A14B needs roughly 40GB VRAM for the single-GPU path; detected {total_memory_gb:.1f}GB."
        )

    if model_preset == "ti2v-5b" and total_memory_gb < 24:
        raise WanWrapperError(
            f"Wan TI2V-5B usually expects at least 24GB VRAM; detected {total_memory_gb:.1f}GB."
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.daemon:
        if not args.input_image or not args.output_video or not args.prompt:
            print("Error: --input-image, --output-video, and --prompt are required when not running in daemon mode.", file=sys.stderr)
            return 1

        output_path = Path(args.output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_image_path = Path(args.input_image)
        if not input_image_path.exists():
            print(f"Input image not found: {input_image_path}", file=sys.stderr)
            return 1

    try:
        _preflight_gpu_check(args.model_preset)
        command = _build_generate_command(args)
    except WanWrapperError as error:
        print(str(error), file=sys.stderr)
        return 1

    repo_dir_value = args.wan_repo_dir or os.environ.get("WAN_REPO_DIR")
    cwd = str(Path(repo_dir_value)) if repo_dir_value else None

    if args.daemon:
        import json
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                cwd=cwd
            )
        except Exception as error:
            print(f"Could not start Wan daemon: {error}", file=sys.stderr)
            return 1

        # Read READY message from subprocess
        ready_line = proc.stdout.readline()
        if ready_line.strip() != "READY":
            print(f"Error starting daemon, expected READY, got: {ready_line}", file=sys.stderr)
            proc.kill()
            return 1

        sys.stdout.write("READY\n")
        sys.stdout.flush()

        while True:
            line = sys.stdin.readline()
            if not line:
                break

            # Send task to generate.py
            proc.stdin.write(line)
            proc.stdin.flush()

            # Wait for response from generate.py
            response = proc.stdout.readline()
            if not response:
                sys.stdout.write(json.dumps({"status": "error", "error": "Daemon process terminated unexpectedly"}) + "\n")
                sys.stdout.flush()
                break

            sys.stdout.write(response)
            sys.stdout.flush()

            try:
                task = json.loads(line.strip())
                if task.get("action") == "exit":
                    break
            except Exception:
                pass

        proc.wait()
        return 0

    try:
        subprocess.run(command, check=True, cwd=cwd, text=True)
    except FileNotFoundError as error:
        print(f"Could not start Wan command: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"Wan generation failed: {error}", file=sys.stderr)
        return 1

    if not output_path.exists():
        print(f"Wan finished but no output file was created: {output_path}", file=sys.stderr)
        return 1

    print(f"Wan generation complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())