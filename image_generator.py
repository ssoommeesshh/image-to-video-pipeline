from __future__ import annotations
import os
from pathlib import Path


class ImageGenerationError(RuntimeError):
    pass


def generate_image_from_prompt(prompt: str, output_path) -> Path:
    import os
    token = os.environ.get("HF_TOKEN")
    print(f"[debug] token inside function: {str(token)[:10] if token else None}")
    if not token:
        raise ImageGenerationError("HF_TOKEN not set.")
    try:
        from huggingface_hub import InferenceClient
    except ImportError as e:
        raise ImageGenerationError("Run: pip install huggingface_hub") from e
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = InferenceClient(model="stabilityai/stable-diffusion-xl-base-1.0", token=token)
        image = client.text_to_image(prompt)
        image.save(output_path)
    except Exception as e:
        raise ImageGenerationError(f"Image generation failed: {e}") from e
    if not output_path.exists():
        raise ImageGenerationError(f"No file written: {output_path}")
    return output_path
