# Use official PyTorch developer image with CUDA 12.1 runtime
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set non-interactive mode for apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (FFmpeg for stitching, curl, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install the exact packages required by Wan2.2 & the pipeline wrapper
RUN pip install --no-cache-dir \
    diffusers>=0.31.0 \
    transformers>=4.49.0,<=4.51.3 \
    tokenizers>=0.20.3 \
    accelerate>=1.1.1 \
    opencv-python>=4.9.0.80 \
    tqdm \
    imageio[ffmpeg] \
    imageio-ffmpeg \
    easydict \
    ftfy \
    dashscope \
    requests \
    pillow

# Attempt to install flash-attn (using pre-built wheels if possible, or falls back to compilation)
# If it fails to compile due to hardware environment matching, PyTorch falls back to SDPA.
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Copy the pipeline codebase
COPY . /app/image-to-video-pipeline

# Set PYTHONPATH environment variable so imports resolve correctly inside the container
ENV PYTHONPATH="/app/image-to-video-pipeline:${PYTHONPATH}"
ENV WAN_REPO_DIR="/app/Wan2.2"

# Default entrypoint runs main.py
ENTRYPOINT ["python", "/app/image-to-video-pipeline/main.py"]
