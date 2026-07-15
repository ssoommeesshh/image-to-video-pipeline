#!/usr/bin/env python3
"""Pre-download models and check API token compatibility before GPU execution."""

from __future__ import annotations
import os
import sys
from pathlib import Path

def download_qwen():
    print("=== Downloading Qwen2.5-3B-Instruct weights to local Hugging Face cache ===")
    try:
        from huggingface_hub import snapshot_download
        print("Starting download (this will cache the model files)...")
        # Downloads model files to ~/.cache/huggingface/hub/
        path = snapshot_download(repo_id="Qwen/Qwen2.5-3B-Instruct", ignore_patterns=["*.gguf"])
        print(f"Success! Qwen2.5-3B-Instruct weights cached at: {path}\n")
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading Qwen weights: {e}", file=sys.stderr)
        sys.exit(1)

def test_flux_connection():
    print("=== Testing HF Token & FLUX.1-dev API Connection ===")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN environment variable is not set.", file=sys.stderr)
        print("Hugging Face API calls to FLUX.1-dev will fail. Please run:\n")
        print("  export HF_TOKEN='your_token_here'\n")
        print("Skipping FLUX API connection check.\n")
        return

    print(f"Token detected (starts with: {token[:8]}...)")
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model="black-forest-labs/FLUX.1-dev", token=token)
        print("Sending dummy test request to black-forest-labs/FLUX.1-dev API...")
        client.text_to_image("a simple scientific glass beaker")
        print("Success! FLUX.1-dev API resolved successfully.\n")
    except Exception as e:
        print(f"Error connecting to FLUX API: {e}", file=sys.stderr)
        print("Please check that your HF_TOKEN is correct and has access permission.\n", file=sys.stderr)

def main():
    download_qwen()
    test_flux_connection()
    print("Preflight checks complete. Ready to swap to GPU environment!")

if __name__ == "__main__":
    main()
