import os
import subprocess
from pathlib import Path

import urllib.request

def download_file(repo_id: str, rpath: str, local_dir: Path, token: str):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{rpath}"
    dest_path = local_dir / rpath
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Hugging Face only places files in the target directory once they are 100% complete
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Skipping {rpath} (already fully downloaded: {dest_path.stat().st_size} bytes)")
        return
            
    print(f"\n--- Downloading {rpath} ---")
    
    cmd = [
        "curl", "-f", "-L", "-C", "-",
        "-H", f"Authorization: Bearer {token}",
        url,
        "-o", str(dest_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {rpath}: {e}")
        raise e

def main():
    repo_id = "Wan-AI/Wan2.2-I2V-A14B"
    local_dir = Path("/teamspace/studios/this_studio/Wan2.2/Wan2.2-I2V-A14B")
    token = os.environ.get("HF_TOKEN") 
    
    # List of all files in the model repository
    files = [
        "configuration.json",
        "Wan2.1_VAE.pth",
        
        # Text Encoder
        "google/umt5-xxl/special_tokens_map.json",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
        
        
        # High Noise Model
        "high_noise_model/config.json",
        "high_noise_model/diffusion_pytorch_model.safetensors.index.json",
        "high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
        "high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
        "high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
        "high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
        "high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
        
        # Low Noise Model
        "low_noise_model/config.json",
        "low_noise_model/diffusion_pytorch_model.safetensors.index.json",
        "low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
        "low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
        "low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
        "low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
        "low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
    ]
    
    print(f"Resuming sequential curl download for {repo_id}...")
    for f in files:
        download_file(repo_id, f, local_dir, token)
    print("\nAll files successfully downloaded and verified!")

if __name__ == "__main__":
    main()
