import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

def download():
    repo_id = "Wan-AI/Wan2.2-I2V-A14B"
    local_dir = "/teamspace/studios/this_studio/Wan2.2/Wan2.2-I2V-A14B"
    token = os.environ.get("HF_TOKEN")
    
    print(f"Starting download of {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
        max_workers=4,
        ignore_patterns=["Wan2.1_VAE.pth"]
    )
    print("Download complete!")

if __name__ == "__main__":
    download()
