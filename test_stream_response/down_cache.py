from huggingface_hub import snapshot_download

# Download and cache the repo
cache_path = snapshot_download("Qwen/Qwen2.5-1.5B")

# print(f"Repository is cached at: {cache_path}")
