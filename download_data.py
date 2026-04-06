from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YashS/Skyline",
    repo_type="dataset",
    local_dir=".",
)

print("Dataset downloaded: skyline_1024/ and skyline_tiny/ are ready.")
