from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Lusmse/pourLeftCereal",
    repo_type="dataset",
    local_dir="pourLeftCereal",
    local_dir_use_symlinks=False,
)