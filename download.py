from huggingface_hub import snapshot_download
model_id="meta-llama/Llama-3.2-1B-Instruct"
snapshot_download(
    repo_id=model_id, 
    local_dir="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct",
    local_dir_use_symlinks=False, 
    revision="main"
)
