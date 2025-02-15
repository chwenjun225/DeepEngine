from huggingface_hub import snapshot_download
model_id="openbmb/MiniCPM-o-2_6"
snapshot_download(
    repo_id=model_id, 
    local_dir="/home/chwenjun225/Projects/Foxer/models/openbmb/MiniCPM-o-2_6",
    local_dir_use_symlinks=False, 
    revision="main"
)
