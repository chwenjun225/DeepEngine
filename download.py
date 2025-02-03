from huggingface_hub import snapshot_download
model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
snapshot_download(
    repo_id=model_id, 
    local_dir="/home/kali/Projects/Models/DeepSeek-R1-Distill-Qwen-1.5B",
    local_dir_use_symlinks=False, 
    revision="main"
)