from huggingface_hub import snapshot_download
snapshot_download(
	repo_id="meta-llama/Llama-3.2-1B-Instruct", 
	local_dir="/home/chwenjun225_laptop/.llama/checkpoints/Llama-3.2-1B-Instruct", 
	local_dir_use_symlinks=False, 
	revision="main"
)
