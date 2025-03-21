from huggingface_hub import snapshot_download
snapshot_download(
	repo_id="leafspark/Llama-3.2-11B-Vision-Instruct-GGUF", 
	local_dir="/home/chwenjun225_laptop/.llama/checkpoints/Llama-3.2-11B-Vision-Instruct-GGUF", 
	local_dir_use_symlinks=False, 
	revision="main"
)
