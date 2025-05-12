from huggingface_hub import snapshot_download
if __name__ == "__main__":
	snapshot_download(
		repo_id="unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF", 
		local_dir="/home/chwenjun225_laptop/.llama/checkpoints/DeepSeek-R1-Distill-Qwen-7B-GGUF", 
		revision="main"
	)
### Những mô hình đã tải xuống:
###		unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
###		unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF
###		unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF
### 	deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
### 	leafspark/Llama-3.2-11B-Vision-Instruct-GGUF
