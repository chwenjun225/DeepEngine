# BASH CLI

# llama_cpp runserver --support tool call for deepseek-r1 successfully 
# Either "json_schema" or "grammar" can be specified, but not both'
/home/chwenjun225/projects/DeepEngine/third_3rdparty/llama.cpp-b4641/build/bin/llama-server \
	--alias tranvantuan_research\
	--model /home/chwenjun225/.llama/checkpoints/Llama-3.2-1B-Instruct/gguf/Llama-3.2-1B-Instruct-F32.gguf \
	--host 127.0.0.1 \
	--port 2026 \
	--flash-attn \
	--no-webui \
	--ubatch-size 1024 \
	--ctx-size 4096 \
	--n-gpu-layers 999 \
	--timeout 30 \
	--temp 0.1 \
	--top-p 0.1 \
	--top-k 1 
# llama.cpp build project 
# CUDA
cmake -B build -DGGML_CUDA=ON 
cmake --build build --config Release 
# CPU
cmake -B build 
cmake --build build --config Release 

# Absolute path of
# 	base model: /home/chwenjun225/.llama/checkpoints/MiniCPM-o-2_6-gguf/Model-7.6B-Q4_K_M.gguf
# 	multi-modal-proj: /home/chwenjun225/.llama/checkpoints/MiniCPM-o-2_6-gguf/mmproj-model-f16.gguf

# Launch multi-modal model server 
python /home/chwenjun225/projects/DeepEngine/third_3rdparty/MiniCPM-o-main/web_demos/minicpm-o_2.6/model_server.py

# Run in interactive mode with ./llama-minicpmv-cli
/home/chwenjun225/projects/DeepEngine/third_3rdparty/MiniCPM-o-main/llama.cpp/build/bin/llama-minicpmv-cli \
	--flash-attn \
	--model /home/chwenjun225/.llama/checkpoints/MiniCPM-o-2_6-gguf/Model-7.6B-Q4_K_M.gguf \
	--mmproj /home/chwenjun225/.llama/checkpoints/MiniCPM-o-2_6-gguf/mmproj-model-f16.gguf \
	--ctx-size 4096 \
	--temp 0.1 \
	--top-p 0.1 \
	--top-k 1 \
	--repeat-penalty 1.05 \
	--image /home/chwenjun225/projects/DeepEngine/third_3rdparty/MiniCPM-o-main/images/cong_nhan_tren_day_chuyen_san_xuat1.jpg \
	--prompt "Describe the image in detail."

# Host model on server with ./llama-server
/home/chwenjun225/projects/DeepEngine/third_3rdparty/MiniCPM-o-main/llama.cpp/build/bin/llama-server \
	--flash-attn \
	--model /home/chwenjun225/.llama/checkpoints/MiniCPM-o-2_6-gguf/Model-7.6B-Q4_K_M.gguf \
	--n-gpu-layers 999 \
	--ctx-size 4096 \
	--temp 0.1 \
	--top-p 0.1 \
	--top-k 1 \
	--host 0.0.0.0 \
	--port 2026 \
	--timeout 30 \
	--no-webui \
	--slot-save-path ./kv_cache

# Docker pgvector connection
postgresql+psycopg://langchain:langchain@localhost:6024/langchain
# Docker run pgvector16
docker run \
	--name pgvector-container \
	-e POSTGRES_USER=langchain \
	-e POSTGRES_PASSWORD=langchain \
	-e POSTGRES_DB=langchain \
	-p 6024:5432 \
	-d pgvector/pgvector:pg16

# lm-eval-harness for finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct
lm_eval --model hf \
	--model_args pretrained="/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct" \
	--tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa \
	--device cuda \
	--batch_size auto 

# VLLM runserver 
# vllm serve /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct \
vllm serve /home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct \
	--host 127.0.0.1 \
	--port 2026 \
	--gpu-memory-utilization 0.3 \
	--guided-decoding-backend lm-format-enforcer \
	--dtype auto \
	--device auto \
	--enable-sleep-mode \
	--allow-credentials \
	--block-size 8 \
	--device auto \
	--enable-auto-tool-choice \
	--enable-prompt-adapter \
	--enable-sleep-mode \
	--enforce-eager \
	--gpu-memory-utilization 0.9 \
	--guided-decoding-backend lm-format-enforcer \
	--load-format auto \
	--max-num-seqs 5 \
	--enable-auto-tool-choice \
	--tool-call-parser llama3_json \
	--max-prompt-adapter-token 2048
# Benchmark inference VLLM runserver
python ./third_3rdparty/vllm-0.7.1/benchmarks/benchmark_serving.py \
	--backend openai \
	--base-url 127.0.0.1:2025 \
	--dataset-name random \
	--model /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct \
	--seed 12345 \
	--tokenizer-mode auto \
	--trust-remote-code \
	--use-beam-search 

# chroma_db runserver 
chroma run \
	--path /home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db \
	--host 127.0.0.1 \
	--port 2027 \
	--log-path /home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db.log

# HF-Transformers convert .pth to .safetensors format
python /home/chwenjun225/.llama/checkpoints/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
	--input_dir /home/chwenjun225/.llama/checkpoints/Llama3.2-11B-Vision-Instruct/pth \
	--model_size 11B \
	--output_dir /home/chwenjun225/.llama/checkpoints/Llama3.2-11B-Vision-Instruct
# Convert DeepSeek-R1-Distill-Qwen-1.5B hf to gguf
python convert_hf_to_gguf.py \
	/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct \
	--outtype f32 \
	--outfile /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/gguf
# Convert Llama-3.2-1B-Instruct hf to gguf
python convert_hf_to_gguf.py \
	/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct \
	--outtype f32 \
	--outfile /home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct/gguf
# Convert Llama-3.2-11B-Vision-Instruct hf to gguf
# ... working on

# Kill the loaded model VRAM 
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

# CUDA static realtime
watch -n0.2 nvidia-smi

# CPU & RAM static realtime
htop

# Kill port
sudo fuser -k 2026/tcp
