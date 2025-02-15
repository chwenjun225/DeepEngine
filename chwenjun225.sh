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

# llama_cpp runserver --support tool call for deepseek-r1 successfully 
# Either "json_schema" or "grammar" can be specified, but not both'
./third_3rdparty/llama.cpp-b4641/build/bin/llama-server \
	--alias foxconn_ai_research_llama32_1b_instruct\
	--model /home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct/gguf/Llama-3.2-1B-Instruct-F32.gguf \
	--parallel 8 \
	--host 127.0.0.1 \
	--port 2026 \
	--ctx-size 0 \
	--predict -2 \
	--mlock \
	--no-mmap \
	--cont-batching \
	--flash-attn \
	--parallel 8 \
	--temp 0 \
	--no-webui \
	--ubatch-size 1024 \
	--ctx-size 4096 \
	--n-gpu-layers 256 \
	--jinja 
	# --no-context-shift 	


# chroma_db runserver 
chroma run \
	--path /home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db \
	--host 127.0.0.1 \
	--port 2027 \
	--log-path /home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db.log

# Convert DeepSeek-R1-Distill-Qwen-1.5B hf to gguf
python convert_hf_to_gguf.py /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct --outtype f32 --outfile /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/gguf
# Convert Llama-3.2-1B-Instruct hf to gguf
python convert_hf_to_gguf.py /home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct \
	--outtype f32 \
	--outfile /home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct/gguf

# Kill the loaded model VRAM 
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

# CUDA static realtime
watch -n0.3 gpustat -cp --color

# CPU & RAM static realtime
htop

# Kill port
sudo fuser -k 2026/tcp
