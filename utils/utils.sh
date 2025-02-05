# lm-eval-harness for finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct
lm_eval --model hf \
--model_args pretrained="/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct" \
--tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa \
--device cuda \
--batch_size auto 

# Run LLMCompiler 
python run_llm_compiler.py \
--model_type vllm \
--model_name /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct \
--benchmark_name parallelqa \
--store /home/chwenjun225/Projects/Foxer/evals/LLMCompiler_store \
--logging \
--api_key "chwenjun225" \
--vllm_port 2025

# VLLM runserver 
vllm serve /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct \
--host 127.0.0.1 \
--port 2025 \
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
--base-url http://127.0.0.1:2025 \
--dataset-name random \
--model /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct \
--seed 12345 \
--tokenizer-mode auto \
--trust-remote-code \
--use-beam-search 

# llama_cpp runserver
./third_3rdparty/llama.cpp-b4641/build/bin/llama-server -m /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/gguf/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct-1.8B-1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct-F32.gguf \
--port 2026 \
-ngl 200

# 1️. Kiểm tra hiệu suất inference trên llama.cpp
./third_3rdparty/llama.cpp-b4641/build/bin/main -m models/your_model.gguf -p "A machine reports high vibration levels. What should be done?" \
--repeat_prompt \
--threads 8 \
--n 256
# -m models/your_model.gguf → Load mô hình .gguf.
# -p → Dùng prompt để kiểm tra inference.
# --repeat_prompt → Lặp lại lệnh để đo tốc độ.
# --threads 8 → Sử dụng 8 luồng CPU (Tăng hoặc giảm tùy phần cứng).
# --n 256 → Sinh ra 256 token đầu ra, giúp đo tốc độ token/s.
# 2. Chạy benchmark tốc độ inference
./third_3rdparty/llama.cpp-b4641/build/bin/main -m models/your_model.gguf -p "Benchmarking Llama.cpp server" \
--tokens 256 \
--threads 8

# Convert DeepSeek-R1-Distill-Qwen-1.5B hf to gguf
python convert_hf_to_gguf.py /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct \
--outtype f32 \
--outfile /home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/gguf

# Kill the loaded model VRAM 
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

# CUDA static realtime
watch -n0.3 gpustat -cp --color

# CPU & RAM static realtime
htop

# Kill port
sudo fuser -k your_port/tcp
sudo fuser -k 2026/tcp
