# lm-eval-harness for finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct
lm_eval --model hf \
--model_args pretrained="/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct" \
--tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa \
--device cuda \
--batch_size auto 

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

# llama_cpp runserver
./third_3rdparty/llama.cpp-b4621/build/bin/llama-server -m /home/chwenjun225/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B/gguf/DeepSeek-R1-Distill-Qwen-1.5B-F32.gguf \
--port 2026 

# Convert DeepSeek-R1-Distill-Qwen-1.5B hf to gguf
python convert_hf_to_gguf.py /home/kali/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B \
--outtype f32 \
--outfile /home/chwenjun225/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B/gguf

# Kill the loaded model VRAM 
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9