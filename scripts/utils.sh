# Convert DeepSeek-R1-Distill-Qwen-1.5B hf to gguf
python convert_hf_to_gguf.py /home/kali/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B \
--outtype f32 \
--outfile /home/kali/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B/gguf

# Start server with DeepSeek-R1-Distill-Qwen-1.5B.gguf
llama-server -m /home/kali/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B/gguf \
--port 8080