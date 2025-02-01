# Convert DeepSeek-R1-Distill-Qwen-1.5B hf to gguf
python convert_hf_to_gguf.py /home/kali/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B \
--outtype f32 \
--outfile /home/kali/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B/gguf

# Start server with DeepSeek-R1-Distill-Qwen-1.5B.gguf
./from_3party/llama.cpp/build/bin/llama-server -m ./models/DeepSeek-R1-Distill-Qwen-1.5B/gguf/DeepSeek-R1-Distill-Qwen-1.5B-F32.gguf \
--port 8080