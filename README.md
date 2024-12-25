Ý tưởng: Tạo app sử dụng torch_mobile, tensorflowlite hoặc padlepadle để inference llama3.2-1b trên thiết bị di động. Từ đó nhắm đến mục tiêu tối tăng cường khả năng thông minh của model llama3.2-1b trên thiết bị di động.

- Thêm **Projected Network** vào kiến trúc Llama3 

torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir /home/chwenjun225/.llama/checkpoints/Llama3.2-1B-Instruct/ --tokenizer_path /home/chwenjun225/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model --max_seq_len 128 --max_batch_size 4


/home/chwenjun225/.llama/checkpoints/Llama3.2-1B-Instruct