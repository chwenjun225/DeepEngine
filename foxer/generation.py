import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
	get_model_parallel_rank,
	initialize_model_parallel,
	model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import ChatFormat, Dialog, Message, Tokenizer

class CompletionPrediction(TypedDict, total=False):
	generation: str
	tokens: List[str]  # not required
	logprobs: List[float]  # not required

class ChatPrediction(TypedDict, total=False):
	generation: Message
	tokens: List[str]  # not required
	logprobs: List[float]  # not required

class Llama:
	@staticmethod
	def build(
		ckpt_dir: str,
		tokenizer_path: str,
		max_seq_len: int,
		max_batch_size: int,
		model_parallel_size: Optional[int] = None,
		seed: int = 1,
	) -> "Llama":
		"""
		Tạo một đối tượng Llama bằng cách khởi tạo và tải checkpoint của mô hình.

		Tham số:
			ckpt_dir (str): Đường dẫn đến thư mục chứa các tệp checkpoint.
			tokenizer_path (str): Đường dẫn đến tệp tokenizer.
			max_seq_len (int): Chiều dài chuỗi tối đa cho đầu vào văn bản.
			max_batch_size (int): Kích thước batch tối đa khi suy luận.
			model_parallel_size (Optional[int], optional): Số tiến trình song song hóa mô hình.
				Nếu không được cung cấp, giá trị sẽ được xác định từ môi trường. Mặc định là None.

		Kết quả trả về:
			Llama: Một đối tượng của lớp Llama với mô hình và tokenizer đã được tải.

		Lỗi có thể gặp:
			AssertionError: Nếu không tìm thấy tệp checkpoint trong thư mục chỉ định,
				hoặc nếu số lượng tiến trình song song không khớp với số lượng tệp checkpoint.

		Ghi chú:
			Phương thức này khởi tạo nhóm tiến trình phân tán, thiết lập thiết bị là CUDA,
			và tải mô hình đã huấn luyện trước cùng với tokenizer.
		"""
		# Kiểm tra chiều dài chuỗi hợp lệ 
		assert 1 <= max_seq_len <= 8192, f"max_seq_len phải nằm trong khoảng [1, 8192]. Nhận được {max_seq_len}."

		# Kiểm tra sự tồn tại của thư mục checkpoint
		assert os.path.isdir(ckpt_dir), f"Thư mục checkpoint '{ckpt_dir}' không tồn tại."

		# Kiểm tra sự tồn tại của tệp tokenizer 
		assert os.path.isfile(tokenizer_path), f"Tệp tokenizer '{tokenizer_path}' không tồn tại."
		
		# Khởi tạo nhóm tiến trình phân tán nếu chưa có 
		if not torch.distributed.is_initialized():
			torch.distributed.init_process_group("nccl")
		
		# Khởi tạo song song mô hình nếu chưa thực hiện  
		if not model_parallel_is_initialized():
			if model_parallel_size is None:
				model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
			initialize_model_parallel(model_parallel_size)

		# Chỉ định thiết bị CUDA cho tình trình cục bộ 
		local_rank = int(os.environ.get("LOCAL_RANK", 0))
		torch.cuda.set_device(local_rank)

		# Đặt seed ngẫu nhiên (phải giống nhau trên tất cả các tiến trình)
		torch.manual_seed(seed)

		# Ẩn đầu ra các tiến trình không phải tiến trình chính 
		if local_rank > 0:
			sys.stdout = open(os.devnull, "w")

		start_time = time.time()

		# Lấy danh sách các tệp checkpoint trong thư mục 
		checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
		assert len(checkpoints) > 0, f"Không tìm thấy tệp checkpoint nào trong: {ckpt_dir}"
		assert model_parallel_size == len(
			checkpoints
		), f"Đang tải số lượng checkpoint với MP={len(checkpoints)} nhưng số lượng GPU hiện có là {model_parallel_size}"
		# Chọn checkpoint theo tiến trình hiện tại 
		ckpt_path = checkpoints[get_model_parallel_rank()]
		checkpoint = torch.load(ckpt_path, map_location="cpu")
		
		# # Tải tham số mô hình từ tệp params.json
		with open(Path(ckpt_dir) / "params.json", "r") as f:
			params = json.loads(f.read())

		# Khởi tạo đối tượng ModelArgs
		model_args: ModelArgs = ModelArgs(
			max_seq_len=max_seq_len,
			max_batch_size=max_batch_size,
			**params,
		)
		# Tải tokenizer 
		tokenizer = Tokenizer(model_path=tokenizer_path)
		assert model_args.vocab_size == tokenizer.n_words

		# Thiết lập kiểu dữ liệu tensor mặc định 
		if torch.cuda.is_bf16_supported():
			torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
		else:
			torch.set_default_tensor_type(torch.cuda.HalfTensor)
		
		# Tạo và tải trạng thái mô hình
		model = Transformer(model_args)
		model.load_state_dict(checkpoint, strict=False)

		# Hiện thông báo thời gian tải 
		print(f"Hoàn thành tải trong {time.time() - start_time:.2f} giây")

		# Trả về đối tượng Llama
		return Llama(model, tokenizer)