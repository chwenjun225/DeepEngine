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
	
	def __init__(self, model: Transformer, tokenizer: Tokenizer):
		self.model = model
		self.tokenizer = tokenizer
		self.formatter = ChatFormat(tokenizer)

	@torch.inference_mode() # Tắt theo dõi gradient để tăng tốc và tiết kiệm bộ nhớ
	def generate(
		self,
		prompt_tokens: List[List[int]],
		max_gen_len: int,
		temperature: float = 0.6,
		top_p: float = 0.9,
		logprobs: bool = False,
		echo: bool = False,
	) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
		"""
		Sinh chuỗi văn bản dựa trên các prompt (đầu vào) đã được cung cấp, sử dụng mô hình ngôn ngữ.

		Args:
			prompt_tokens (List[List[int]]): Danh sách các prompt đã được tokenize, mỗi prompt là một danh sách các số nguyên.
			max_gen_len (int): Chiều dài tối đa của chuỗi văn bản được sinh ra.
			temperature (float, optional): Giá trị temperature để điều chỉnh độ ngẫu nhiên khi lấy mẫu (sampling). Mặc định là 0.6.
			top_p (float, optional): Ngưỡng xác suất top-p để sampling nucleus. Mặc định là 0.9.
			logprobs (bool, optional): Nếu True, tính toán log xác suất của từng token. Mặc định là False.
			echo (bool, optional): Nếu True, bao gồm cả prompt trong chuỗi đầu ra. Mặc định là False.
		Returns:
			Tuple[List[List[int]], Optional[List[List[float]]]]: 
				- Một tuple chứa:
					+) Danh sách các chuỗi token đã được sinh ra.
					+) Nếu `logprobs=True`, trả về thêm danh sách các log xác suất tương ứng của các token.
		Note:
			- Phương thức này sử dụng các prompt làm cơ sở để sinh văn bản.
			- Sử dụng nucleus sampling (top-p sampling) để tạo chuỗi văn bản với độ ngẫu nhiên được kiểm soát.
			- Nếu `logprobs=True`, log xác suất của từng token được tính toán cho mỗi token được sinh ra.
		"""
		params = self.model.params # Lấy các tham số từ mô hình
		bsz = len(prompt_tokens) # Số lượng prompt trong batch
		assert bsz <= params.max_batch_size, (bsz, params.max_batch_size) # Kiểm tra batch size không vượt quá giới hạn

		# Lấy chiều dài ngắn nhất và dài nhất của các prompt
		min_prompt_len = min(len(t) for t in prompt_tokens)
		max_prompt_len = max(len(t) for t in prompt_tokens)
		assert max_prompt_len <= params.max_seq_len # Kiểm tra chiều dài chuỗi không vượt quá giới hạn của mô hình
		
		# Xác định tổng chiều dài (bao gồm cả prompt và phần sinh thêm)
		total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

		pad_id = self.tokenizer.pad_id # Căn chỉnh độ dài các prompt sao cho bằng nhau
		tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
		# Sao chép các token của prompt vào tensor `tokens`
		for k, t in enumerate(prompt_tokens):
			tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
		if logprobs:
			token_logprobs = torch.zeros_like(tokens, dtype=torch.float) # Khởi tạo tensor lưu log xác suất

		prev_pos = 0 # Vị trí trước đó trong quá trình sinh token
		eos_reached = torch.tensor([False] * bsz, device="cuda") # Theo dõi các chuỗi đã kết thúc (EOS)
		input_text_mask = tokens != pad_id # Mask để phân biệt token thực với padding
		if min_prompt_len == total_len: # Trường hợp toàn bộ prompt đã đầy đủ
			logits = self.model.forward(tokens, prev_pos) # Tính toán logits từ mô hình
			token_logprobs = -F.cross_entropy( # Tính toán log xác suất
				input=logits.transpose(1, 2),
				target=tokens,
				reduction="none",
				ignore_index=pad_id,
			)

		# Token kết thúc chuỗi
		stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))
		# Vòng lặp sinh token
		for cur_pos in range(min_prompt_len, total_len):
			# Tính toán logits (đầu ra) của các token tiếp theo
			logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
			if temperature > 0: # Sampling nếu tham số temperature lớn hơn 0
				probs = torch.softmax(logits[:, -1] / temperature, dim=-1) # Chuẩn hóa xác suất
				next_token = sample_top_p(probs, top_p) # Sampling (lấy mẫu) dựa trên top-p
			else:
				next_token = torch.argmax(logits[:, -1], dim=-1)

			next_token = next_token.reshape(-1) # Định hình lại token
			# only replace token if prompt has already been generated
			next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
			tokens[:, cur_pos] = next_token # Lưu token vào tensor `tokens`
			if logprobs: # Nếu cần tính log xác suất
				token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
					input=logits.transpose(1, 2),
					target=tokens[:, prev_pos + 1 : cur_pos + 1],
					reduction="none",
					ignore_index=pad_id,
				)
			# Cập nhật trạng thái EOS nếu token kết thúc xuất hiện
			eos_reached |= (~input_text_mask[:, cur_pos]) & (
				torch.isin(next_token, stop_tokens)
			)
			prev_pos = cur_pos # Cập nhật vị trí trước đó
			if all(eos_reached): # Dừng vòng lặp nếu tất cả chuỗi đã kết thúc
				break

		if logprobs:
			token_logprobs = token_logprobs.tolist() # Chuyển tensor log xác suất sang dạng danh sách
		out_tokens, out_logprobs = [], []
		for i, toks in enumerate(tokens.tolist()):  # Duyệt qua từng chuỗi đã sinh
			# cut to max gen len
			start = 0 if echo else len(prompt_tokens[i]) # Bắt đầu từ đầu hoặc sau prompt
			toks = toks[start : len(prompt_tokens[i]) + max_gen_len] # Cắt chuỗi theo chiều dài tối đa
			probs = None
			if logprobs:
				probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
			# Cắt chuỗi nếu gặp token kết thúc
			for stop_token in self.tokenizer.stop_tokens:
				try:
					eos_idx = toks.index(stop_token)
					toks = toks[:eos_idx]
					probs = probs[:eos_idx] if logprobs else None
				except ValueError:
					pass
			out_tokens.append(toks) # Thêm chuỗi đã sinh vào kết quả
			out_logprobs.append(probs) # Thêm log xác suất (nếu có)
		return (out_tokens, out_logprobs if logprobs else None) # Trả về kết quả

	def text_completion(
		self,
		prompts: List[str], # Danh sách các văn bản đầu vào (prompt) cần hoàn thiện
		temperature: float = 0.6, # Giá trị temperature để điều chỉnh độ ngẫu nhiên trong sampling (mặc định: 0.6)
		top_p: float = 0.9, # Ngưỡng xác suất top-p cho nucleus sampling (mặc định: 0.9)
		max_gen_len: Optional[int] = None, # Chiều dài tối đa của văn bản được sinh, nếu không cung cấp sẽ được tính tự động
		logprobs: bool = False, # Có tính log xác suất của các token hay không (mặc định: False)
		echo: bool = False, # Có bao gồm các token prompt trong kết quả đầu ra hay không (mặc định: False)
	) -> List[CompletionPrediction]:
		"""
		Hoàn thiện văn bản cho danh sách các prompt sử dụng language generation model.

		Args:
			prompts (List[str]): Danh sách các văn bản prompt cần hoàn thiện.
			temperature (float, optional): Giá trị temperature để điều chỉnh độ ngẫu nhiên khi sampling. Mặc định là 0.6.
			top_p (float, optional): Ngưỡng top-p cho nucleus sampling. Mặc định là 0.9.
			max_gen_len (Optional[int], optional): Chiều dài tối đa cho chuỗi văn bản được sinh.
				Nếu không cung cấp, giá trị mặc định là chiều dài tối đa của mô hình trừ 1.
			logprobs (bool, optional): Nếu True, tính log xác suất của các token. Mặc định là False.
			echo (bool, optional): Nếu True, bao gồm các token prompt trong kết quả đầu ra. Mặc định là False.

		Returns:
			List[CompletionPrediction]: Danh sách các dự đoán hoàn thiện, mỗi phần tử chứa chuỗi văn bản được sinh.

		Note:
			- Hàm này sử dụng nucleus sampling để kiểm soát độ ngẫu nhiên khi sinh văn bản.
			- Nếu `logprobs=True`, log xác suất của từng token sẽ được tính.
		"""
		# Nếu không cung cấp max_gen_len, thiết lập nó bằng chiều dài tối đa của mô hình trừ đi 1
		if max_gen_len is None:
			max_gen_len = self.model.params.max_seq_len - 1
		
		# Mã hóa từng prompt trong danh sách `prompts` thành danh sách các token
		# bos=True: Thêm token bắt đầu, eos=False: Không thêm token kết thúc
		prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
		
		# Gọi hàm `generate` để sinh chuỗi văn bản từ các prompt đã được mã hóa
		generation_tokens, generation_logprobs = self.generate(
			prompt_tokens=prompt_tokens, # Danh sách các token của prompt
			max_gen_len=max_gen_len, # Chiều dài tối đa của văn bản được sinh
			temperature=temperature, # Giá trị temperature để điều chỉnh ngẫu nhiên
			top_p=top_p, # Ngưỡng xác suất top-p
			logprobs=logprobs, # Có tính log xác suất hay không
			echo=echo, # Có bao gồm token prompt trong đầu ra hay không
		)

		# Nếu logprobs=True, trả về danh sách các dự đoán với log xác suất
		if logprobs:
			return [
				{
					"generation": self.tokenizer.decode(t), # Giải mã danh sách token thành văn bản
					"tokens": [self.tokenizer.decode([x]) for x in t], # Giải mã từng token riêng lẻ
					"logprobs": logprobs_i, # Log xác suất của từng token
				}
				for t, logprobs_i in zip(generation_tokens, generation_logprobs) # Kết hợp token và logprobs tương ứng
			]
		# Nếu logprobs=False, chỉ trả về danh sách văn bản được sinh ra
		return [
			{
				"generation": self.tokenizer.decode(t) # Giải mã danh sách token thành văn bản
			} 
			for t in generation_tokens # Duyệt qua từng chuỗi token đã được sinh ra
		]

	def chat_completion(
		self, 
		dialogs: List[Dialog], # Danh sách các hội thoại, mỗi hội thoại là một danh sách các tin nhắn
		temperature: float = 0.6, # Giá trị temperature để điều chỉnh độ ngẫu nhiên khi sampling (mặc định: 0.6)
		top_p: float = 0.9, # Ngưỡng top-p cho nucleus sampling (mặc định: 0.9)
		max_gen_len: Optional[int] = None, # Chiều dài tối đa của phản hồi được sinh, nếu không cung cấp sẽ được đặt tự động
		logprobs: bool = False, # Có tính log xác suất của từng token hay không (mặc định: False)
	) -> List[ChatPrediction]:
		"""
		Sinh phản hồi từ trợ lý ảo (assistant) cho danh sách các hội thoại đã cung cấp.

		Args:
			dialogs (List[Dialog]): Danh sách các hội thoại, mỗi hội thoại bao gồm một danh sách các tin nhắn.
			temperature (float, optional): Giá trị nhiệt độ để điều chỉnh độ ngẫu nhiên khi sampling. Mặc định là 0.6.
			top_p (float, optional): Ngưỡng top-p cho nucleus sampling. Mặc định là 0.9.
			max_gen_len (Optional[int], optional): Chiều dài tối đa của phản hồi được sinh.
				Nếu không cung cấp, giá trị mặc định là chiều dài tối đa của mô hình trừ 1.
			logprobs (bool, optional): Nếu True, tính log xác suất cho từng token được sinh. Mặc định là False.

		Returns:
			List[ChatPrediction]: Danh sách các dự đoán hội thoại, mỗi dự đoán chứa phản hồi của trợ lý ảo.

		Note:
			- Phương thức này sinh các phản hồi của trợ lý ảo dựa trên hội thoại đã cung cấp.
			- Sử dụng nucleus sampling để kiểm soát độ ngẫu nhiên trong quá trình sinh văn bản.
			- Nếu `logprobs=True`, log xác suất của từng token được tính.
		"""
		if max_gen_len is None:
			max_gen_len = self.model.params.max_seq_len - 1

		prompt_tokens = [
			self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
		]
		generation_tokens, generation_logprobs = self.generate(
			prompt_tokens=prompt_tokens,
			max_gen_len=max_gen_len,
			temperature=temperature,
			top_p=top_p,
			logprobs=logprobs,
		)
		if logprobs:
			return [
				{
					"generation": {
						"role": "assistant",
						"content": self.tokenizer.decode(t),
					},
					"tokens": [self.tokenizer.decode([x]) for x in t],
					"logprobs": logprobs_i,
				}
				for t, logprobs_i in zip(generation_tokens, generation_logprobs)
			]
		return [
			{
				"generation": {
					"role": "assistant",
					"content": self.tokenizer.decode(t),
				},
			}
			for t in generation_tokens
		]


def sample_top_p(probs, p):
	"""
	Thực hiện lấy mẫu (sampling) top-p (nucleus sampling) trên một phân phối xác suất.

	Args:
		probs (torch.Tensor): Tensor chứa phân phối xác suất.
		p (float): Ngưỡng xác suất cho top-p sampling.

	Returns:
		torch.Tensor: Chỉ số của token được chọn.

	Note:
		- Top-p sampling chọn tập hợp nhỏ nhất các token sao cho tổng xác suất tích lũy vượt quá ngưỡng p.
		- Phân phối xác suất được chuẩn hóa lại dựa trên các token được chọn.
	"""
	probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # Sắp xếp xác suất theo thứ tự giảm dần
	probs_sum = torch.cumsum(probs_sort, dim=-1) # Tính tổng tích lũy của các xác suất
	mask = probs_sum - probs_sort > p # Tạo mặt nạ (mask) để xác định các token vượt ngưỡng p
	probs_sort[mask] = 0.0 # Đặt xác suất của các token bị loại bỏ thành 0
	probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # Chuẩn hóa lại phân phối xác suất cho các token còn lại
	next_token = torch.multinomial(probs_sort, num_samples=1) # Lấy mẫu token tiếp theo từ phân phối đã được chuẩn hóa
	next_token = torch.gather(probs_idx, -1, next_token) # Dựa vào chỉ số ban đầu để lấy token được chọn trong tập ban đầu
	return next_token # Trả về chỉ số của token được chọn