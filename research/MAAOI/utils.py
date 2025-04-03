import os 
import cv2
import csv 
import time
import base64
import datetime
import asyncio
import numpy as np 
from PIL import Image 
from io import BytesIO
from typing import Callable



from langchain_core.runnables import RunnableLambda



from langchain_core.messages import (
	BaseMessage				, 
	SystemMessage			,
)



from state import State 
from const import (
	WANNA_MEASURE_TIME		,
	MEASURE_LOG_FILE		,
	ENCODING				,
	MAX_TOKENS				,
)



def list_image_files_from_folder(folder_path: str, valid_exts={".png", ".jpg", ".jpeg"}) -> list:
	"""Duyệt qua một thư mục và trả về danh sách đầy đủ đường dẫn các ảnh."""
	image_paths = []
	for fname in os.listdir(folder_path):
		if os.path.splitext(fname)[1].lower() in valid_exts:
			image_paths.append(os.path.join(folder_path, fname))
	return sorted(image_paths)



def measure_time(tag:str="Execution") -> Callable:
	"""Decorator dùng để đo thời gian thực thi của hàm (cả async & sync), ghi kết quả để vẽ biểu đồ."""
	def decorator(func):
		if WANNA_MEASURE_TIME:
			if asyncio.iscoroutinefunction(func):
				async def async_wrapper(*args, **kwargs):
					start = time.time()
					result = await func(*args, **kwargs)
					end = time.time()
					duration_ms = (end - start) * 1000
					print(f">>> [{tag}] took {duration_ms:.2f} ms")
					with open(MEASURE_LOG_FILE, mode="a", newline="") as f:
						writer = csv.writer(f)
						writer.writerow([tag, f"{duration_ms:.2f}"])
					return result
				return async_wrapper
			else:
				def sync_wrapper(*args, **kwargs):
					start = time.time()
					result = func(*args, **kwargs)
					end = time.time()
					duration_ms = (end - start) * 1000
					print(f">>> [{tag}] took {duration_ms:.2f} ms")
					with open(MEASURE_LOG_FILE, mode="a", newline="") as f:
						writer = csv.writer(f)
						writer.writerow([tag, f"{duration_ms:.2f}"])
					return result
				return sync_wrapper
	return decorator



def save_frame_result(image:Image.Image, filename:str, status:str, output_dir:str="evals") -> None:
	"""Lưu ảnh đã dự đoán vào thư mục tương ứng theo trạng thái (OK/NG/UNKNOWN),"""
	status = (status or "").strip().upper()
	if status not in {"OK", "NG"}:
		status = "UNKNOWN"

	save_path = os.path.join(output_dir, f"{status.upper()}")
	os.makedirs(save_path, exist_ok=True)

	timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
	filename = f"{status}__{timestamp}.jpg"

	full_path = os.path.join(save_path, filename)
	image.save(full_path)
	print(f">>> Saved result to: {full_path}")



def extract_frames_from_video(video_path:str, output_dir:str, interval:int=10) -> None:
	"""Trích xuất ảnh từ video."""
	os.makedirs(output_dir, exist_ok=True)
	cap = cv2.VideoCapture(video_path)
	frame_id = 0
	saved = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		if frame_id % interval == 0:
			filename = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
			cv2.imwrite(filename, frame)
			saved += 1
		frame_id += 1
	cap.release()
	print(f">>> Extracted {saved} frames to {output_dir}")



def draw_defect_overlay(
	frame: np.ndarray,
	bboxes: dict, 
	product_status: str
) -> np.ndarray:
	"""Hiển thị bounding boxes, trạng thái sản phẩm, và timestamp lên ảnh."""
	image = frame.copy()
	font = cv2.FONT_HERSHEY_SIMPLEX

	if product_status.upper() == "OK":
		main_color = (0, 255, 0)  			### Green
		label_text = "OK"
	elif product_status.upper() == "NG":
		main_color = (0, 0, 255)  			### Red
		label_text = "NG"
	else:
		main_color = (0, 255, 255)  		### Yellow
		label_text = "Processing..."

	cv2.putText(
		image, label_text, org=(20, 40), fontFace=font, 
		fontScale=1.2, color=main_color, thickness=3, lineType=cv2.LINE_AA)

	timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	text_size, _ = cv2.getTextSize(timestamp, font, 0.6, 1)
	text_w, _ = text_size
	_, img_w = image.shape[:2]
	cv2.putText(
		image, timestamp, org=(img_w-text_w-20, 30), 
		fontFace=font, fontScale=1, thickness=1, lineType=cv2.LINE_AA, color=main_color)

	for obj in bboxes:
		x1, y1, x2, y2 = obj["bbox"]
		label = obj["label"]

		box_color = (0, 0, 255) if product_status.upper() == "NG" else (0, 200, 0)

		cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

		label_str = f"{label}"
		(tw, th), _ = cv2.getTextSize(label_str, font, 0.5, 1)
		cv2.rectangle(
			image, (x1, y1-th-4), 
			(x1+tw+4, y1), box_color, -1
		)
		cv2.putText(
			image, label_str, (x1+2, y1-2), font, 0.5, 
			(255, 255, 255), 1, lineType=cv2.LINE_AA
		)
	return image



def image_to_base64(pil_img: Image.Image | np.ndarray) -> str:
	"""Convert PIL or NumPy image to base64 string (PNG format), optimized for real-time usage."""
	with BytesIO() as buffer:
		pil_img.save(buffer, format="PNG")
		return base64.b64encode(buffer.getvalue()).decode("utf-8")



def expand_bbox(bbox:list[int], delta:int=6) -> list[int]:
	"""Điều chỉnh bbox theo hệ số cho trước."""
	x1, y1, x2, y2 = bbox 
	x2 = x2 + delta//2 
	x1 = x1 - delta//2
	y2 = y2 + delta//2
	y1 = y1 - delta//2 
	return [x1, y1, x2, y2]



def passthrough() -> RunnableLambda:
	"""Trả về một Runnable không thay đổi state."""
	return RunnableLambda(lambda x: x)



def prepare_context(
		state: State, 
		agent: str, 
		system_prompt: str
	) -> tuple[list[BaseMessage], SystemMessage|None]:
	"""Thêm system message nếu chưa có và trả về context đã trim."""
	ctx = get_msgs(state)
	sys_msg = None
	if not has_agent_got_sys_prompt(ctx, agent):
		sys_msg = SystemMessage(content=system_prompt, name=agent)
		ctx.append(sys_msg)
	return trim_context(ctx), sys_msg



def trim_context(
		context: list[BaseMessage]
	) -> list[BaseMessage]:
	"""Cắt context từ đầu nếu vượt token limit, nhưng luôn giữ system message cuối cùng (nếu có)."""
	while (count_tokens(context)>MAX_TOKENS) \
			and (len(context)>1):
		context = context[1:]
	return context



def replace_message_content(
		msg: BaseMessage, 
		new_content: str
	) -> BaseMessage:
	"""Chỉnh sửa content của một BaseMessage."""
	return type(msg)(
		content=f"Forwarded to `{new_content}`.",
		name=getattr(msg, "name", None),
		additional_kwargs=msg.additional_kwargs,
		response_metadata=msg.response_metadata,
	)



def has_agent_got_name_attr(
		response: dict|BaseMessage, agent_name: str
	) -> dict|BaseMessage:
	"""Gán thuộc tính name cho phản hồi của AI nếu chưa có."""
	if isinstance(response, dict):
		if "name" not in response:
			response["name"] = agent_name
		return response 

	elif isinstance(response, BaseMessage):
		if not hasattr(response, "name") \
			or getattr(response, "name") is None:
			setattr(response, "name", agent_name)

		return response 
	raise TypeError(f">>> [has_name_attr] Không hỗ trợ kiểu dữ liệu: {type(response)} ")



def has_agent_got_sys_prompt(context: list[dict|BaseMessage], agent_name: str) -> bool:
	"""Kiểm tra xem đã có system message cho agent được chỉ định chưa, type==role"""
	for c in context:
		role = name = None

		if isinstance(c, dict):
			role = c.get("role")
			name = c.get("name")
		elif isinstance(c, BaseMessage):
			role = getattr(c, "type", None)
			name = getattr(c, "name", None)

		if role == "system" and name == agent_name:
			return True
	return False



def count_tokens(messages: list[BaseMessage]) -> int:
	"""Đếm tổng số tokens trong messages theo model. 
		Token structure theo chuẩn OpenAI ChatML"""
	num_tokens = 0 
	for msg in messages:
		role = getattr(msg, "type", "user")
		content = getattr(msg, "content", "")
		name = getattr(msg, "name", None)
		num_tokens += 4 
		num_tokens += len(ENCODING.encode(role))
		num_tokens += len(ENCODING.encode(content))
		if name: 
			num_tokens += len(ENCODING.encode(name))

	num_tokens += 2
	return num_tokens



def estimate_tokens(text: str) -> int:
	"""Ước lượng token trong prompt."""
	return len(ENCODING.encode(text))



def get_safe_num_predict(prompt: str, max_context: int = 131072, buffer: int = 512) -> int:
	"""Hàm tự động tính num_predict

	Args:
		prompt: nội dung bạn muốn gửi vào mô hình
		max_context: tổng context window của model (token)
		model_name: để chọn đúng tokenizer
		buffer: token chừa ra để tránh cắt hoặc lỗi
	"""
	prompt_tokens = estimate_tokens(prompt)
	available_tokens = max_context - prompt_tokens - buffer
	return max(256, min(available_tokens, 128000))



def get_msgs(state: State) -> list[BaseMessage]:
	"""Lấy danh sách tin nhắn từ State."""
	return state["messages"]



def get_latest_msg(state:State, type_msgs:str) -> BaseMessage:
	"""Lấy tin nhắn mới nhất của một agent, O(1)."""
	return state[type_msgs][-1]
