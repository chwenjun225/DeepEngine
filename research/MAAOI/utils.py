import cv2
import base64
import numpy as np 
from PIL import Image 
from io import BytesIO



from langchain_core.runnables import RunnableLambda



from langchain_core.messages import (
	BaseMessage				, 
	SystemMessage			,
)



from state import State 
from const import (
	ENCODING				, 
	MAX_TOKENS				, 
)



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



def draw_defect_overlay(frame, bboxes, status=None):
	"""Hiển thị bounding boxes và trạng thái QC lên frame ảnh."""
	# Copy ảnh để không ghi đè
	image = frame.copy()

	# Font setup
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.6
	thickness = 2

	# 1. Vẽ các bounding boxes
	for obj in bboxes:
		x1, y1, x2, y2 = map(int, obj["bbox"])
		label = obj.get("label", "unknown")
		conf = obj.get("conf", None)

		# Màu mặc định
		color = (0, 255, 255)  # vàng

		# Vẽ box
		cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

		# Nội dung nhãn
		text = f"{label}"
		if conf is not None:
			text += f" ({conf:.2f})"

		# Ghi text trên box
		cv2.putText(image, text, (x1, y1 - 5), font, font_scale, color, thickness)

	# 2. Hiển thị trạng thái OK/NG ở góc phần tư thứ nhất
	if status in ("OK", "NG"):
		color = (0, 200, 0) if status == "OK" else (0, 0, 255)  # xanh hoặc đỏ
		cv2.putText(
			image,
			f">>> Status: {status}",
			(20, 40),
			font,
			1.0,
			color,
			3,
			lineType=cv2.LINE_AA
		)

	return image