import cv2 
import numpy as np 
import base64
import json
import fire 
import gradio as gr 
import time 
from io import BytesIO
from PIL import Image 



from agentic import AGENTIC
from const_vars import (
    VISISON_AGENT_PROMPT_MSG,
    YOLO_OBJECT_DETECTION,
    VISION_INSTRUCT_LLM,
    CONFIG,
    PRODUCT_STATUS,
    STATUS_LOCK
)
from utils import get_latest_msg



def get_status() -> str:
	"""Trả về trạng thái lỗi hiện tại của sản phẩm."""
	with STATUS_LOCK:
		return PRODUCT_STATUS



def image_to_base64(pil_img: Image.Image | np.ndarray) -> base64:
	"""Convert PIL or NumPy image to base64 string (PNG format), optimized for real-time usage."""
	if isinstance(pil_img, np.ndarray):
		pil_img = Image.fromarray(cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB))
	with BytesIO() as buffer:
		pil_img.save(buffer, format="PNG")
		return base64.b64encode(buffer.getvalue()).decode("utf-8")



def crop_from_bbox(
		image: Image.Image, 
		bbox: tuple[int, int, int, int]
	) -> Image.Image:
	"""Crop ảnh theo bounding box (x1, y1, x2, y2)."""
	x1, y1, x2, y2 = bbox
	return image.crop((x1, y1, x2, y2))



def resize_or_pad_img_cut_from_bbox(image: Image.Image, size=(52, 52)) -> Image.Image:
	"""Resize ảnh cắt từ bounding box theo kích thước cho trước."""
	return image.resize(size)



def describe_defect_from_bbox(full_image: Image.Image, bbox: tuple[int, int, int, int]) -> str:
	"""Mô tả ảnh cắt bởi bounding-box và gửi vào Llama-3.2-11b-Vision-Instruct."""
	cropped = crop_from_bbox(full_image, bbox)
	resized = resize_or_pad_img_cut_from_bbox(cropped, size=(224, 224))
	base64_img = image_to_base64(resized)
	### System prompt
	prompt = f"""You are a visual inspector. Please describe the defect in the following image region:
<image>{base64_img}</image>
"""
	return VISION_INSTRUCT_LLM.invoke(prompt).content



def process_frame(image: Image.Image) -> list[Image.Image, str, str]:
	"""Xử lý ảnh YOLO detect + LLM reasoning."""
	# resize_pil_img = image.resize((224, 224))
	resize_pil_img = image
	results = YOLO_OBJECT_DETECTION.predict(
		resize_pil_img,
		conf=0.,  			# phát hiện nhạy hơn, cần chỉnh cho yolo bắt được càng nhiều box nhỏ càng tốt, sau đó cắt các box nhỏ đưa vào trong LLM
		iou=0.1, 			# ít bbox trùng
		max_det=50,  		# nhiều vật thể
		imgsz=640, 			# ảnh to hơn
		vid_stride=4
	)
	processed_img = Image.fromarray(results[0].plot(pil=True)[..., ::-1])
	### Extract detection info
	detections = []
	for box in results[0].boxes:
		label = YOLO_OBJECT_DETECTION.names[int(box.cls)]
		conf = float(box.conf)
		bbox = tuple(map(int, box.xyxy[0]))
		detections.append({
			"label": label,
			"confidence": round(conf, 2),
			"bbox": bbox
		})
	product_status = "NG" if detections else "OK"
	### Tạo prompt + gọi LLM
	json_str = json.dumps(detections, indent=2)
	img_b64 = image_to_base64(resize_pil_img)
	### Tạo instruction
	instruction = VISISON_AGENT_PROMPT_MSG.format(
		json_str=json_str, image_base64=img_b64
	)
	### Lấy các bbox cắt được cho vào LLM để cho ra kết quả, rồi lấy các kết quả thu được tính toán dựa trên phương trình xác xuất.
	###################### Cần cắt các ảnh nhỏ predict được ở đây
	describe_defect_from_bbox
	llm_resp = VISION_INSTRUCT_LLM.invoke(instruction)
	return processed_img.resize(image.size), product_status, llm_resp.content



def _detect_pcb_image(image: Image.Image) -> list[Image.Image, str, str]:
	"""Nhận ảnh (PIL), xử lý bằng YOLO + LLM."""
	return process_frame(image)



def _detect_pcb_video(video_path: str):
	"""Kiểm tra lỗi từ video, tạo chuỗi prompt + gửi vào LLaMA Vision, trả về kết quả reasoning."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		yield None, "Cannot open video", ""
		return
	fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
	delay = 1 / fps 
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		pil_frame = Image.fromarray(frame[..., ::-1])
		processed_frame, product_status, reasoning = process_frame(pil_frame)
		time.sleep(delay)
		yield processed_frame, product_status, reasoning
	cap.release()
	yield None, "Video ended", ""



def _chatbot(user_query: str, history: list[dict]) -> list[dict]:
	"""Handle user input and return assistant reply in OpenAI-style format."""
	if not user_query.strip():
		return history + [{"role": "assistant", "content": "You haven't entered a message."}]
	state_data = AGENTIC.invoke(
		input={"messages": [{"role": "user", "content": user_query}]}, 
		config=CONFIG
	)
	ai_msg = get_latest_msg(state=state_data)

	history.extend([
		{"role": "user", "content": user_query},
		{"role": "assistant", "content": ai_msg.content}
	])
	return history



def main() -> None:
	"""Giao diện ứng dụng."""
	with gr.Blocks() as ui:
		gr.Markdown("# AI-Research: AutoML-MultiAgent for AOI Tasks")
		with gr.Row():
			### Reasoning Agent
			with gr.Column():
				gr.Markdown("### Reasoning Agent")
				chatbot = gr.Chatbot(
					label="Reasoning Agent", 
					height=400, 
					type="messages"
				)
				chatbot_input = gr.Textbox(
					label="Type a messages...", 
					placeholder="Ask me anything..."
				)
				chatbot_button = gr.Button("Submit")
				chatbot_button.click(
					fn=_chatbot, 
					inputs=[chatbot_input, chatbot], 
					outputs=chatbot
				)
			### Vision Agent
			with gr.Column():
				gr.Markdown("### Vision Agent")
				### Image predict
				with gr.Tab("Image Predict"):
					image_input = gr.Image(label="Upload Image", type="pil")
					image_output = gr.Image(label="Processed Image")
					status_box_img = gr.Textbox(label="Product Status", interactive=False)
					reasoning_output_img = gr.Textbox(label="Reasoning", lines=6)
					gr.Button("Analyze Image").click(
						fn=_detect_pcb_image, 
						inputs=image_input,
						outputs=[image_output, status_box_img, reasoning_output_img]
					)
				### Video predict
				with gr.Tab("Video Predict"):
					video_input = gr.File(label="Upload Video", file_types=[".mp4", ".avi"], height=120)
					video_output = gr.Image(label="Predicted Frame", streaming=True) 
					status_box = gr.Textbox(label="Product Status", value=PRODUCT_STATUS, interactive=False)
					reasoning_output = gr.Textbox(label="LLaMA3.2 Reasoning Result", lines=8)
					process_video_button = gr.Button("Start Video Inspection")
					process_video_button.click(
						fn=_detect_pcb_video,
						inputs=video_input,
						outputs=[video_output, status_box, reasoning_output]
					)
	ui.launch()



if __name__ == "__main__":
	fire.Fire(main)
