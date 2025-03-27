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
	VISISON_AGENT_PROMPT_MSG	,
	STATUS_LOCK					,
	PRODUCT_STATUS				, 
	YOLO_OBJECT_DETECTION		, 
	VISION_INSTRUCT_LLM			,
	CONFIG						, 
)
from utils import get_latest_msg



def get_status() -> str:
	"""Trả về trạng thái lỗi hiện tại của sản phẩm."""
	with STATUS_LOCK:
		return PRODUCT_STATUS



def image_to_base64(pil_img) -> base64:
	"""Convert image (PIL or NumPy) to base64 string."""
	if isinstance(pil_img, np.ndarray):
		pil_img = Image.fromarray(cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB))
	buffered = BytesIO()
	try:
		pil_img.save(buffered, format="PNG")
		return base64.b64encode(buffered.getvalue()).decode("utf-8")
	finally:
		buffered.close() 



def process_frame(image: Image.Image) -> list[Image.Image, str, str]:
	"""Xử lý ảnh YOLO detect + LLM reasoning."""
	resize_pil_img = image.resize((200, 200))
	results = YOLO_OBJECT_DETECTION.predict(resize_pil_img)
	processed_img = Image.fromarray(results[0].plot(pil=True)[..., ::-1])
	### Extract detection info
	detected_data = []
	for box in results[0].boxes:
		label = YOLO_OBJECT_DETECTION.names[int(box.cls)]
		conf = float(box.conf)
		bbox = tuple(map(int, box.xyxy[0]))
		detected_data.append({
			"label": label,
			"confidence": round(conf, 2),
			"bbox": bbox
		})
	product_status = "NG" if detected_data else "OK"
	### Tạo prompt + gọi LLM
	json_str = json.dumps(detected_data, indent=2)
	image_base64 = image_to_base64(resize_pil_img)
	### Tạo instruction
	instruction = VISISON_AGENT_PROMPT_MSG.format(
		json_str=json_str, image_base64=image_base64
	)
	llm_resp = VISION_INSTRUCT_LLM.invoke(instruction)
	return [processed_img.resize(image.size), product_status, llm_resp.content]



def gr_detect_pcb_image(image: Image.Image) -> list[Image.Image, str, str]:
	"""Nhận ảnh (PIL), xử lý bằng YOLO + LLM."""
	return process_frame(image)



def gr_detect_pcb_video(video_path: str) -> any:
	"""Kiểm tra lỗi từ video hoặc ảnh, tạo prompt + gửi vào LLaMA Vision, trả về cả kết quả reasoning."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		yield None, "Can not open video.", ""
		return

	fps = int(cap.get(cv2.CAP_PROP_FPS))
	delay = 1 / fps if fps > 0 else 0.03 

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		pil_frame = Image.fromarray(frame[..., ::-1])
		processed_frame, product_status, reasoning = \
			process_frame(pil_frame)

		time.sleep(delay)
		yield processed_frame, product_status, reasoning

	cap.release()
	yield None, "Video ended.", ""



def gr_chatbot(user_query: str, history: list[dict]) -> list[dict]:
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
					fn=gr_chatbot, 
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
						fn=gr_detect_pcb_image, 
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
						fn=gr_detect_pcb_video,
						inputs=video_input,
						outputs=[video_output, status_box, reasoning_output]
					)
	ui.launch()



if __name__ == "__main__":
	fire.Fire(main)
