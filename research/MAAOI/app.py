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
	INSTRUCT_VISION_EXPLAIN_AGENT_PROMPT_MSG	,
	STATUS_LOCK									,
	PRODUCT_STATUS								, 
	YOLO_OBJECT_DETECTION						, 
	VISION_INSTRUCT_LLM							,
	CONFIG										, 
)
from utils import get_latest_msg



def get_status():
	"""Trả về trạng thái lỗi hiện tại của sản phẩm."""
	with STATUS_LOCK:
		return PRODUCT_STATUS



def image_to_base64(pil_img):
	"""Convert image (PIL or NumPy) to base64 string."""
	if isinstance(pil_img, np.ndarray):
		pil_img = Image.fromarray(cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB))
	buffered = BytesIO()
	try:
		pil_img.save(buffered, format="PNG")
		return base64.b64encode(buffered.getvalue()).decode("utf-8")
	finally:
		buffered.close() 



def detect_pcb_video(video_path: str):
	"""Kiểm tra lỗi từ video, tạo prompt + gửi vào LLaMA Vision, trả về cả kết quả reasoning."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		return None, "Can not open video.", ""

	fps = int(cap.get(cv2.CAP_PROP_FPS))
	delay = 1 / fps if fps > 0 else 0.03 

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break 
		### Chuyển sang PIL và dự đoán bằng YOLO
		frame_pil 			= 	Image.fromarray(frame[..., ::-1])
		frame_pil_rs 		= 	frame_pil.resize((200, 200))
		results				= 	YOLO_OBJECT_DETECTION.predict(frame_pil_rs)
		processed_frame 	= 	results[0].plot(pil=True)
		### Lấy kết quả từ YOLO
		detected_data = []
		for box in results[0].boxes:
			label 	= 	YOLO_OBJECT_DETECTION.names[int(box.cls)]
			conf 	= 	float(box.conf)
			bbox 	= 	tuple(map(int, box.xyxy[0]))
			detected_data.append({
				"label"			: 	label,
				"confidence"	: 	round(conf, 3),
				"bbox"			:	bbox
			})
		product_status = "NG" if detected_data else "OK"
		### Tạo prompt cho LLaMA
		json_str = json.dumps(detected_data, indent=2)
		image_base64 = image_to_base64(frame_pil_rs)
		inst = INSTRUCT_VISION_EXPLAIN_AGENT_PROMPT_MSG.format(json_str=json_str, image_base64=image_base64)
		### Gửi prompt vào LLM
		llm_resp = VISION_INSTRUCT_LLM.invoke(input=inst)

		time.sleep(delay)
		yield processed_frame, product_status, llm_resp.content

	cap.release()
	yield None, "Video ended.", ""



def user_interface_chatbot_resp(user_query: str, history: list[dict]) -> list[dict]:
	"""Xử lý truy vấn của người dùng và trả về hội thoại theo OpenAI-style, trên giao diện người dùng."""
	if not user_query.strip(): 
		return history + [{"role": "assistant", "content": "You have not entered any content."}]
	state_data = AGENTIC.invoke(
		input={"messages": [{"role": "user", "content": user_query}]}, config=CONFIG
	)
	ai_msg = get_latest_msg(state=state_data)

	history.append({"role": "user", "content": user_query})
	history.append({"role": "assistant", "content": ai_msg.content})

	return history



def main() -> None:
	"""Giao diện ứng dụng."""
	with gr.Blocks() as ui:
		gr.Markdown("# AI-Research: AutoML-MultiAgent for AOI Tasks")
		with gr.Row():
			### Reasoning Agent
			with gr.Column():
				gr.Markdown("### Reasoning Agent")
				chatbot = gr.Chatbot(label="Reasoning Agent", height=400, type="messages")
				chatbot_input = gr.Textbox(label="Type a messages...")
				chatbot_button = gr.Button("Submit")

				chatbot_button.click(
					fn=user_interface_chatbot_resp, inputs=[chatbot_input, chatbot], outputs=chatbot
				)
			### Vision Agent
			with gr.Column():
				gr.Markdown("### Vision Agent")
				with gr.Tab("Video Predict"):
					video_input = gr.File(label="Upload Video", file_types=[".mp4", ".avi"], height=120)
					video_output = gr.Image(label="Predicted Frame", streaming=True) 
					status_box = gr.Textbox(label="Product Status", value=PRODUCT_STATUS, interactive=False)

					reasoning_output = gr.Textbox(label="LLaMA3.2 Reasoning Result", lines=8)

					process_video_button = gr.Button("Start Video Inspection")
					process_video_button.click(
						fn=detect_pcb_video,
						inputs=video_input,
						outputs=[video_output, status_box, reasoning_output]
					)
	ui.launch()



if __name__ == "__main__":
	fire.Fire(main)
