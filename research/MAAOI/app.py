import json
import fire 
import gradio as gr 
import cv2 
import numpy as np 
import time 
import threading 
from queue import Queue 
from typing_extensions import Tuple, Dict, List
from PIL import Image 



from ultralytics import YOLO 



from langchain_core.messages import HumanMessage



from agentic import AGENTIC
from const_vars import QUERIES, CONFIG
from state import default_messages
from utils import get_latest_msg



PRODUCT_STATUS = "..."
FRAME_QUEUE = Queue()
MESSAGES = []
STATUS_LOCK = threading.Lock()
OPEN_VISION_AGENT = False
YOLO_MODEL = YOLO("/home/chwenjun225/projects/DeepEngine/research/MAAOI/VisionAgent/runs/detect/train/weights/best.pt")



def detect_pcb_video(video_path: str):
	"""Kiểm tra lỗi từ video & hiển thị từng frame lên giao diện."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		return None, "Can not open video."

	fps = int(cap.get(cv2.CAP_PROP_FPS))
	delay = 1 / fps if fps > 0 else 0.03 

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break 

		frame_pil = Image.fromarray(frame[..., ::-1])
		results = YOLO_MODEL.predict(frame_pil)
		processed_frame = results[0].plot(pil=True)

		detected_classes = [YOLO_MODEL.names[int(box.cls)] for box in results[0].boxes]
		product_status = "NG" if detected_classes else "OK"

		time.sleep(delay)
		yield processed_frame, product_status
	cap.release()
	yield None, "Video ended."



def get_status():
	"""Trả về trạng thái lỗi hiện tại của sản phẩm."""
	with STATUS_LOCK:
		return PRODUCT_STATUS



def gr_chatbot_resp(user_input: str, history: list) -> str:
	"""Xử lý truy vấn của người dùng và trả về hội thoại theo OpenAI-style."""

	if not user_input.strip(): return history + [{"role": "assistant", "content": "You have not entered any content."}]
	state_data = AGENTIC.invoke(
		input={"human_query": [HumanMessage(user_input)], "messages": default_messages()},
		config=CONFIG
	)
	check_req_ver = get_latest_msg(state=state_data, node="REQUEST_VERIFY", msgs_type="AI")

	if check_req_ver.content == "NO":
		ai_resp = get_latest_msg(state=state_data, node="MANAGER_AGENT", msgs_type="AI")
		history.append({"role": "user", "content": user_input})
		history.append({"role": "assistant", "content": ai_resp.content})
		return history

	ai_resp = get_latest_msg(state=state_data, node="PROMPT_AGENT", msgs_type="AI")
	history.append({"role": "user", "content": user_input})
	history.append({"role": "assistant", "content": f"I understood! To ensure higher accuracy, I will collaborate with the {json.loads(ai_resp.content)['tool_execution']} to analyze the PCB defects together. The result of output will be OK or NG."})
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
				chatbot_button.click(fn=gr_chatbot_resp, inputs=[chatbot_input, chatbot], outputs=chatbot)
			### Vision Agent 
			with gr.Column():
				gr.Markdown("### Vision Agent")
				with gr.Tab("Video Predict"):
					video_input = gr.File(label="Upload Video", file_types=[".mp4", ".avi"], height=120)
					video_output = gr.Image(label="Predicted Frame", streaming=True) 
					status_box = gr.Textbox(label="Product Status", value=PRODUCT_STATUS, interactive=False)
					process_video_button = gr.Button("Start Video Inspection")
					
					process_video_button.click(fn=detect_pcb_video, inputs=video_input, outputs=[video_output, status_box])
	ui.launch()



if "On Terminal":
	def display_conversation_results_terminal(messages: dict) -> None:
		"""Hiển thị kết quả hội thoại từ tất cả các agent trong hệ thống."""
		if not messages:
			print("[INFO]: Không có tin nhắn nào trong hội thoại.")
			return
		for node, msgs in messages.items():
			print(f"\n[{node}]")
			if isinstance(msgs, dict):
				for msg_category, msg_list in msgs.items():
					if msg_list:
						print(f"  {msg_category}:")
						for msg in msg_list:
							content = getattr(msg, "content", "[No content]")
							print(f"\t- {content}")
			else:
				raise ValueError(f"msgs phải là một dictionary chứa danh sách tin nhắn, msgs hiện tại là: {msgs}")



	def chatbot_resp_terminal() -> str:
		"""Xử lý truy vấn của người dùng và hiển thị phản hồi từ AI."""
		for i, user_query in enumerate(QUERIES, 1):
			print(f"\n👨_query_{i}:")
			print(user_query)
			print("\n🤖_response:")
			user_query = user_query.strip()
			if user_query.lower() == "exit": 
				break
			state_data = AGENTIC.invoke(input={"human_query": [HumanMessage(user_query)], 
					"messages": default_messages()}, config=CONFIG)
			if not isinstance(state_data, dict): raise ValueError("[ERROR]: app.invoke() không trả về dictionary.")
			if "messages" not in state_data: raise ValueError("[ERROR]: Key 'messages' không có trong kết quả.")
			messages = state_data["messages"]
			print("\n===== [CONVERSATION RESULTS] =====\n")
			display_conversation_results_terminal(messages)
			print("\n===== [END OF CONVERSATION] =====\n")



if __name__ == "__main__":
	fire.Fire(main)
