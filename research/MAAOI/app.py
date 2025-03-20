import fire 
import gradio as gr 
import cv2 
import numpy as np 
import time 
import threading 
from queue import Queue 



from langchain_core.messages import HumanMessage



from agentic import AGENTIC
from const_vars import QUERIES, CONFIG, DEBUG
from state import default_messages



STATUS = "..."
FRAME_QUEUE = Queue()



def display_conversation_results(messages: dict) -> None:
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
			raise ValueError(f"`msgs` phải là một dictionary chứa danh sách tin nhắn, `msgs` hiện tại là: {msgs}")



def chatbot_resp() -> str:
	"""Xử lý truy vấn của người dùng và hiển thị phản hồi từ AI."""
	for i, user_query in enumerate(QUERIES, 1):
		if DEBUG:
			print(f"\n👨_query_{i}:")
			print(user_query)
			print("\n🤖_response:")
		user_query = user_query.strip()
		if user_query.lower() == "exit": break
		state_data = AGENTIC.invoke(input={"human_query": [HumanMessage(user_query)], "messages": default_messages()}, config=CONFIG)
		if not isinstance(state_data, dict): raise ValueError("[ERROR]: app.invoke() không trả về dictionary.")
		if "messages" not in state_data: raise ValueError("[ERROR]: Key 'messages' không có trong kết quả.")
		messages = state_data["messages"]
		if DEBUG:
			print("\n===== [CONVERSATION RESULTS] =====\n")
			display_conversation_results(messages)
			print("\n===== [END OF CONVERSATION] =====\n")



def camera_feed():
	"""Hiển thị video từ camera."""
	if FRAME_QUEUE.empty():
		return None
	return FRAME_QUEUE.get()



def detect_pcb_camera():
	"""Kiểm tra lỗi từ camera."""
	global status_text
	cap = cv2.VideoCapture(0)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		status_text = np.random.choice(["OK", "NG", "..."], p=[0.7, 0.2, 0.1])
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		FRAME_QUEUE.put(frame)
		time.sleep(0.1)
	cap.release()



def detect_pcb_video(video_path):
	"""Kiểm tra lỗi từ video."""
	global status_text
	cap = cv2.VideoCapture(video_path)
	frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		status_text = np.random.choice(["OK", "NG", "..."], p=[0.7, 0.2, 0.1])
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frames.append(frame)
		time.sleep(0.1)
	cap.release()
	return frames



def main() -> None:
	"""Giao diện ứng dụng."""
	with gr.Blocks() as ui:
		gr.Markdown("**AutoML-MultiAgent for AOI Tasks**")

		with gr.Row():
			with gr.Column():
				gr.Markdown("**Chat Agent**")
				chat_agent_input = gr.Textbox(label="Ask Agent anything")
				chat_agent_output = gr.Textbox(label="Agent response", interactive=False)
				chat_agent_button = gr.Button("Send")
				chat_agent_button.click(
					fn=chatbot_resp, 
					inputs=chat_agent_input, 
					outputs=chat_agent_output
				)
			with gr.Column():
				gr.Markdown("**Vision Agent**")
				choose_mode = gr.Radio(["Camera", "Video"], label="Choose mode")
				if "cam":
					cam_feed = gr.Image(label="Camera Realtime", streaming=True)
					start_cam_button = gr.Button("Start Camera")
					start_cam_button.click(fn=camera_feed, outputs=cam_feed)
				if "vid":
					video_input = gr.File(label="Upload Video", file_types=[".mp4", ".avi"])
					video_output = gr.Video(label="Start Video")
					process_video_button = gr.Button("Inspect from Video")
					process_video_button.click(
						fn=detect_pcb_video, 
						inputs=video_input,
						outputs=video_output
					)
				status_box = gr.Textbox(label="Product State", value=STATUS, interactive=False)
		cam_thread = threading.Thread(target=detect_pcb_camera, daemon=True)
		cam_thread.start()
	ui.launch()



if __name__ == "__main__":
	fire.Fire(main)