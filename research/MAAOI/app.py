import asyncio
import cv2 
import numpy as np 
import base64
import fire 
import gradio as gr 
from io import BytesIO
from PIL import Image 



from langchain_core.messages import BaseMessage



from agentic import AGENTIC
from const_vars import (
	VISUAL_AGENT_PROMPT_MSG		,
	YOLO_OBJECT_DETECTION		,
	VISION_INSTRUCT_LLM			,
	CONFIG						,
	PRODUCT_STATUS				,
	STATUS_LOCK					, 
)
from utils import get_latest_msg



def get_status() -> str:
	"""Trả về trạng thái lỗi hiện tại của sản phẩm."""
	with STATUS_LOCK:
		return PRODUCT_STATUS



def image_to_base64(pil_img: Image.Image | np.ndarray) -> str:
	"""Convert PIL or NumPy image to base64 string (PNG format), optimized for real-time usage."""
	with BytesIO() as buffer:
		pil_img.save(buffer, format="PNG")
		return base64.b64encode(buffer.getvalue()).decode("utf-8")



async def async_llm_inference(prompt):
	"""Thực thi LLM không chặn."""
	return await asyncio.to_thread(VISION_INSTRUCT_LLM.invoke, prompt)



async def async_process_frame(image: Image.Image): 
	"""Xử lý khung hình bằng YOLO và LLM với asyncio."""
	results = YOLO_OBJECT_DETECTION.predict(image, conf=0., iou=0.1, max_det=5) 
	processed_img = Image.fromarray(results[0].plot(pil=True)[..., ::-1]) 
	bboxes = [tuple(map(int, box.xyxy[0])) for box in results[0].boxes] 
	tasks = []
	for bbox in bboxes: 
		cropped = image.crop(bbox)
		b64_img = image_to_base64(cropped)
		prompt = VISUAL_AGENT_PROMPT_MSG.format(base64_image=b64_img)
		tasks.append(async_llm_inference(prompt))
	texts = await asyncio.gather(*tasks)
	return processed_img, texts



async def async_video_processing(video_path: str):
	"""Xử lý video không chặn bằng asyncio."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		yield None, "Cannot open video", ""
		return
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		pil_frame = Image.fromarray(frame[..., ::-1]).resize((640, 640))
		processed_img, texts = await async_process_frame(pil_frame)
		text_combined = " | ".join([text.content if isinstance(text, BaseMessage) else str(text) for text in texts])
		yield processed_img, "NG", text_combined
	cap.release()
	yield None, "Video ended", ""



def __detect_pcb_video(video_path: str):
	"""Giao diện Gradio để xử lý video."""
	try:
		loop = asyncio.get_running_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	async def async_generator_wrapper():
		async for processed_img, status, text_combined in async_video_processing(video_path):
			yield processed_img, status, text_combined
	async_gen = async_generator_wrapper()
	while True:
		try:
			result = loop.run_until_complete(async_gen.__anext__())
			yield result
		except StopAsyncIteration:
			break



async def __detect_pcb_image(image: Image.Image):
	"""Xử lý ảnh PCB bằng YOLO và LLM."""
	try:
		processed_img, texts = await async_process_frame(image)
		text_combined = " | ".join([
			text.content \
				if isinstance(text, BaseMessage) \
				else str(text) \
				for text in texts
		])
		status = get_status()
		return processed_img, status, text_combined
	except Exception as e:
		return None, "Error during image processing", str(e)



def __chatbot(user_query: str, history: list[dict]) -> list[dict]:
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
					fn=__chatbot, 
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
						fn=__detect_pcb_image, 
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
						fn=__detect_pcb_video, 
						inputs=video_input,
						outputs=[video_output, status_box, reasoning_output]
					)
	ui.launch()



if __name__ == "__main__":
	fire.Fire(main)