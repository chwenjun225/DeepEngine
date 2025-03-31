import json
import asyncio
import cv2 
import fire 
import numpy as np 
import gradio as gr 
from PIL import Image 



from langchain_core.messages import BaseMessage



from ultralytics.engine.results import Results



from agentic import AGENTIC
from const_vars import (
	VISUAL_AGENT_PROMPT_MSG		,
	CONFIG						,
	PRODUCT_STATUS				,
	STATUS_LOCK					,

	YOLO_OBJECT_DETECTION		,
	VISION_INSTRUCT_LLM			,
)
from utils import (
	image_to_base64				,
	expand_bbox					,
	get_latest_msg				,
)



def get_status_product() -> str:
	"""Trả về trạng thái lỗi hiện tại của sản phẩm."""
	with STATUS_LOCK:
		return PRODUCT_STATUS



async def async_llm_inference(prompt):
	"""Thực thi LLM không chặn."""
	return await asyncio.to_thread(
		VISION_INSTRUCT_LLM.invoke, prompt
	)



def single_frame_detections_to_json(results:Results, frame_id:int) -> str:
	"""Chuyển kết quả YOLOv8 từ 1 frame thành JSON, tối ưu tốc độ."""
	result = results[0]
	boxes: np.ndarray = result.boxes.xyxy.cpu().numpy().astype(int)
	confidences: np.ndarray = result.boxes.conf.cpu().numpy().astype(float)
	class_ids: np.ndarray = result.boxes.cls.cpu().numpy().astype(int)
	class_names = result.names
	frame_data = {
		"id": frame_id, 
		"metadata": [
			{
				"bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
				"confidence": round(float(conf), 3),
				"class_id": int(cls_id), 
				"label": class_names[int(cls_id)]
			}
			for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confidences, class_ids)
		]
	}
	return json.dumps(frame_data, indent=1)



async def async_process_frames(ctx_frames:list[Image.Image]) -> tuple[Image.Image, list[str]]: 
	"""Xử lý khung hình bằng YOLO và LLM với asyncio."""
	ctx_frames_metadata = [] ### Khởi tạo context-frames-metadata
	for idx, frame in enumerate(ctx_frames):
		results = YOLO_OBJECT_DETECTION.predict(frame, conf=0., iou=0.1, max_det=5) 
		frame_metadata = single_frame_detections_to_json(results, idx) 
		ctx_frames_metadata.append(frame_metadata)


	# TODO: Ngày mai cần build nhanh một hệ thống Multi-Agent ở đây để lý giải tính toán các "ctx_frame_metadata"
	print("ctx_frames_metadata:", ctx_frames_metadata) 
	print("DEBUG")

	### IMPORTANT: Kết quả đầu ra phải là một list["Ảnh pil đã vẽ các dấu hiệu lỗi","PRODUCT_STATUS OK hoặc NG",  "Lý luận để hiển thị"]

























	### Đây là hình ảnh plot kết quả lên yolo, bây giờ chưa cần thiết
	processed_img = Image.fromarray(results[0].plot()[..., ::-1]) 
	bboxes = [list(map(int, box.xyxy[0])) for box in results[0].boxes] 
	tasks = []
	for bbox in bboxes: 
		bbox_expanded = expand_bbox(bbox)
		cropped = image.crop(bbox_expanded) 
		b64_img = image_to_base64(cropped) 
		prompt = VISUAL_AGENT_PROMPT_MSG.format(base64_image=b64_img)
		tasks.append(async_llm_inference(prompt))
	texts = await asyncio.gather(*tasks)
	return processed_img, texts



async def async_video_processing(video_path:str, resize_to:tuple=(640, 640), ctx_frames_limit:int=10) -> any:
	"""Xử lý video không chặn bằng asyncio."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		yield None, "Cannot open video", ""
		return

	ctx_frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: 
			break

		frame = Image.fromarray(frame[..., ::-1]).resize(resize_to) 
		ctx_frames.append(frame) 

		if len(ctx_frames) == ctx_frames_limit: ### Nếu tích đủ 10 frames thì xử lý 
			processed_img, texts = await async_process_frames(ctx_frames) ### Xử lý asyncio với Multi-Agent System
			text_combined = " | ".join([
				text.content 
					if isinstance(text, BaseMessage) 
					else str(text) 
					for text in texts
			]) ### Chuyển kết quả reasoning ra giao diện 
			yield processed_img, "NG", text_combined ### Stream kết quả ra giao diện 
			ctx_frames.clear() ### Làm sạch context-frames 
	cap.release()
	yield None, "Video ended", ""



def __detect_pcb_video(video_path: str) -> any:
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



async def __detect_pcb_image(image: Image.Image) -> any:
	"""Xử lý ảnh PCB bằng YOLO và LLM."""
	try:
		processed_img, texts = await async_process_frames(image)
		text_combined = " | ".join([
			text.content \
				if isinstance(text, BaseMessage) \
				else str(text) \
				for text in texts
		])
		status = get_status_product()
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
		gr.Markdown("# AI-Research --- AutoML-MultiAgent for AOI Tasks --- Owner: 陳文俊 --- V1047876")
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
