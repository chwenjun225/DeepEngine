import cv2 
import fire 
import asyncio
import numpy as np 
import gradio as gr 
from PIL import Image 



from langchain_core.messages import BaseMessage, AIMessage



from ultralytics.engine.results import Results



from agentic import AGENTIC
from const import (
	CONFIG						,
	PRODUCT_STATUS				,
	SAVE_FRAME_RESULTS			,

	YOLO_OBJECT_DETECTION		,
	VISION_LLM					,
)
from utils import (
	get_latest_msg				,
	draw_defect_overlay			,
	save_frame_result			,
	measure_time				,
)



async def async_llm_inference(prompt):
	"""Thực thi LLM không chặn."""
	return await asyncio.to_thread(VISION_LLM.invoke, prompt)



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
	return frame_data



@measure_time("Process 10 Frames")
async def async_process_frames(ctx_frames:list[Image.Image]) -> tuple[Image.Image, str, list[str]]: 
	"""Xử lý khung hình bằng YOLO và LLM với asyncio."""

	ctx_frames_metadata = [] 
	for idx, frame in enumerate(ctx_frames):
		results = YOLO_OBJECT_DETECTION.predict(frame, conf=0., iou=0.1, max_det=5, verbose=False) 
		frame_metadata = single_frame_detections_to_json(results, idx) 
		ctx_frames_metadata.append(frame_metadata)

	response = AGENTIC.invoke(
		input={"VISION_AGENT_MSGS": [AIMessage(
			content=ctx_frames_metadata, 
			name="VISION_AGENT"
		)]}, 
	config=CONFIG)

	visual_metadata = get_latest_msg(response, "VISUAL_AGENT_MSGS").content
	if isinstance(visual_metadata, str):
		visual_metadata = eval(visual_metadata)

	PRODUCT_STATUS: str = visual_metadata["ngok"]
	bbox_per_frame: dict = visual_metadata["bbox"]

	last_idx = len(ctx_frames) - 1
	last_frame_pil = ctx_frames[last_idx]
	last_frame_np = cv2.cvtColor(
		np.array(last_frame_pil), 
		cv2.COLOR_RGB2BGR
	)

	last_bboxes = []
	for frame_bboxes in bbox_per_frame.values():
		last_bboxes.extend(frame_bboxes)

	annotated_np = draw_defect_overlay(last_frame_np, last_bboxes, PRODUCT_STATUS)
	annotated_pil = Image.fromarray(cv2.cvtColor(annotated_np, cv2.COLOR_BGR2RGB))

	if SAVE_FRAME_RESULTS:
		save_frame_result(annotated_pil, f"frame__{last_idx:03d}__.jpg", PRODUCT_STATUS)

	reasoning_texts = [f"{obj['label']} at {obj['bbox']}" for obj in last_bboxes]
	if PRODUCT_STATUS:
		reasoning_texts.insert(0, f"{PRODUCT_STATUS}")
	return annotated_pil, PRODUCT_STATUS, reasoning_texts



async def async_video_processing(video_path:str, resize_to:tuple=(640, 640), ctx_frames_limit:int=10) -> any:	
	"""Xử lý video không chặn bằng asyncio."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(">>> Cannot open video...")
		yield None, "Cannot open video", ""
		return

	ctx_frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: 
			print(">>> Video Ended...")
			break

		frame = Image.fromarray(frame[..., ::-1]).resize(resize_to) 
		ctx_frames.append(frame) 

		if False:
			preview_frame = frame.copy()
			yield preview_frame, "WAITING", "Analyzing..."

		if len(ctx_frames) == ctx_frames_limit: 
			annotated_pil, PRODUCT_STATUS, reasoning_texts = await async_process_frames(ctx_frames) 
			reasoning_text_combined = " | ".join([
				t.content 
					if isinstance(t, BaseMessage) 
					else str(t) 
					for t in reasoning_texts
			])
			yield annotated_pil, PRODUCT_STATUS, reasoning_text_combined
			ctx_frames.clear()
	cap.release()



def __detect_pcb_video__(video_path: str) -> any:
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



async def __detect_pcb_image__(image: Image.Image) -> any:
	"""Xử lý ảnh PCB bằng YOLO và LLM."""
	try:
		processed_img, PRODUCT_STATUS, texts = await async_process_frames(image)
		text_combined = " | ".join([
			text.content \
				if isinstance(text, BaseMessage) \
				else str(text) \
				for text in texts
		])
		return processed_img, PRODUCT_STATUS, text_combined
	except Exception as e:
		return None, "Error during image processing", str(e)



def main() -> None:
	"""Giao diện ứng dụng AOI Multi-Agent."""
	with gr.Blocks(title="AOI Multi-Agent QC System") as user_interface:
		gr.Markdown(
			"# CPEG-AI-Research: MultiAgent for AOI Tasks\n"
			"#### Owner: Tran Van Tuan - V1047876"
		)
		### IMAGE PREDICTION TAB 
		with gr.Tab("Image Prediction"):
			with gr.Row():
				### LEFT: Processed image
				image_output = gr.Image(label="Processed Image", scale=1, height=550, show_label=False)
				### RIGHT: Input & Outputs
				with gr.Column(scale=1):
					image_input = gr.Image(label="Upload Image", type="pil", height=160)
					analyze_img_btn = gr.Button("Analyze Image")
					status_box_img = gr.Textbox(label="Product Status", interactive=False)
					reasoning_output_img = gr.Textbox(label="Reasoning Explanation", lines=6)
					analyze_img_btn.click(fn=__detect_pcb_image__, inputs=image_input, outputs=[image_output, status_box_img, reasoning_output_img])
		### VIDEO PREDICTION TAB 
		with gr.Tab("Video Prediction"):
			with gr.Row():
				### LEFT: Predicted frame
				video_output = gr.Image(streaming=True, scale=1, height=550, show_label=False)
				### RIGHT: Upload & Info
				with gr.Column(scale=1):
					video_input = gr.File(label="Upload Video (.mp4, .avi)", file_types=[".mp4", ".avi"], height=120)
					process_video_button = gr.Button("Start Video Inspection")
					status_box = gr.Textbox(label="Product Status", value=PRODUCT_STATUS, interactive=False)
					reasoning_output = gr.Textbox(label="Reasoning Explanation", lines=6)
					process_video_button.click(fn=__detect_pcb_video__, inputs=video_input, outputs=[video_output, status_box, reasoning_output])
	user_interface.launch()



if __name__ == "__main__":
	fire.Fire(main)


"""
1. Đã thu thập ~7000 sample PCB images
2. Viết script chạy evals

IMPORTANT: Hiện tại cần tăng tốc viết bài báo khoa học rồi nộp cho taos, vậy làm sao để viết:
1. Bắt đầu với phần tiêu đề pattern là AutoML-AOI: A Multi Agent Large Language Model Framework for Automated Optical Inspection Printed Circuit Board in Realtime 

2. Viết phần abstract 

"""
