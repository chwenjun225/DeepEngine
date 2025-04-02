import asyncio
import cv2 
import fire 
import numpy as np 
import gradio as gr 
from PIL import Image 



from langchain_core.messages import BaseMessage, AIMessage



from ultralytics.engine.results import Results



from agentic import AGENTIC
from const import (
	CONFIG						,
	PRODUCT_STATUS				,

	YOLO_OBJECT_DETECTION		,
	VISION_LLM					,
)
from utils import (
	get_latest_msg				,
	draw_defect_overlay			,
)



async def async_llm_inference(prompt):
	"""Thực thi LLM không chặn."""
	return await asyncio.to_thread(
		VISION_LLM.invoke, prompt
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
	return frame_data



async def async_process_frames(ctx_frames:list[Image.Image]) -> tuple[Image.Image, list[str]]: 
	"""Xử lý khung hình bằng YOLO và LLM với asyncio."""
	ctx_frames_metadata = [] 
	for idx, frame in enumerate(ctx_frames):
		results = YOLO_OBJECT_DETECTION.predict(frame, conf=0., iou=0.1, max_det=5) 
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

	PRODUCT_STATUS = visual_metadata["ngok"]
	bbox_per_frame = visual_metadata["bbox"]

	last_idx = len(ctx_frames) - 1
	last_frame_pil = ctx_frames[last_idx]
	last_frame_np = cv2.cvtColor(
		np.array(last_frame_pil), 
		cv2.COLOR_RGB2BGR
	)

	last_bboxes = bbox_per_frame[last_idx] ### TODO: Cần sửa lại chỗ này, phải lấy toàn bộ các bbox từ tất cả 10 frames để vẽ lên ảnh cuối 

	annotated_np = draw_defect_overlay(
		last_frame_np, 
		last_bboxes, 
		PRODUCT_STATUS
	)
	annotated_pil = Image.fromarray(cv2.cvtColor(annotated_np, cv2.COLOR_BGR2RGB))

	texts = [f"{obj['label']} at {obj['bbox']}" for obj in last_bboxes]
	if PRODUCT_STATUS:
		texts.insert(0, f"{PRODUCT_STATUS}")
	return annotated_pil, PRODUCT_STATUS, texts



async def async_video_processing(video_path:str, resize_to:tuple=(640, 640), ctx_frames_limit:int=10) -> any:	
	"""Xử lý video không chặn bằng asyncio."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		yield None, "Cannot open video", ""
		return

	ctx_frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break

		frame = Image.fromarray(frame[..., ::-1]).resize(resize_to) 
		ctx_frames.append(frame) 

		if False:
			preview_frame = frame.copy()
			yield preview_frame, "WAITING", "Analyzing..."

		if len(ctx_frames) == ctx_frames_limit: 
			processed_img, PRODUCT_STATUS, texts = await async_process_frames(ctx_frames) 
			text_combined = " | ".join([
				text.content 
					if isinstance(text, BaseMessage) 
					else str(text) 
					for text in texts
			])
			yield processed_img, PRODUCT_STATUS, text_combined
			ctx_frames.clear()
	cap.release()



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
					analyze_img_btn.click(fn=__detect_pcb_image, inputs=image_input, outputs=[image_output,status_box_img,reasoning_output_img])
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
					process_video_button.click(fn=__detect_pcb_video, inputs=video_input, outputs=[video_output, status_box, reasoning_output])
	user_interface.launch()



if __name__ == "__main__":
	fire.Fire(main)


"""
Hiện tại đã hoàn thiện một số chức năng cơ bản. Tuy nhiên vẫn còn một số điểm cần cải thiện:

Task:
	1. Stream Frame: Quá trình tích lũy frame ta cũng sẽ stream frame đó lên màn ảnh, nhằm đảm bảo độ mượt mà, tăng cường trải nghiệm người dùng.
	-> Câu hỏi: Làm thế nào để vừa tích lũy frame và vừa stream frame. Đến khi có kết quả thì frame thứ 10 sẽ show ra kết quả ===> Có thể chưa cần thiết

	2. Nếu NG thì cần vẽ chữ NG màu đỏ in đậm, cùng các box màu đỏ lên hình ảnh cuối trong ctx_frames. Nếu OK thì cần hiển thị chữ OK màu xanh in đậm. ===> Xong


	3. Tiếp tục nghiên cứu async để cải thiện tốc độ. 


	4. Nếu Agent không cho ra kết quả là OK hoặc NG thì bắt Agent phải inference lại. ===> Xong


	5. Cần chuẩn bị dữ liệu để xác nhận độ chính xác, 100 ảnh NG, 100 ảnh OK.
"""
"""
Cần hoàn thành số liệu để báo cáo cho ngày mai:
1. Thời gian inference cho ra kết quả, tốc độ inference là bao nhiêu ms
2. TP-FP-TN-FN
"""