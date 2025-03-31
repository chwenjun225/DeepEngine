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



def single_frame_detections_to_json(results: Results) -> str:
	"""Chuyển kết quả YOLOv8 từ 1 frame thành JSON, tối ưu tốc độ."""
	result = results[0]
	boxes: np.ndarray = result.boxes.xyxy.cpu().numpy().astype(int)
	confidences: np.ndarray = result.boxes.conf.cpu().numpy().astype(float)
	class_ids: np.ndarray = result.boxes.cls.cpu().numpy().astype(int)
	class_names = result.names

	frame_data = [
		{
			"bbox": {
				"x1": int(x1),
				"y1": int(y1),
				"x2": int(x2),
				"y2": int(y2), 
			},
			"confidence": round(float(conf), 3),
			"class_id": int(cls_id),
			"label": class_names[int(cls_id)]
		}
		for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confidences, class_ids)
	]

	return json.dumps(frame_data, indent=1)



async def async_process_frame(image: Image.Image) -> tuple[Image.Image, list[str]]: 
	"""Xử lý khung hình bằng YOLO và LLM với asyncio."""
	results = YOLO_OBJECT_DETECTION.predict(image, conf=0., iou=0.1, max_det=5) 
	single_frame_data = single_frame_detections_to_json(results) 
	print(single_frame_data)
	



	### chỗ này cần sửa lại, 
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
	ctx_frames = []
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		yield None, "Cannot open video", ""
		return
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break

		frame = Image.fromarray(frame[..., ::-1]).resize(resize_to) ### Cần tích đủ 10 frames
		ctx_frames.append(frame)

		if len(ctx_frames) == ctx_frames_limit: ### TODO: Viết thuật toán xử lý tiếp ở đây
			processed_img, texts = await async_process_frame(frame)
			text_combined = " | ".join([text.content if isinstance(text, BaseMessage) else str(text) for text in texts])
			yield processed_img, "NG", text_combined
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
		processed_img, texts = await async_process_frame(image)
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


"""
Cho kết quả (tọa độ, confidence) từ yolo vào trong llm, llm sẽ judge xem ảnh kết quả từ ảnh đó có đáng tin cậy không, 
các thuộc tính trong results[0]:
1. boxes (chứa tọa độ thuộc tính)
2. names (tên các nhãn)
3. orig_shape (tọa độ ban đầu của ảnh)



### frame 1 
0: 640x640 1 open_circuit, 3 shorts, 1 spur, 10.4ms
Speed: 0.9ms preprocess, 10.4ms inference, 3.0ms postprocess per image at shape (1, 3, 640, 640)
[ 
	{"bbox": {"x1": 473,"y1": 263,"x2": 485,"y2": 277},"confidence": 0.867,"class_id": 0,"label": "missing_hole"},
	{"bbox": {"x1": 236,"y1": 261,"x2": 247,"y2": 274},"confidence": 0.842,"class_id": 0,"label": "missing_hole"},
	{"bbox": {"x1": 212,"y1": 406,"x2": 228,"y2": 421},"confidence": 0.219,"class_id": 3,"label": "short"},
	{"bbox": {"x1": 160,"y1": 486,"x2": 170,"y2": 493},"confidence": 0.185,"class_id": 4,"label": "spur"},
	{"bbox": {"x1": 123,"y1": 302,"x2": 133,"y2": 314},"confidence": 0.086,"class_id": 0,"label": "missing_hole"}
], 



### frame 2
0: 640x640 1 open_circuit, 2 shorts, 2 spurs, 5.1ms
Speed: 0.9ms preprocess, 5.1ms inference, 3.2ms postprocess per image at shape (1, 3, 640, 640)
[ 
	{"bbox": {"x1": 282,"y1": 447,"x2": 290,"y2": 458},"confidence": 0.619,"class_id": 2,"label": "open_circuit"},
	{"bbox": {"x1": 248,"y1": 191,"x2": 256,"y2": 207},"confidence": 0.463,"class_id": 4,"label": "spur"},
	{"bbox": {"x1": 419,"y1": 388,"x2": 429,"y2": 406},"confidence": 0.337,"class_id": 4,"label": "spur"},
	{"bbox": {"x1": 240,"y1": 256,"x2": 251,"y2": 269},"confidence": 0.199,"class_id": 3,"label": "short"},
	{"bbox": {"x1": 147,"y1": 218,"x2": 159,"y2": 234},"confidence": 0.177,"class_id": 4,"label": "spur"}
],



### frame 3
0: 640x640 1 open_circuit, 3 shorts, 1 spur, 4.5ms
Speed: 0.8ms preprocess, 4.5ms inference, 2.8ms postprocess per image at shape (1, 3, 640, 640)
[ 
	{"bbox": {"x1": 282,"y1": 446,"x2": 290,"y2": 457},"confidence": 0.52,"class_id": 2,"label": "open_circuit"},
	{"bbox": {"x1": 418,"y1": 387,"x2": 429,"y2": 405},"confidence": 0.349,"class_id": 4,"label": "spur"},
	{"bbox": {"x1": 417,"y1": 388,"x2": 428,"y2": 405},"confidence": 0.305,"class_id": 5,"label": "spurious_copper"},
	{"bbox": {"x1": 182,"y1": 316,"x2": 191,"y2": 329},"confidence": 0.197,"class_id": 4,"label": "spur" },
	{"bbox": {"x1": 259,"y1": 324,"x2": 279,"y2": 339},"confidence": 0.061,"class_id": 3,"label": "short"}
],



### frame 4
0: 640x640 1 open_circuit, 3 shorts, 1 spur, 4.8ms
Speed: 0.8ms preprocess, 4.8ms inference, 2.1ms postprocess per image at shape (1, 3, 640, 640)
[ 
	{"bbox": {"x1": 281,"y1": 445,"x2": 290,"y2": 456},"confidence": 0.643,"class_id": 2,"label": "open_circuit"},
	{"bbox": {"x1": 419,"y1": 387,"x2": 429,"y2": 405},"confidence": 0.268,"class_id": 4,"label": "spur"},
	{"bbox": {"x1": 258,"y1": 324,"x2": 279,"y2": 340},"confidence": 0.142,"class_id": 3,"label": "short"},
	{"bbox": {"x1": 182,"y1": 315,"x2": 191,"y2": 329},"confidence": 0.135,"class_id": 4,"label": "spur"},
	{"bbox": {"x1": 418,"y1": 387,"x2": 428,"y2": 404},"confidence": 0.079,"class_id": 5,"label": "spurious_copper"}
],



### frame 5
0: 640x640 1 mouse_bite, 1 open_circuit, 3 shorts, 6.5ms
Speed: 0.8ms preprocess, 6.5ms inference, 2.2ms postprocess per image at shape (1, 3, 640, 640)
[
	{"bbox": {"x1": 251,"y1": 569,"x2": 269,"y2": 583},"confidence": 0.779,"class_id": 1,"label": "mouse_bite"},
	{"bbox": {"x1": 496,"y1": 450,"x2": 506,"y2": 471},"confidence": 0.54,"class_id": 5,"label": "spurious_copper"},
	{"bbox": {"x1": 354,"y1": 209,"x2": 366,"y2": 220},"confidence": 0.348,"class_id": 2,"label": "open_circuit"},
	{"bbox": {"x1": 126,"y1": 470,"x2": 137,"y2": 490},"confidence": 0.14,"class_id": 4,"label": "spur"},
	{"bbox": {"x1": 126,"y1": 469,"x2": 138,"y2": 489},"confidence": 0.079,"class_id": 2,"label": "open_circuit"}
]



Mỗi frame sẽ có 5 bbox được phát hiện, mỗi bbox sẽ có [xyxy, confidence, class_id, label] 

### Nếu 1 object xuất hiện nhiều lần (giống vị trí) thì ...

Với n frame, gom tất cả [xyxy, confidence, class_id] → tạo thành context_metadata, rồi đưa vào agent để đánh giá tổng thể.



📦 Dữ liệu context mẫu:
	Cấu trúc dữ liệu dạng:
	{
		"frames": [
			{
			"frame_id": 1,
			"detections": [
				[473, 263, 485, 277, 0.867, 0],  // [x1, y1, x2, y2, confidence, class_id]
				[236, 261, 247, 274, 0.842, 0],
				...
			]
			},
			...
			{
				"frame_id": 10,
				"detections": [...]
			}
		]
	}



🧠 Mục đích reasoning agent:
	Xem các object có “ổn định” không? (confidence tăng/giảm?)

	Từ đó, cho kết luận OK / NG
"""
