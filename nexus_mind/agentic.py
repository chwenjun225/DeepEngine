# Tài liệu liên quan:  
# 	Giới thiệu ngắn gọn về nguyên lý ReAct Prompting, không bao gồm mã nguồn:  
#		https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_prompt.md  
#   Triển khai ReAct Prompting dựa trên giao diện `model.chat` 
# 			(chế độ đối thoại), bao gồm cả tích hợp công cụ với LangChain:  
#		https://github.com/QwenLM/Qwen-7B/blob/main/examples/langchain_tooluse.ipynb  
#   Triển khai ReAct Prompting dựa trên giao diện `model.generate` 
# 			(chế độ tiếp tục viết), phức tạp hơn so với chế độ chat:  
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_demo.py (tệp này)  
# 	Tài liệu sử dụng ChatOllama làm agent_core:
# 		 https://python.langchain.com/docs/integrations/providers/ollama/



from datetime import datetime 
import tqdm
import requests
import json
import json5
import fire 
from PIL import Image
from io import BytesIO



from pydantic import BaseModel, Field



from langchain.tools import StructuredTool 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.fake import FakeStreamingListLLM
from langchain_core.messages import (AIMessage, HumanMessage, ToolMessage)



# Constant vars 
TOOLS = [
	{
		"name_for_human": "image_to_text", 
		"name_for_model": "image_to_text", 
		"description_for_model": "image_to_text is a service that generates textual descriptions from images. By providing the URL of an image, it returns a detailed and realistic description of the image.",
		"parameters": [
			{
				"name": "image_path",
				"description": "the URL of the image to be described",
				"required": True,
				"schema": {"type": "string"},
			}
		],
	},
	{
		"name_for_human": "text_to_image",
		"name_for_model": "text_to_image",
		"description_for_model": "text_to_image is an AI image generation service. It takes a text description as input and returns a URL of the generated image.",
		"parameters": [
			{
				"name": "text",
				"description": "english keywords or a text prompt describing what you want in the image.",
				"required": True,
				"schema": {"type": "string"}
			}
		]
	},
	{
		"name_for_human": "modify_text",
		"name_for_model": "modify_text",
		"description_for_model": "modify_text changes the original prompt based on the input request to make it more suitable.",
		"parameters": [
			{
				"name": "describe_before",
				"description": "the prompt or image description before modification.",
				"required": True,
				"schema": {"type": "string"}
			},
			{
				"name": "modification_request",
				"description": "the request to modify the prompt or image description, e.g., change 'cat' to 'dog' in the text.",
				"required": True,
				"schema": {"type": "string"}
			}
		]
	}, 
	{
		"name_for_human": "ai_vision", 
		"name_for_model": "ai_vision", 
		"description_for_model": "ai_vision is a service that detects and extracts characters from a product image. It processes the image and returns the recognized text to the AI agent for further analysis.",
		"parameters": [
			{
				"name": "video_path",
				"description": "The file path of a video or the IP address of a camera for real-time character detection.",
				"required": True,
				"schema": {"type": "string"}
			}
		],
	},
]



FAKE_RESPONSES = [
# "Hello, Good afternoon!", -- Fake resp id 0 -- No tool
	"""
	Thought: The input is a greeting. No tools are needed.
	Final Answer: Hello! Good afternoon! How can I assist you today?
	""", 
# "Who is Jay Chou?", -- Fake resp id 1 -- No tool 
	"""
	Thought: The user is asking for information about Jay Chou. I should retrieve general knowledge.
	Final Answer: Jay Chou is a Taiwanese singer, songwriter, and actor, widely known for his influence in Mandopop music. He has released numerous albums and is recognized for his unique blend of classical and contemporary music.
	""", 
# "Who is his wife?", -- Fake resp id 2 -- No tool 
	"""
	Thought: The previous question was about Jay Chou. "His wife" likely refers to Jay Chou's spouse.
	Final Answer: Jay Chou's wife is Hannah Quinlivan, an actress and model from Taiwan.
	""", 
# "Describe what is in this image, this is URL of the image: https://www.night_city_img.com", --> Fake resp id 3 -- Tool-use: image_to_text
	"""
	Thought: The user wants a description of an image. I should use the image_to_text API.
	Action: image_to_text
	Action Input: {"image_path": "https://www.tushengwen.com"}
	Observation: "The image depicts a vibrant cityscape at night, illuminated by neon lights and tall skyscrapers."
	Thought: I now know the final answer.
	Final Answer: The image depicts a vibrant cityscape at night, illuminated by neon lights and tall skyscrapers.
	""",
# "Draw me a cute kitten, preferably a black cat", -- Fake resp id 4 -- Tool-use: text_to_image
	"""
	Thought: The user is requesting an image generation. I should use the text_to_image API.
	Action: text_to_image
	Action Input: {"text": "A cute black kitten with big eyes, fluffy fur, and a playful expression"}
	Observation: Here is the generated image URL: [https://www.wenshengtu.com]
	Thought: I now know the final answer.
	Final Answer: Here is an image of a cute black kitten: [https://www.wenshengtu.com]
	""", 
# "Modify this description: 'A blue Honda car parked on the street' to 'A red Mazda car parked on the street'", -- Fake resp id 5 -- Tool-use: modify_text
	"""
	Thought: The user wants to modify a text description. They want change from `A blue honda car parked on the street` to `A red Mazda car parked on the street`.
	Final Answer: "A red Mazda car parked on the street."
	""", 
# Check chữ trên sản phẩm
	"""
	Tạm Thời:
	Nhận được tín hiệu xử lý từ AI-Vision, bao gồm 1. fuzetea, Fuzetea là một nhãn hiệu trà giải khát được phân phối tại Việt nam, các thông tin chuỗi nhận diện được còn lại như passion fruit tea and chia seeds and youthful-life-every day đều đầy đủ ngữ nghĩa -- sản phẩm OK.
	"""
# "exit",
	"""Goodbye! Have a great day! 😊"""
]



DICT_FAKE_RESPONSES = {idx: fresp for idx, fresp in enumerate(FAKE_RESPONSES)}



MODEL = FakeStreamingListLLM(responses=[""])



EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. 
What is the {name_for_human} API useful for? 
{description_for_model}.
Parameters: {parameters}"""



PROMPT_REACT = """You are an AI assistant that follows the ReAct reasoning framework. 
You have access to the following APIs:

{tools_desc}

Use the following strict format:

### Input Format:

Question: [The input question]
Thought: [Think logically about the next step]
Action: [Select from available tools: {tools_name}]
Action Input: [Provide the required input]
Observation: [Record the output from the action]
... (Repeat the Thought/Action/Observation loop as needed)
Thought: I now know the final answer
Final Answer: [Provide the final answer]

Begin!

Question: {query}"""



STOP_WORDS = ["Observation:", "Observation:\n"]



class ResponseWithChainOfThought(BaseModel):
	"""LLM xuất output định dạng ReAct khi gặp truy vấn cần CoT."""
	question: str
	thought: str
	action: str
	action_input: str
	observation: str
	final_thought: str
	final_answer: str



# Hàm đầu vào chính của đoạn mã ví dụ này.
#
# Input:
# prompt: query mới nhất từ ​​người dùng.
#   history: Lịch sử hội thoại giữa người dùng và mô hình, dưới dạng một list.
#       mỗi phần tử trong danh sách có dạng:
#           {"user": "query của user", "agent": "respond của agent"}.
#       hội thoại mới nhất sẽ nằm ở cuối danh sách. Không bao gồm câu hỏi mới nhất. 
#   tools: Danh sách các tools có thể sử dụng, được lưu trong một list.
#       ví dụ tools = [tool_info_0, tool_info_1, tool_info_2]，
#       trong đó tool_info_0, tool_info_1, tool_info_2 là thông tin chi tiết của 
#           từng plugin, đã được đề cập trước đó trong tài liệu này.
#
# output:
#   phản hồi của agent cho query của người dùng. 



def llm_with_tools(query, history, tools, idx):
	# TODO: Lịch sử hội thoại càng to, thời gian thực thi vòng lặp for càng lớn. Làm sao để giải quyết?
	chat_history = [(x["user"], x["bot"]) for x in history] + [(query, "")]
	# Ngữ cảnh trò chuyện để mô hình tiếp tục nội dung
	planning_prompt = build_input_text(chat_history=chat_history, tools=tools)
	text = ""
	while True:
		resp = model_invoke(input_text=planning_prompt+text, idx=idx)
		action, action_input, output = parse_latest_tool_call(resp=resp) 
		if action: # Cần phải gọi tools 
			# action và action_input lần lượt là tool cần gọi và tham số đầu vào
			# observation là kết quả trả về từ tool, dưới dạng chuỗi
			res = tool_exe(
				tool_name=action, tool_args=action_input, idx=idx
			)
			text += res
			break
		else:  # Quá trình sinh nội dung kết thúc và không cần gọi tool 
			text += output
			break
	new_history = []
	new_history.extend(history)
	new_history.append(
		{'user': query, 'bot': text}
	)
	idx += 1
	return text, new_history



def build_input_text(
		chat_history, 
		tools, 
		im_start = "<|im_start|>", 
		im_end = "<|im_end|>"
	):
	"""Tổng hợp lịch sử hội thoại và thông tin plugin thành một văn bản đầu vào (context history)."""
	prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
	tools_text = []
	for tool_info in tools:
		tool = TOOL_DESC.format(
			name_for_model=tool_info["name_for_model"],
			name_for_human=tool_info["name_for_human"],
			description_for_model=tool_info["description_for_model"],
			parameters=json.dumps(tool_info["parameters"], ensure_ascii=False)
		)
		if dict(tool_info).get("args_format", "json") == "json":
			tool += ". Format the arguments as a JSON object."
		elif dict(tool_info).get("args_format", "code") == "code":
			tool += ". Enclose the code within triple backticks (`) at the beginning and end of the code."
		else:
			raise NotImplementedError
		tools_text.append(tool)
	tools_desc = "\n\n".join(tools_text)
	tools_name = ", ".join([tool_info["name_for_model"] for tool_info in tools])
	for i, (query, response) in enumerate(chat_history):
		# TODO: Nghiên cứu thêm RAG-tools-call xử lý gọi tool 
		if tools:  # Nếu có gọi tool 
			# Quyết định điền thông tin chi tiết của tool vào cuối hội thoại hoặc trước cuối hội thoại.
			# TODO: Cần làm rõ dòng lệnh if -- tại line 244
			if (len(chat_history) == 1) or (i == len(chat_history) - 2):
				query = PROMPT_REACT.format(
					tools_desc=tools_desc, 
					tools_name=tools_name, 
					query=query
				)
		query = query.strip() # Quan trọng! Nếu không áp dụng strip, cấu trúc dữ liệu sẽ khác so với cách được xây dựng trong quá trình huấn luyện.
		if isinstance(response, str):
			response = response.strip()
		elif not response:
			raise ValueError(">>> Error: response is None or empty, expected a string.")  
		else:
			try:
				response = str(response).strip() # Quan trọng! Nếu không áp dụng strip, cấu trúc dữ liệu sẽ khác so với cách được xây dựng trong quá trình huấn luyện.
			except Exception as e:
				raise e
		# Trong text_completion, sử dụng định dạng sau để phân biệt giữa User và AI 
		prompt += f"\n{im_start}user\n{query}{im_end}"
		prompt += f"\n{im_start}assistant\n{response}{im_end}"
	assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
	prompt = prompt[: -len(f"{im_end}")]
	return prompt



def model_invoke(input_text, idx):
	"""Text completion, sau đó chỉnh sửa kết quả inference output."""
	res = MODEL.invoke(input=input_text)
	res = llm_fake_response(idx=idx)
	return res 



def llm_fake_response(idx):
	"""Giả lập kết quả inference LLM."""
	return DICT_FAKE_RESPONSES[idx]



def parse_latest_tool_call(resp):
	"""Xử lý kết quả inference LLM, phân tích chuỗi để thực thi công cụ."""
	tool_name, tool_args = "", ""
	action = str(resp).rfind("Action:")
	action_input = str(resp).rfind("Action Input:")
	observation = str(resp).rfind("Observation:")
	if 0 <= action < action_input < observation:
		tool_name = str(resp[action + len("Action:") : action_input]).strip()
		tool_args = str(resp[action_input + len("Action Input:") : observation]).strip()
		resp = resp[:observation]
	return tool_name, tool_args, resp



def text_to_image(tool_args):
	import urllib.parse
	prompt = json5.loads(tool_args)["text"]
	prompt = urllib.parse.quote(prompt)
	return json.dumps({"image_url": f"https://image.pollinations.ai/prompt/{prompt}"}, ensure_ascii=False)



def image_to_text(tool_args, idx, img_save_path="./"):
	# Giả lập thực thi công cụ image_to_text
	if "": 
		img = request_image_from_web(
			tool_args=tool_args, 
			img_save_path=img_save_path
		)
	resp = llm_fake_response(idx=idx)
	return resp[resp.rfind("Final Answer") :]



def llm_vision(tool_args, idx):
	import numpy as np 
	import cv2 
	from paddleocr import PaddleOCR, draw_ocr
	from PIL import Image 
	vid_path = "G:/tranvantuan/fuzetea_vid2.mp4"
	ocr = PaddleOCR(use_angle_cls=True, lang='en') 
	cap = cv2.VideoCapture(vid_path)
	if not cap.isOpened():
		print(">>> Can not open camera")
		exit()
	print(">>> Starting real-time OCR. Press 'q' to exit.")
	while True:
		ret, frame = cap.read()
		if not ret:
			print(">>> Can't receive frame (stream end?). Exiting...")
			break 
		# Perform OCR on the current frame
		result = ocr.ocr(frame, cls=False)
		# Draw detected text on the frame
		for res in result:
			for line in res:
				box, (text, score) = line 
				box = np.array(box, dtype=np.int32)
				# Draw bounding box
				cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
				# Display text near the bounding box
				x, y = box[0]
				cv2.putText(frame, f"{text} ({score:.2f})", (x, y - 10), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		cv2.imshow("Research Demo AI-Agent create AI-Vision", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break 

	cap.release()
	cv2.destroyAllWindows()
	print(">>> OCR session ended.")
# Input:
#   tool_name: Tool được gọi, tương ứng với name_for_model.
#   tool_args：Tham số đầu vào của tool, là một dict. key và value của dict lần lượt là tên tham số và giá trị tham số
# Output:
#   Kết quả trả về của tool là dạng chuỗi.
#   Khi đầu ra ban đầu là JSON, sử dụng json.dumps(..., ensure_ascii=False) để chuyển đổi thành chuỗi.



def tool_exe(tool_name: str, tool_args: str, idx: int, video_path: str) -> str:
	"""Thực thi công cụ (tool execution) được LLM gọi."""
	if tool_name == "image_to_text":
		resp = image_to_text(
			tool_args=tool_args, idx=idx
		)
		return resp
	elif tool_name == "text_to_image":
		resp = text_to_image(tool_args=tool_args)
	elif tool_name == "llm_vision":
		resp = llm_vision(
			tool_args=tool_args, 
			idx=idx, 
		)
		return resp
	else:
		raise NotImplementedError



def request_image_from_web(tool_args, img_save_path="./"):
	try:
		img_path = json5.loads(tool_args)["image_path"]
		if str(img_path).startswith("http"):
			headers = {
				"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
				"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
				"Accept-Language": "en-US,en;q=0.5",
				"Accept-Encoding": "gzip, deflate, br",
				"Connection": "keep-alive",
				"Upgrade-Insecure-Requests": "1"
			}
			yzmdata = requests.get(url=img_path, headers=headers)
			tmp_img = BytesIO(yzmdata.content)
			img = Image.open(tmp_img).convert('RGB')
			img.save(img_save_path)
			img = Image.open(img_save_path).convert('RGB')
			return img
		else:
			img = Image.open(img_path).convert('RGB')
			return img 
	except:
		img_path = input(">>> Vui lòng nhập địa chỉ hình ảnh hoặc URL: ")
		if img_path.startswith('http'):
			headers = {
				"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
				"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
				"Accept-Language": "en-US,en;q=0.5",
				"Accept-Encoding": "gzip, deflate, br",
				"Connection": "keep-alive",
				"Upgrade-Insecure-Requests": "1"
			}
			yzmdata = requests.get(img_path,headers=headers)
			tmp_img = BytesIO(yzmdata.content)
			img = Image.open(tmp_img).convert('RGB')
			img.save(img_save_path)
			img = Image.open(img_save_path).convert('RGB')
			return img
		else:
			img = Image.open(img_path).convert('RGB')
			return img 



def main():
	history = []
	for idx, query in tqdm.tqdm(enumerate([
		"Hello, Good afternoon!", # -- id 0 
		"Who is Jay Chou?", # -- id 1
		"Who is his wife?", # -- id 2
		"Describe what is in this image, this is URL of the image: https://www.night_city_img.com", # -- id 3
		"Draw me a cute kitten, preferably a black cat", # -- id 4
		"Modify this description: 'A blue Honda car parked on the street' to 'A red Mazda car parked on the street'", # --id 5
		"I need to verify the characters on this product. Here is the image path of the product: G:/tranvantuan/fuzetea.jpg", # --id 6
		"exit" 
	])):
		print("\n")
		print(f">>> 🧑 query:\n\t{query}\n")
		if query.lower() == "exit":
			print(f">>> 🤖 response:\nGoodbye! Have a great day! 😊\n")
			break 
		response, history = llm_with_tools(
			query=query, 
			history=history, 
			tools=TOOLS, 
			idx=idx
		)
		print(f">>> 🤖 response:\n{response}\n")



if __name__ == "__main__":
	fire.Fire(main) 



# TODO: 
# Kịch bản Demo -- Tôi cần bạn giúp tôi tracking chữ trên vật thể này, 
# với các yêu cầu sau: 
# 1. Chữ có đủ không. 
# 2. Có bị sai nghĩa không


if False:
		
	import numpy as np 
	import cv2 
	from paddleocr import PaddleOCR, draw_ocr
	from PIL import Image


	vid_path = "G:/tranvantuan/fuzetea_vid2.mp4"



	ocr = PaddleOCR(use_angle_cls=True, lang='en') 



	cap = cv2.VideoCapture(vid_path)
	if not cap.isOpened():
		print(">>> Can not open camera")
		exit()
	print(">>> Starting real-time OCR. Press 'q' to exit.")
	while True:
		ret, frame = cap.read()
		if not ret:
			print(">>> Can't receive frame (stream end?). Exiting...")
			break 
		# Perform OCR on the current frame
		result = ocr.ocr(frame, cls=False)
		# Draw detected text on the frame
		for res in result:
			for line in res:
				box, (text, score) = line 
				box = np.array(box, dtype=np.int32)
				# Draw bounding box
				cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
				# Display text near the bounding box
				x, y = box[0]
				cv2.putText(frame, f"{text} ({score:.2f})", (x, y - 10), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		cv2.imshow("Research Demo AI-Agent create AI-Vision", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break 

	cap.release()
	cv2.destroyAllWindows()
	print(">>> OCR session ended.")

	# img_path = 'G:/tranvantuan/fuzetea.jpg'
	# result = ocr.ocr(image, cls=True)
	# for idx in range(len(result)):
	#     res = result[idx]
	#     for line in res:
	#         print(line)



	# # draw result
	# result = result[0]
	# image = Image.open(img_path).convert('RGB')
	# boxes = [line[0] for line in result]
	# txts = [line[1][0] for line in result]
	# scores = [line[1][1] for line in result]
	# im_show = draw_ocr(image, boxes, txts, scores)
	# im_show = Image.fromarray(im_show)
	# im_show.save('result.jpg')
