import uuid
import random
import tqdm
import requests
import json
import json5
import fire 
from PIL import Image
from io import BytesIO



from pydantic import BaseModel, Field
from typing_extensions import (Annotated, TypedDict, Sequence, Union, Optional, Literal, List, Dict)



from langchain_core.tools import InjectedToolCallId, BaseTool
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import (HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage)
from langchain_core.prompts import PromptTemplate



from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.managed import IsLastStep
from langgraph.graph import (MessagesState, StateGraph, START, END)
from langgraph.prebuilt import (create_react_agent, ToolNode, tools_condition)



from prompts import (TOOL_DESC_PROMPT, REACT_PROMPT)
from tools import (add, subtract, multiply, divide, power, square_root)



begin_of_text = "<|begin_of_text|>"
end_of_text = "<|end_of_text|>"
start_header_id = "<|start_header_id|>"
end_header_id = "<|end_header_id|>"
end_of_message_id = "<|eom_id|>"
end_of_turn_id = "<|eot_id|>"



TOOLS = [add, subtract, multiply, divide, power, square_root]



MESSAGE_TYPES = {SystemMessage: "SYSTEM", HumanMessage: "HUMAN", AIMessage: "AI"}



def add_unique_messages(
		messages: Dict[str, List[BaseMessage]], 
		message: BaseMessage
	) -> None:
	"""Custom reducer, adds a message to the appropriate category if it does not already exist."""

	category = MESSAGE_TYPES.get(type(message))
	if category and all(message.content != msg.content for msg in messages[category]):
		messages[category].append(message)



class State(BaseModel):
	"""Represents the structured conversation state with categorized messages."""
	user_query: Annotated[List, add_messages] = Field(default_factory=list)
	messages: Dict[str, Annotated[List[BaseMessage], add_unique_messages]] = Field(
		default_factory=lambda: {
			"SYSTEM": [], 
			"HUMAN": [], 
			"AI": []
		}, description="Categorized messages: SYSTEM, HUMAN, AI."
	)
	is_last_step: bool = False
	remaining_steps: int = 3

	def get_all_messages(self) -> List[BaseMessage]:
		"""Returns all messages in chronological order."""
		return sum(self.messages.values(), [])

	def get_latest_message(self, category: str) -> BaseMessage:
		"""Returns the latest message from a given category, if available."""
		if category not in self.messages:
			raise ValueError(f"[ERROR]: Invalid category '{category}'. Must be 'SYSTEM', 'HUMAN', or 'AI'.")
		return self.messages[category][-1] if self.messages[category] else None 

	def get_messages_by_category(self, category: str) -> List[BaseMessage]:
		"""Returns all messages from a specific category."""
		if category not in self.messages:
			raise ValueError(f"[ERROR]: Invalid category '{category}'. Must be 'SYSTEM', 'HUMAN', or 'AI'.")
		return self.messages[category]



class React(TypedDict):
	"""ReAct structured response format."""
	final_answer: str



def build_system_prompt(tool_desc_prompt: str, react_prompt: str, tools: List[BaseTool]) -> SystemMessage:
	"""Builds a formatted system prompt with tool descriptions.

	Args:
		tool_desc_prompt (str): Template for tool descriptions.
		react_prompt (str): Template for constructing the final system prompt.
		tools (list): List of tool objects.

	Returns:
		str: A fully formatted system prompt with tool descriptions.
	"""
	list_tool_desc = [
		tool_desc_prompt.format(
			name_for_model=(tool_info := getattr(tool.args_schema, "model_json_schema", lambda: {})()).get("title", "Unknown Tool"),
			name_for_human=tool_info.get("title", "Unknown Tool"),
			description_for_model=tool_info.get("description", "No description available."),
			type=tool_info.get("type", "N/A"),
			properties=json.dumps(tool_info.get("properties", {}), ensure_ascii=False),
			required=json.dumps(tool_info.get("required", []), ensure_ascii=False),
		) + " Format the arguments as a JSON object."
		for tool in tools
	]
	return SystemMessage(react_prompt.format(
		begin_of_text=begin_of_text, 
		start_header_id=start_header_id, 
		end_header_id=end_header_id, 
		end_of_turn_id=end_of_turn_id, 
		tools_desc="\n\n".join(list_tool_desc), 
		tools_name=", ".join(tool.name for tool in tools),
	))



def enhance_human_message(state: State) -> HumanMessage:
	"""Enhances the human query by formatting it with system and assistant structure.

	This function constructs a structured prompt including:
	- `system_prompt (from context)
	- `user_query (latest human query)
	- `ai_resp (space for AI response)

	Args:
		state (State): The current conversation state.

	Returns:
		HumanMessage: A formatted human query wrapped with special tokens.

	Example:
		>>> state.user_query = [HumanMessage(content="What is AI?")]
		>>> enhance_human_query(state)
		HumanMessage(content='<|start_header_id|>user<|end_header_id|>What is AI?<|end_of_turn_id|><|start_header_id|>assistant<|end_header_id|>')
	"""
	user_query = state.user_query[-1].content if state.user_query else ""
	formatted_query = (
		f"{start_header_id}user{end_header_id}"
		f"{user_query}"
		f"{end_of_turn_id}"
		f"{start_header_id}assistant{end_header_id}"
	)
	return HumanMessage(content=formatted_query)



def add_eot_id_to_ai_message(ai_message: AIMessage, special_token: str = end_of_turn_id) -> AIMessage:
	"""Appends a special token at the end of an AIMessage's content if it's not already present.

	This function ensures that the AI-generated message always ends with the specified special token.
	If the message already contains the token at the end, it remains unchanged.

	Args:
		ai_message (AIMessage): The AI-generated message.
		special_token (str, optional): The special token to append (default is `end_of_turn_id`).

	Returns:
		AIMessage: The updated AI message with the special token appended if necessary.

	Example:
		>>> message = AIMessage(content="Hello, how can I assist you?")
		>>> add_eot_id_to_ai_message(message, special_token="<|eot_id|>")
		AIMessage(content="Hello, how can I assist you?<|eot_id|>")
	"""
	if not ai_message.content.strip().endswith(special_token):
		return AIMessage(content=ai_message.content.strip() + special_token)
	return ai_message 



CONFIG = {"configurable": {"thread_id": str(uuid.uuid4())}}
CHECKPOINTER = MemorySaver()
STORE = InMemoryStore()
MODEL = ChatOllama(model="llama3.2:1b-instruct-fp16", temperature=0.8, num_predict=128_000)
MODEL_BIND_TOOLS = MODEL.bind_tools(tools=TOOLS)
REACT_SYSTEM_MESSAGE_PROMPT = build_system_prompt(tool_desc_prompt=TOOL_DESC_PROMPT, react_prompt=REACT_PROMPT, tools=TOOLS)



def chatbot_react(state: State) -> State:
	"""Processes a user query and updates the conversation state with the chatbot's response.

	This function:
	- Enhances the user query.
	- Calls the AI model with `SYSTEM_PROMPT` and the enhanced query.
	- Ensures the AI response is formatted correctly.
	- Appends the special token `<|eot_id|>` if missing.
	- Prevents duplicate messages in `state.messages`.

	Args:
		state (State): The current conversation state, containing message history.

	Example:
		>>> state = State(messages={"SYSTEM": [], "HUMAN": [], "AI": []})
		>>> chatbot(state)
		{'messages': 
			{
				'HUMAN': HumanMessage(content='Formatted user query'), 
				'AI': AIMessage(content='Chatbot response<|eot_id|>')
			}
		}
	"""
	human_message = enhance_human_message(state=state)
	resp = MODEL.invoke([REACT_SYSTEM_MESSAGE_PROMPT, human_message])

	if not isinstance(resp, AIMessage):
		resp = AIMessage(
			content=resp.strip() 
			if isinstance(resp, str) 
			else "I'm unable to generate a response."
		)
	resp = add_eot_id_to_ai_message(ai_message=resp, special_token=end_of_turn_id)
	return {"messages": {
		"SYSTEM": [REACT_SYSTEM_MESSAGE_PROMPT], 
		"HUMAN": [human_message], 
		"AI": [resp]
	}}



workflow = StateGraph(State)

workflow.add_node("chatbot_react", chatbot_react)

workflow.add_edge(START, "chatbot_react")
workflow.add_edge("chatbot_react", END)

app = workflow.compile(checkpointer=CHECKPOINTER, store=STORE)



def main() -> None:
	"""Hàm chính để nhận truy vấn từ người dùng và hiển thị phản hồi."""
	while True: 
		user_query = input("👨_query: ")
		if user_query.lower() == "exit":
			print(">>> SystemExit: Goodbye! Have a great day!😊")
			break
		print_stream(
			app.stream(input={"user_query": [user_query]}, 
			stream_mode="values", config=CONFIG)
	)



def print_stream(stream) -> None:
	"""Hiển thị kết quả hội thoại theo cách dễ đọc hơn."""
	for s in stream:
		if len(list(s.keys())) == 2:
			messages = s["messages"]
			if "SYSTEM" in messages and messages["SYSTEM"]:
				print("\n ⚙️ **SYSTEM**:")
				messages["SYSTEM"][-1].pretty_print()

			if "HUMAN" in messages and messages["HUMAN"]:
				print("\n 👨 **HUMAN**:")
				messages["HUMAN"][-1].pretty_print()

			if "AI" in messages and messages["AI"]:
				print("\n 🤖 **AI**:")
				messages["AI"][-1].pretty_print()
			print("🔹" * 30)



if __name__ == "__main__":
	fire.Fire(main)



# def llm_with_tools(query, history, tools, idx):
# 	chat_history = [(x["user"], x["bot"]) for x in history] + [(query, "")]
# 	planning_prompt = build_input_text(chat_history=chat_history, tools=tools)
# 	text = ""
# 	while True:
# 		resp = model_invoke(input_text=planning_prompt+text, idx=idx)
# 		action, action_input, output = parse_latest_tool_call(resp=resp) 
# 		if action:
# 			res = tool_exe(
# 				tool_name=action, tool_args=action_input, idx=idx
# 			)
# 			text += res
# 			break
# 		else:  
# 			text += output
# 			break
# 	new_history = []
# 	new_history.extend(history)
# 	new_history.append(
# 		{'user': query, 'bot': text}
# 	)
# 	idx += 1
# 	return text, new_history



# def build_input_text_archive(
# 		chat_history, 
# 		tools, 
# 		im_start = "<|im_start|>", 
# 		im_end = "<|im_end|>"
# 	):
# 	"""Tổng hợp lịch sử hội thoại và thông tin plugin thành một văn bản đầu vào (context history)."""
# 	prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
# 	tools_text = []
# 	for tool_info in tools:
# 		tool = tool_desc.format(
# 			name_for_model=tool_info["name_for_model"],
# 			name_for_human=tool_info["name_for_human"],
# 			description_for_model=tool_info["description_for_model"],
# 			parameters=json.dumps(tool_info["parameters"], ensure_ascii=False)
# 		)
# 		if dict(tool_info).get("args_format", "json") == "json":
# 			tool += ". Format the arguments as a JSON object."
# 		elif dict(tool_info).get("args_format", "code") == "code":
# 			tool += ". Enclose the code within triple backticks (`) at the beginning and end of the code."
# 		else:
# 			raise NotImplementedError
# 		tools_text.append(tool)
# 	tools_desc = "\n\n".join(tools_text)
# 	tools_name = ", ".join([tool_info["name_for_model"] for tool_info in tools])
# 	for i, (query, response) in enumerate(chat_history):
# 		if tools:  # Nếu có gọi tool 
# 			# Quyết định điền thông tin chi tiết của tool vào cuối hội thoại hoặc trước cuối hội thoại.
# 			if (len(chat_history) == 1) or (i == len(chat_history) - 2):
# 				query = react_prompt.format(
# 					tools_desc=tools_desc, 
# 					tools_name=tools_name, 
# 					query=query
# 				)
# 		query = query.strip() # Quan trọng! Nếu không áp dụng strip, cấu trúc dữ liệu sẽ khác so với cách được xây dựng trong quá trình huấn luyện.
# 		if isinstance(response, str):
# 			response = response.strip()
# 		elif not response:
# 			raise ValueError(">>> Error: response is None or empty, expected a string.")  
# 		else:
# 			try:
# 				response = str(response).strip() # Quan trọng! Nếu không áp dụng strip, cấu trúc dữ liệu sẽ khác so với cách được xây dựng trong quá trình huấn luyện.
# 			except Exception as e:
# 				raise e
# 		# Trong text_completion, sử dụng định dạng sau để phân biệt giữa User và AI 
# 		prompt += f"\n{im_start}user\n{query}{im_end}"
# 		prompt += f"\n{im_start}assistant\n{response}{im_end}"
# 	assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
# 	prompt = prompt[: -len(f"{im_end}")]
# 	return prompt



# def llm_with_tools(query, history, tools, idx):
# 	chat_history = [(x["user"], x["bot"]) for x in history] + [(query, "")]
# 	planning_prompt = build_input_text(chat_history=chat_history, tools=tools)
# 	text = ""
# 	while True:
# 		resp = model_invoke(input_text=planning_prompt+text, idx=idx)
# 		action, action_input, output = parse_latest_tool_call(resp=resp) 
# 		if action:
# 			res = tool_exe(
# 				tool_name=action, tool_args=action_input, idx=idx
# 			)
# 			text += res
# 			break
# 		else:  
# 			text += output
# 			break
# 	new_history = []
# 	new_history.extend(history)
# 	new_history.append(
# 		{'user': query, 'bot': text}
# 	)
# 	idx += 1
# 	return text, new_history



# def parse_latest_tool_call(resp):
# 	"""Xử lý kết quả inference LLM, phân tích chuỗi để thực thi công cụ."""
# 	tool_name, tool_args = "", ""
# 	action = str(resp).rfind("Action:")
# 	action_input = str(resp).rfind("Action Input:")
# 	observation = str(resp).rfind("Observation:")
# 	if 0 <= action < action_input < observation:
# 		tool_name = str(resp[action + len("Action:") : action_input]).strip()
# 		tool_args = str(resp[action_input + len("Action Input:") : observation]).strip()
# 		resp = resp[:observation]
# 	return tool_name, tool_args, resp



# def ai_vision(tool_args, idx):
# 	import numpy as np 
# 	import cv2 
# 	from paddleocr import PaddleOCR, draw_ocr
# 	from PIL import Image 
# 	vid_path = json5.loads(tool_args)["video_path"]
# 	ocr = PaddleOCR(use_angle_cls=False, lang='en') 
# 	cap = cv2.VideoCapture(vid_path)
# 	if not cap.isOpened():
# 		print(">>> Can not open camera")
# 		exit()
# 	print(">>> Starting real-time OCR. Press 'q' to exit.")
# 	while True:
# 		ret, frame = cap.read()
# 		if not ret:
# 			print(">>> Can't receive frame (stream end?). Exiting...")
# 			break 
# 		# Perform OCR on the current frame
# 		result = ocr.ocr(frame, cls=False)
# 		# Draw detected text on the frame
# 		for res in result:
# 			if res is not None:
# 				for line in res:
# 					box, (text, score) = line 
# 					box = np.array(box, dtype=np.int32)
# 					# Draw bounding box
# 					cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
# 					# Display text near the bounding box
# 					x, y = box[0]
# 					cv2.putText(frame, f"{text} ({score:.2f})", (x, y - 10), 
# 					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 		cv2.imshow("Research Demo LLM+Vision", frame)
# 		if cv2.waitKey(1) & 0xFF == ord('q'):
# 			break 
# 	cap.release()
# 	cv2.destroyAllWindows()
# 	print(">>> OCR session ended.")



# def tool_exe(tool_name: str, tool_args: str, idx: int) -> str:
# 	"""Thực thi công cụ (tool execution) được LLM gọi."""
# 	if tool_name == "image_to_text":
# 		resp = image_to_text(
# 			tool_args=tool_args, idx=idx
# 		)
# 		return resp
# 	elif tool_name == "text_to_image":
# 		resp = text_to_image(tool_args=tool_args)
# 		return resp
# 	elif tool_name == "ai_vision":
# 		resp = ai_vision(
# 			tool_args=tool_args, 
# 			idx=idx
# 		)
# 		return "Finish AI-Vision, released all VRAM and Windows."
# 	else:
# 		raise NotImplementedError



# def request_image_from_web(tool_args, img_save_path="./"):
# 	try:
# 		img_path = json5.loads(tool_args)["image_path"]
# 		if str(img_path).startswith("http"):
# 			headers = {
# 				"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
# 				"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
# 				"Accept-Language": "en-US,en;q=0.5",
# 				"Accept-Encoding": "gzip, deflate, br",
# 				"Connection": "keep-alive",
# 				"Upgrade-Insecure-Requests": "1"
# 			}
# 			yzmdata = requests.get(url=img_path, headers=headers)
# 			tmp_img = BytesIO(yzmdata.content)
# 			img = Image.open(tmp_img).convert('RGB')
# 			img.save(img_save_path)
# 			img = Image.open(img_save_path).convert('RGB')
# 			return img
# 		else:
# 			img = Image.open(img_path).convert('RGB')
# 			return img 
# 	except:
# 		img_path = input(">>> Vui lòng nhập địa chỉ hình ảnh hoặc URL: ")
# 		if img_path.startswith('http'):
# 			headers = {
# 				"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
# 				"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
# 				"Accept-Language": "en-US,en;q=0.5",
# 				"Accept-Encoding": "gzip, deflate, br",
# 				"Connection": "keep-alive",
# 				"Upgrade-Insecure-Requests": "1"
# 			}
# 			yzmdata = requests.get(img_path,headers=headers)
# 			tmp_img = BytesIO(yzmdata.content)
# 			img = Image.open(tmp_img).convert('RGB')
# 			img.save(img_save_path)
# 			img = Image.open(img_save_path).convert('RGB')
# 			return img
# 		else:
# 			img = Image.open(img_path).convert('RGB')
# 			return img 



# def main():
# 	history = []
# 	for idx, query in tqdm.tqdm(enumerate([
# 		"Hello, Good afternoon!", # -- id 0 
# 		"Who is Jay Chou?", # -- id 1
# 		"Who is his wife?", # -- id 2
# 		"Describe what is in this image, this is URL of the image: https://www.night_city_img.com", # -- id 3
# 		"Draw me a cute kitten, preferably a black cat", # -- id 4
# 		"Modify this description: 'A blue Honda car parked on the street' to 'A red Mazda car parked on the street'", # --id 5
# 		"I need to check the text on this product in real-time to see if it is accurate and complete. Here is the link of video product: /home/chwenjun225/projects/DeepEngine/nexus_mind/images/fuzetea_vid2.mp4", # -- id 6
# 		"exit" 
# 	])):
# 		print("\n")
# 		print(f">>> 🧑 query:\n\t{query}\n")
# 		if query.lower() == "exit":
# 			print(f">>> 🤖 response:\nGoodbye! Have a great day! 😊\n")
# 			break 
# 		response, history = llm_with_tools(
# 			query=query, 
# 			history=history, 
# 			tools=tools, 
# 			idx=idx
# 		)
# 		print(f">>> 🤖 response:\n{response}\n")



# if __name__ == "__main__":
# 	fire.Fire(main) 



# tools = [
# 	{
# 		"name_for_human": "image_to_text", 
# 		"name_for_model": "image_to_text", 
# 		"description_for_model": "image_to_text is a service that generates textual descriptions from images. By providing the URL of an image, it returns a detailed and realistic description of the image.",
# 		"parameters": [
# 			{
# 				"name": "image_path",
# 				"description": "the URL of the image to be described",
# 				"required": True,
# 				"schema": {"type": "string"},
# 			}
# 		],
# 	},
# 	{
# 		"name_for_human": "text_to_image",
# 		"name_for_model": "text_to_image",
# 		"description_for_model": "text_to_image is an AI image generation service. It takes a text description as input and returns a URL of the generated image.",
# 		"parameters": [
# 			{
# 				"name": "text",
# 				"description": "english keywords or a text prompt describing what you want in the image.",
# 				"required": True,
# 				"schema": {"type": "string"}
# 			}
# 		]
# 	},
# 	{
# 		"name_for_human": "modify_text",
# 		"name_for_model": "modify_text",
# 		"description_for_model": "modify_text changes the original prompt based on the input request to make it more suitable.",
# 		"parameters": [
# 			{
# 				"name": "describe_before",
# 				"description": "the prompt or image description before modification.",
# 				"required": True,
# 				"schema": {"type": "string"}
# 			},
# 			{
# 				"name": "modification_request",
# 				"description": "the request to modify the prompt or image description, e.g., change 'cat' to 'dog' in the text.",
# 				"required": True,
# 				"schema": {"type": "string"}
# 			}
# 		]
# 	}, 
# 	{ 
# 		"name_for_human": "ai_vision", 
# 		"name_for_model": "ai_vision", 
# 		"description_for_model": "ai_vision is a service that use ai-vision to detects and extracts characters from a product image. It processes the image and returns the recognized text to the AI agent for further analysis.",
# 		"parameters": [
# 			{
# 				"name": "video_path",
# 				"description": "The file path of a video or the IP address of a camera for real-time character detection.",
# 				"required": True,
# 				"schema": {"type": "string"}
# 			}
# 		],
# 	},
# ]



# fake_response = [
# # "Hello, Good afternoon!", -- Fake resp id 0 -- No tool
# 	"""
# 	Thought: The input is a greeting. No tools are needed.
# 	Final Answer: Hello! Good afternoon! How can I assist you today?
# 	""", 
# # "Who is Jay Chou?", -- Fake resp id 1 -- No tool 
# 	"""
# 	Thought: The user is asking for information about Jay Chou. I should retrieve general knowledge.
# 	Final Answer: Jay Chou is a Taiwanese singer, songwriter, and actor, widely known for his influence in Mandopop music. He has released numerous albums and is recognized for his unique blend of classical and contemporary music.
# 	""", 
# # "Who is his wife?", -- Fake resp id 2 -- No tool 
# 	"""
# 	Thought: The previous question was about Jay Chou. "His wife" likely refers to Jay Chou's spouse.
# 	Final Answer: Jay Chou's wife is Hannah Quinlivan, an actress and model from Taiwan.
# 	""", 
# # "Describe what is in this image, this is URL of the image: https://www.night_city_img.com", --> Fake resp id 3 -- Tool-use: image_to_text
# 	"""
# 	Thought: The user wants a description of an image. I should use the image_to_text API.
# 	Action: image_to_text
# 	Action Input: {"image_path": "https://www.tushengwen.com"}
# 	Observation: "The image depicts a vibrant cityscape at night, illuminated by neon lights and tall skyscrapers."
# 	Thought: I now know the final answer.
# 	Final Answer: The image depicts a vibrant cityscape at night, illuminated by neon lights and tall skyscrapers.
# 	""",
# # "Draw me a cute kitten, preferably a black cat", -- Fake resp id 4 -- Tool-use: text_to_image
# 	"""
# 	Thought: The user is requesting an image generation. I should use the text_to_image API.
# 	Action: text_to_image
# 	Action Input: {"text": "A cute black kitten with big eyes, fluffy fur, and a playful expression"}
# 	Observation: Here is the generated image URL: [https://www.wenshengtu.com]
# 	Thought: I now know the final answer.
# 	Final Answer: Here is an image of a cute black kitten: [https://www.wenshengtu.com]
# 	""", 
# # "Modify this description: 'A blue Honda car parked on the street' to 'A red Mazda car parked on the street'", -- Fake resp id 5 -- Tool-use: modify_text
# 	"""
# 	Thought: The user wants to modify a text description. They want change from `A blue honda car parked on the street` to `A red Mazda car parked on the street`.
# 	Final Answer: "A red Mazda car parked on the street."
# 	""", 
# # "I need to check the text on this product in real-time to see if it is accurate and complete. Here is the link of video product: /home/chwenjun225/projects/DeepEngine/nexus_mind/images/fuzetea_vid1.mp4", # -- id 6
# 	"""
# 	Thought: I need to analyze the text on the product in real-time to verify its accuracy and completeness. I should process the video file to extract frames and apply OCR. 
# 	Action: ai_vision
# 	Action Input: {"video_path": "/home/chwenjun225/projects/DeepEngine/nexus_mind/images/fuzetea_vid1.mp4"}
# 	Observation: The OCR result has been extracted from the product video. The detected text is: ["fuzetea", "Passion fruit tea and chia seeds"]
# 	Thought: I will now compare the extracted text with the expected information to check if it is accurate and complete.
# 	Final Answer: The text on the product has been successfully extracted. The accuracy and completeness can now be verified based on the expected product information.
# 	"""
# # "exit",
# 	"""Goodbye! Have a great day! 😊"""
# ]



# dict_fake_responses = {idx: fresp for idx, fresp in enumerate(fake_response)}



# STOP_WORDS = ["Observation:", "Observation:\n"]