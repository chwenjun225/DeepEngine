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



import requests
import json
import json5
import fire 
import torch 
from PIL import Image
from io import BytesIO



from pydantic import BaseModel, Field
from transformers import (StoppingCriteria, StoppingCriteriaList, AutoTokenizer)



from langchain.tools import StructuredTool 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.fake import FakeStreamingListLLM
from langchain_core.messages import (AIMessage, HumanMessage, ToolMessage)



# Constant vars 
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
	Action Input: {"image_path": "[User provided image URL]"}
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
	Thought: The user wants to modify a text description. I should use the modify_text API.
	Action: modify_text
	Action Input: {"describe_before": "A blue honda car parked on the street", "modification_request": "Change 'blue Honda' to 'red Mazda'"}
	Observation: "A red Mazda car parked on the street"
	Thought: I now know the final answer.
	Final Answer: "A red Mazda car parked on the street"
	"""
]



DICT_FAKE_RESPONSES = {idx: fresp for idx, fresp in enumerate(FAKE_RESPONSES)}



MODEL = FakeStreamingListLLM(responses=[""])



EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



TOKENIZER = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")



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



class SequenceStoppingCriteria(StoppingCriteria):
	"""Tùy chỉnh điều kiện dừng sinh chuỗi cho LLM."""
	def __init__(self, sequence_ids):
		self.sequence_ids = sequence_ids
		self.current_sequence = []
	def check_sequences(self, current_tokens, sequences):
		"""
		Kiểm tra các tokens được tạo có chứa một chuỗi ký tự lặp hay không.

		:param current_tokens: 
			Danh sách các tokens hiện đang được tạo.
		:param sequences: 
			Một danh sách chứa nhiều chuỗi ký tự lặp.
		:return: 
			Trả về True nếu chuỗi ký tự lặp nào xuất hiện trong current_token, nếu không thì trả về False.
		"""
		for i in range(len(current_tokens) - max(map(len, sequences)) + 1):
			for seq in sequences:
				if current_tokens[i:i+len(seq)] == seq:
					return True
		return False
	def __call__(self, input_ids, scores, **kwargs):
		# Nhận các tokens hiện tại đang được tạo.
		current_tokens = [input_ids[-1][-1]]
		# Kiểm tra các tokens liên tiếp có khớp với chuỗi dừng không
		self.current_sequence.extend(current_tokens)
		# Kiểm tra xem các mã thông báo hiện được tạo có chứa một chuỗi số liên tiếp cụ thể hay không
		if self.check_sequences(self.current_sequence, self.sequence_ids):
			return True  # Dừng tạo
		return False



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
	chat_history = [(x["user"], x["bot"]) for x in history] + [(query, "")]
	# Ngữ cảnh trò chuyện để mô hình tiếp tục nội dung
	planning_prompt = build_input_text(chat_history=chat_history, tools=tools)
	text = ""
	while True:
		resp = model_invoke(input_text=planning_prompt+text, idx=idx)
		action, action_input, output = parse_latest_tool_call(response=resp) 
		if action: # Cần phải gọi tools 
			# action và action_input lần lượt là tool cần gọi và tham số đầu vào
			# observation là kết quả trả về từ tool, dưới dạng chuỗi
			observation = call_tool(action, action_input)
			output += f"\nObservation: {observation}\nThought:"
			text += output
		else:  # Quá trình sinh nội dung kết thúc và không cần gọi tool nữa 
			text += output
			break
	new_history = []
	new_history.extend(history)
	new_history.append(
		{'user': query, 'bot': text}
	)
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
		# Khi sử dụng chế độ text_completion, bạn cần sử dụng định dạng sau để phân biệt giữa người dùng và AI 
		prompt += f"\n{im_start}user\n{query}{im_end}"
		prompt += f"\n{im_start}assistant\n{response}{im_end}"
	assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
	prompt = prompt[: -len(f"{im_end}")]
	return prompt



def model_invoke(input_text, idx):
	# TODO: Giả lập, mô phỏng response từ LLM
	"""Text completion sau đó chỉnh sửa kết quả inference output."""
	res = MODEL.invoke(input=input_text)
	res = DICT_FAKE_RESPONSES[idx]
	return res 



def llm_fake_response():
	fake_responses = {}



def parse_latest_tool_call(response):
	tool_name, tool_args = "", ""
	i = str(response).rfind("\nAction:")
	j = str(response).rfind("\nAction Input:")
	k = str(response).rfind("\nObservation:")
	if 0 <= i < j < k:
		tool_name = str(response[i + len("\nAction:") : j]).strip()
		tool_args = str(response[j + len("\nAction Input:") : k]).strip()
		response = response[:k]
	return tool_name, tool_args, response



# Input:
#   tool_name: Tool được gọi, tương ứng với name_for_model.
#   tool_args：Tham số đầu vào của tool, là một dict. key và value của dict lần lượt là tên tham số và giá trị tham số
# Output:
#   Kết quả trả về của tool là dạng chuỗi.
#   Khi đầu ra ban đầu là JSON, sử dụng json.dumps(..., ensure_ascii=False) để chuyển đổi thành chuỗi.



def call_tool(tool_name: str, tool_args: str) -> str:
	img_save_path = "./"
	tokenizer = TOKENIZER
	model = MODEL
	if tool_name == "image_gen_prompt":
		try:
			img_path = json5.loads(tool_args)["image_path"]
			if img_path.startswith("http"):
				headers = {
					"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
					"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
					"Accept-Language": "en-US,en;q=0.5",
					"Accept-Encoding": "gzip, deflate, br",
					"Connection": "keep-alive",
					"Upgrade-Insecure-Requests": "1"
				}
				yzmdata = requests.get(img_path, headers=headers)
				tmp_img = BytesIO(yzmdata.content)
				img = Image.open(tmp_img).convert('RGB')
				img.save(img_save_path)
				img = Image.open(img_save_path).convert('RGB')
			else:
				img = Image.open(img_path).convert('RGB')
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
			else:
				img = Image.open(img_path).convert('RGB')
		question = "Please describe all the details in this picture in detail?"
		msgs = [{"role": "user", "content": question}]
		res = model.chat(image=img, msgs=msgs, tokenizer=tokenizer)
		return res
	elif tool_name == "image_gen":
		import urllib.parse
		prompt = json5.loads(tool_args)["prompt"]
		prompt = urllib.parse.quote(prompt)
		return json.dumps({"image_url": f"https://image.pollinations.ai/prompt/{prompt}"}, ensure_ascii=False)
	elif tool_name == "modify_text":
		import urllib.parse
		prompt_input = json5.loads(tool_args)["describe_before"]
		modification_request = json5.loads(tool_args)["modification_request"]
		input_prompt = "Please modify the prompt: {}. According to the following requirements:{}. The modified prompt is: ".format(prompt_input, modification_request)
		im_start = "<|im_start|>"
		im_end = "<|im_end|>"
		prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"+f"\n{im_start}user\n{input_prompt}{im_end}"
		input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
		output = model.llm.generate(input_ids, max_length=4096)
		output = output.tolist()[0]
		output = tokenizer.decode(output, errors="ignore")
		return output
	else:
		raise NotImplementedError



def token_counter(messages):
	"""Đếm số lượng token từ danh sách tin nhắn."""
	tokenizer = TOKENIZER
	text = " ".join([msg.content for msg in messages])
	return len(tokenizer.encode(text)) 



def main():
	tools = [
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
		}
	]
	history = []
	for idx, query in enumerate([
		"Hello, Good afternoon!", # -- id 0 
		"Who is Jay Chou?", # -- id 1
		"Who is his wife?", # -- id 2
		"Describe what is in this image, this is URL of the image: https://www.night_city_img.com", # -- id 3
		"Draw me a cute kitten, preferably a black cat", # -- id 4
		"Modify this description: 'A blue Honda car parked on the street' to 'A red Mazda car parked on the street'", # --id 5
		"exit"
	]):
		print(f">>> 🧑 query: \n{query}\n")
		if query.lower() == "exit":
			print(f">>> 🤖 response:\nGoodbye! Have a great day! 😊\n")
			break 
		response, history = llm_with_tools(
			query=query, 
			history=history, 
			tools=tools, 
			idx=idx
		)
		print(f">>> 🤖 response:\n{response}\n")



if __name__ == "__main__":
	fire.Fire(main) 










































































# def select_tools(state: State) -> State:
# 	query = state["messages"][-1].content
# 	tool_docs = tools_retriever.invoke(query)
# 	return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}

# def reflect(state: State) -> State:
# 	class_map = {
# 		AIMessage: HumanMessage, 
# 		HumanMessage: AIMessage, 
# 		ToolMessage: HumanMessage 
# 	}
# 	translated = [reflection_prompt, state["messages"][0]] + [
# 		class_map[msg.__class__](content=msg.content) 
# 		for msg in state["messages"][1:]
# 	]
# 	answer = model.invoke(translated)
# 	return {"messages": [HumanMessage(content=answer.content)]}

# def should_continue(state: State):
# 	if len(state["messages"]) > 6:
# 		return END
# 	else:
# 		return "reflect"

# def chatbot(state: State) -> State:
# 	selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
# 	answer = model.bind_tools(selected_tools).invoke([generate_prompt] + state["messages"])
# 	return {"messages": [answer]}

# def main():
# 	"""Thực thi chương trình."""
# 	builder = StateGraph(State)

# 	builder.add_node("select_tools", select_tools)
# 	builder.add_node("chatbot", chatbot)
# 	builder.add_node("tools", ToolNode(tools))
# 	builder.add_node("reflect", reflect)

# 	builder.add_edge(START, "select_tools")
# 	builder.add_edge("select_tools", "chatbot")
# 	builder.add_conditional_edges("chatbot", tools_condition)
# 	builder.add_edge("tools", "chatbot")
# 	builder.add_conditional_edges("chatbot", should_continue)
# 	builder.add_edge("reflect", "chatbot")
	
# 	graph = builder.compile(checkpointer=MemorySaver())

# 	user_input = {
# 		"messages": [HumanMessage("""What is Large Language Model?""")]
# 	}
# 	for chunk in graph.stream(user_input, config):
# 		print(chunk)

# if __name__ == "__main__":
# 	fire.Fire(main)
