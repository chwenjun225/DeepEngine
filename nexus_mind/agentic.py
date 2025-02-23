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



import warnings
warnings.filterwarnings("ignore")


import requests
import json
import json5
import fire 
import torch 
from PIL import Image
from io import BytesIO


from transformers import (StoppingCriteria, StoppingCriteriaList, AutoTokenizer)


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import (AIMessage, HumanMessage, ToolMessage)



# Constant vars 
MODEL = ChatOllama(model="llama3.2:1b-instruct-fp16", temperature=0.1)
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
TOKENIZER = AutoTokenizer.from_pretrained("/home/chwenjun225_laptop/.llama/checkpoints/Llama-3.2-1B-Instruct")
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""
PROMPT_REACT = """Answer the following questions as best you can. You have access to the following APIs:

{tools_desc}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""



#
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
#



def llm_with_tools(query, history, tools):
	chat_history = [(x["user"], x["agent"]) for x in history] + [(query, "")]
	# Ngữ cảnh trò chuyện để mô hình tiếp tục nội dung
	planning_prompt = build_input_text(
		chat_history=chat_history, 
		tools=tools
	)
	text = ""
	while True:
		output = text_completion(
			planning_prompt + text, 
			stop_words=["Observation:", "Observation:\n"]
		)
		action, action_input, output = parse_latest_tool_call(output)
		if action: # Cần phải gọi tools
			# action và action_input lần lượt là mã của tool cần gọi và tham số đầu vào
			# observation là kết quả trả về từ tool, dưới dạng chuỗi
			observation = call_tool(action, action_input)
			output += f"\nObservation: {observation}\nThought:"
			text += output
		else:  # Quá trình sinh nội dung kết thúc và không cần gọi plugin nữa
			text += output
			break
	new_history = []
	new_history.extend(history)
	new_history.append(
		{'user': query, 'bot': text}
	)
	return text, new_history



def build_input_text(chat_history, tools):
	"""Tổng hợp lịch sử hội thoại và thông tin plugin thành một văn bản đầu vào (context history)."""
	tools_text = []
	for tool_info in tools:
		tool = TOOL_DESC.format(
			name_for_model=tool_info["name_for_model"],
			name_for_human=tool_info["name_for_human"],
			description_for_model=tool_info["description_for_model"],
			parameters=json.dumps(tool_info["parameters"], ensure_ascii=False)
		)
		if tool_info.get("args_format", "json") == "json":
			tool += ". Format the arguments as a JSON object."
		elif tool_info["args_format"] == "code":
			tool += " Enclose the code within triple backticks (`) at the beginning and end of the code."
		else:
			raise NotImplementedError
		tools_text.append(tool)
	# tool desc
	tools_desc = "\n\n".join(tools_text)
	# tool name
	tools_name = ", ".join([tool_info["name_for_model"] for tool_info in tools])

	im_start = "<|im_start|>"
	im_end = "<|im_end|>"
	prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
	for i, (query, response) in enumerate(chat_history):
		if tools:  # Nếu có gọi tool
			# Quyết định điền thông tin chi tiết của tool vào cuối hội thoại hoặc trước cuối hội thoại.
			if (len(chat_history) == 1) or (i == len(chat_history) - 2):
				query =  PROMPT_REACT.format(
					tools_desc=tools_desc,
					tools_name=tools_name,
					query=query
				)
		query = query.lstrip("\n").rstrip() # Quan trọng! Nếu không áp dụng strip, cấu trúc dữ liệu sẽ khác so với cách được xây dựng trong quá trình huấn luyện.
		response = response.lstrip("\n").rstrip() # Quan trọng! Nếu không áp dụng strip, cấu trúc dữ liệu sẽ khác so với cách được xây dựng trong quá trình huấn luyện.
		# Khi sử dụng chế độ hoàn thành văn bản, bạn cần sử dụng định dạng sau để phân biệt giữa người dùng và AI:
		prompt += f"\n{im_start}user\n{query}{im_end}"
		prompt += f"\n{im_start}assistant\n{response}{im_end}"

	assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
	prompt = prompt[: -len(f"{im_end}")]
	return prompt



def text_completion(input_text: str, stop_words) -> str:  # Sử dụng cho task text completion
	model = MODEL
	tokenizer = TOKENIZER
	im_end = "<|im_end|>"
	if im_end not in stop_words:
		stop_words = stop_words + [im_end]
	res = model.invoke(input=input_text)
	# Xử lý kết quả trả về: nếu kết quả bao gồm cả input_text ban đầu, loại bỏ nó đi.
	if res.content.startswith(input_text):
		res = res.content[len(input_text):]
	# Loại bỏ các token đặc biệt nếu có
	res = res.content.replace("<|endoftext|>", "").replace(im_end, "")
	# Cắt kết quả nếu gặp từ dừng nào trong stop_words
	for stop_str in stop_words:
		idx = res.find(stop_str)
		if idx != -1:
			output = res[:idx + len(stop_str)]
	return output # Trả về phần tiếp nối của input_text





def parse_latest_tool_call(text):
	tool_name, tool_args = "", ""
	i = text.rfind("\nAction:")
	j = text.rfind("\nAction Input:")
	k = text.rfind("\nObservation:")
	if 0 <= i < j:  # If the text has `Action` and `Action input`,
		if k < j:  # but does not contain `Observation`,
			# then it is likely that `Observation` is skipped by the LLM,
			# because the output text may have discarded the stop word.
			text = text.rstrip() + "\nObservation:"  # Add it back.
		k = text.rfind("\nObservation:")
		tool_name = text[i + len("\nAction:") : j].strip()
		tool_args = text[j + len("\nAction Input:") : k].strip()
		text = text[:k]
	return tool_name, tool_args, text



#
# Input:
#   tool_name: Tool được gọi, tương ứng với name_for_model.
#   tool_args：Tham số đầu vào của tool, là một dict. key và value của dict lần lượt là tên tham số và giá trị tham số
# Output:
#   Kết quả trả về của tool là dạng chuỗi.
#   Khi đầu ra ban đầu là JSON, sử dụng json.dumps(..., ensure_ascii=False) để chuyển đổi thành chuỗi.
#



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
	for query in ["Hello", "Who is Jay Chou?", "Who is his wife?", "Draw me a cute kitten, preferably a black cat", "exit"]:
		if query.lower() == "exit":
			break 
		response, history = llm_with_tools(
			query=query, 
			history=history, 
			tools=tools
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
