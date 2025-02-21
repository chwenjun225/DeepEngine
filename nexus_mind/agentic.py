# Tài liệu liên quan:  
# 	Giới thiệu ngắn gọn về nguyên lý ReAct Prompting, không bao gồm mã nguồn:  
#		https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_prompt.md  
#   Triển khai ReAct Prompting dựa trên giao diện `model.chat` 
# 			(chế độ đối thoại), bao gồm cả tích hợp công cụ với LangChain:  
#		https://github.com/QwenLM/Qwen-7B/blob/main/examples/langchain_tooluse.ipynb  
#   Triển khai ReAct Prompting dựa trên giao diện `model.generate` 
# 			(chế độ tiếp tục viết), phức tạp hơn so với chế độ chat:  
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_demo.py (tệp này)  

import warnings
warnings.filterwarnings("ignore")


import json 
from uuid import uuid4 
from pprint import pprint
import fire 
from datetime import datetime
from transformers import AutoTokenizer, StoppingCriteria
from typing_extensions import Literal


from langchain_openai import ChatOpenAI
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_core.messages import ToolCall, AIMessage, HumanMessage, ToolMessage, trim_messages # TODO: Tính năng lọc và cắt tin nhắn


from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END


from state import SupervisorDecision, State, Input, Output
from prompts import generate_prompt, reflection_prompt, system_prompt_part_1, system_prompt_part_2
from tools_use import DuckDuckGoSearchRun, calculator



# Cấu hình các biến hằng số aaaa
TOKENIZER = AutoTokenizer.from_pretrained("/home/chwenjun225/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B")
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
MODEL = ChatOpenAI(model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct", openai_api_base="http://127.0.0.1:2026/v1", openai_api_key="chwenjun225", temperature=0.1)
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""
PROMPT_REACT = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""
search = DuckDuckGoSearchRun()
tools = [search, calculator]
tools_retriever = InMemoryVectorStore.from_documents([Document(tool.description, metadata={"name": tool.name}) for tool in tools], EMBEDDING_MODEL).as_retriever()
config = {"configurable": {"thread_id": "1"}}



#
# hàm đầu vào chính của đoạn mã ví dụ này.  
#
# Input:
# prompt: query mới nhất từ ​​người dùng.
#   history: Lịch sử hội thoại giữa người dùng và mô hình, dưới dạng một list.
#       mỗi phần tử trong danh sách có dạng: 
#           {"user": "query của người dùng", "bot": "respond của mô hình"}.
#       hội thoại mới nhất sẽ nằm ở cuối danh sách. Không bao gồm câu hỏi mới nhất. 
#   list_of_plugin_info: Danh sách các plugin có thể sử dụng, được lưu trong một list.
#       ví dụ list_of_plugin_info = [plugin_info_0, plugin_info_1, plugin_info_2]，
#       trong đó plugin_info_0, plugin_info_1, plugin_info_2 là thông tin chi tiết của 
#           từng plugin, đã được đề cập trước đó trong tài liệu này.
#
# output:
#   câu trả lời của mô hình cho câu hỏi mới nhất của người dùng. 
#



def llm_with_plugin(prompt: str, history, list_of_plugin_info=()):
	chat_history = [(x["user"], x["bot"]) for x in history] + [(prompt, "")]
	# Văn bản ban đầu cần để mô hình tiếp tục sinh nội dung
	planning_prompt = build_input_text(chat_history, list_of_plugin_info)
	text = ""
	while True:
		output = text_completion(planning_prompt + text, stop_words=["Observation:", "Observation:\n"])
		action, action_input, output = parse_latest_plugin_call(output)
		if action: # Cần phải gọi plug-in
			# action và action_input lần lượt là mã của plugin cần gọi và tham số đầu vào
			# observation là kết quả trả về từ plugin, dưới dạng chuỗi
			observation = call_plugin(action, action_input)
			output += f"\nObservation: {observation}\nThought:"
			text += output
		else:  # Quá trình sinh nội dung kết thúc và không cần gọi plugin nữa
			text += output
			break
	new_history = []
	new_history.extend(history)
	new_history.append({'user': prompt, 'bot': text})
	return text, new_history



def build_input_text(list_of_plugin_info) -> str:
	# Thông tin chi tiết của các plugin có thể sử dụng
	tools_text = []
	for plugin_info in list_of_plugin_info:
		tool = TOOL_DESC.format(
			name_for_model=plugin_info["name_for_model"],
			name_for_human=plugin_info["name_for_human"],
			description_for_model=plugin_info["description_for_model"],
			parameters=json.dumps(plugin_info["parameters"], ensure_ascii=False)
		)
		if plugin_info.get("args_format", "json") == "json":
			tool += " Format the arguments as a JSON object."
		elif plugin_info["args_format"] == "code":
			tool += " Enclose the code within triple backticks (`) at the beginning and end of the code."
		else:
			raise NotImplementedError
		tools_text.append(tool)
	tools_text = "\n\n".join(tools_text)



def text_completion(input_text: str, stop_words) -> str:  # 作为一个文本续写模型来使用
	im_end = "<|im_end|>"
	if im_end not in stop_words:
		stop_words = stop_words + [im_end]
	stop_words_ids = [tokenizer.encode(w) for w in stop_words]
	#stop_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in stop_words]
	#stop_words_ids = [token_id for sublist in stop_words_ids for token_id in sublist]
	stopping_criteria = StoppingCriteriaList([SequenceStoppingCriteria(stop_words_ids)])

	# TODO: 增加流式输出的样例实现
	input_ids = torch.tensor([tokenizer.encode(input_text)]).to(model.device)
	output = model.llm.generate(input_ids, stopping_criteria=stopping_criteria,max_length=4096,do_sample=False)
	output = output.tolist()[0]
	output = tokenizer.decode(output, errors="ignore")
	assert output.startswith(input_text)
	output = output[len(input_text) :].replace('<|endoftext|>', '').replace(im_end, '')

	for stop_str in stop_words:
		idx = output.find(stop_str)
		if idx != -1:
			output = output[: idx + len(stop_str)]
	return output  # 续写 input_text 的结果，不包含 input_text 的内容



def parse_latest_plugin_call(text):
	plugin_name, plugin_args = '', ''
	i = text.rfind('\nAction:')
	j = text.rfind('\nAction Input:')
	k = text.rfind('\nObservation:')
	if 0 <= i < j:  # If the text has `Action` and `Action input`,
		if k < j:  # but does not contain `Observation`,
			# then it is likely that `Observation` is ommited by the LLM,
			# because the output text may have discarded the stop word.
			text = text.rstrip() + '\nObservation:'  # Add it back.
		k = text.rfind('\nObservation:')
		plugin_name = text[i + len('\nAction:') : j].strip()
		plugin_args = text[j + len('\nAction Input:') : k].strip()
		text = text[:k]
	return plugin_name, plugin_args, text



#
# 输入：
#   plugin_name: 需要调用的插件代号，对应 name_for_model。
#   plugin_args：插件的输入参数，是一个 dict，dict 的 key、value 分别为参数名、参数值。
# 输出：
#   插件的返回结果，需要是字符串。
#   即使原本是 JSON 输出，也请 json.dumps(..., ensure_ascii=False) 成字符串。
#
def call_plugin(plugin_name: str, plugin_args: str) -> str:
	#
	# 请开发者自行完善这部分内容。这里的参考实现仅是 demo 用途，非生产用途。
	#
	if plugin_name == 'image_gen_prompt':
		# 使用 SerpAPI 需要在这里填入您的 SERPAPI_API_KEY！
		try:
			image_path = json5.loads(plugin_args)["image_path"]
			if image_path.startswith('http'):
				headers = {
			'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
			'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
			'Accept-Language': 'en-US,en;q=0.5',
			'Accept-Encoding': 'gzip, deflate, br',
			'Connection': 'keep-alive',
			'Upgrade-Insecure-Requests': '1'
		}
				yzmdata = requests.get(image_path,headers=headers)
				tempIm = BytesIO(yzmdata.content)
				image1 = Image.open(tempIm).convert('RGB')
				image1.save(img_save_path)
				image1 = Image.open(img_save_path).convert('RGB')
			else:
				image1 = Image.open(image_path).convert('RGB')
		except:
			image_path=input("请输入图片地址或网址：")
			if image_path.startswith('http'):
				headers = {
			'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
			'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
			'Accept-Language': 'en-US,en;q=0.5',
			'Accept-Encoding': 'gzip, deflate, br',
			'Connection': 'keep-alive',
			'Upgrade-Insecure-Requests': '1'
		}
				yzmdata = requests.get(image_path,headers=headers)
				tempIm = BytesIO(yzmdata.content)
				image1 = Image.open(tempIm).convert('RGB')
				image1.save(img_save_path)
				image1 = Image.open(img_save_path).convert('RGB')
			else:
				image1 = Image.open(image_path).convert('RGB')
		question1 = 'Please describe all the details in this picture in detail?'
		msgs = [
			{'role': 'user', 'content': question1},
		]

		res = model.chat(
			image=image1,
			msgs=msgs,
			tokenizer=tokenizer
		)
		return res
	elif plugin_name == 'image_gen':
		import urllib.parse
		prompt = json5.loads(plugin_args)["prompt"]
		prompt = urllib.parse.quote(prompt)
		return json.dumps({'image_url': f'https://image.pollinations.ai/prompt/{prompt}'}, ensure_ascii=False)
	elif plugin_name == 'Modify_text':
		import urllib.parse
		prompt_input = json5.loads(plugin_args)["describe_before"]
		Modification_request = json5.loads(plugin_args)["Modification_request"]
		input_prompt = "请将以下的prompt:{}按照以下要求修改:{}.修改后的prompt:".format(prompt_input,Modification_request)
		im_start = '<|im_start|>'
		im_end = '<|im_end|>'
		prompt = f'{im_start}system\nYou are a helpful assistant.{im_end}'+f"\n{im_start}user\n{input_prompt}{im_end}"
		input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
		output = model.llm.generate(input_ids, max_length=4096)
		output = output.tolist()[0]
		output = tokenizer.decode(output, errors="ignore")
		return output
	else:
		raise NotImplementedError



# 定义自定义的 StoppingCriteria 类 Dìngyì zì dìngyì de Stopping_Criteria lèi Định nghĩa lớp StoppingCriteria tùy chỉnh
class SequenceStoppingCriteria(StoppingCriteria):
	def __init__(self, sequence_ids):
		self.sequence_ids = sequence_ids
		self.current_sequence = []
	def check_sequences(self, current_tokens, sequences):
		"""
		检查当前生成的tokens是否包含了特定的连续数字序列。Jiǎnchá dāngqián shēngchéng de tokens shìfǒu bāohánle tèdìng de liánxù shùzì xùliè. Kiểm tra xem các mã thông báo hiện được tạo có chứa một chuỗi các chữ số liên tiếp cụ thể hay không.

		:param current_tokens: 当前生成的 tokens 列表. Dāngqián shēngchéng de tokens lièbiǎo. Danh sách các tokens hiện đang được tạo.
		:param sequences: 包含多个连续数字序列的列表. Bāohán duō gè liánxù shùzì xùliè dì lièbiǎo. Một danh sách chứa nhiều chuỗi số liên tiếp.
		:return: 如果 current_tokens 中出现了任何序列，则返回 True; 否则返回 False. Rúguǒ current_token zhòng chūxiànle rènhé xùliè, zé fǎnhuí True; fǒuzé fǎnhuí False. Trả về True nếu bất kỳ chuỗi nào xuất hiện trong current_token; nếu không thì trả về False.
		"""
		for i in range(len(current_tokens) - max(map(len, sequences)) + 1):
			for seq in sequences:
				if current_tokens[i:i+len(seq)] == seq:
					return True
		return False
	def __call__(self, input_ids, scores, **kwargs):
		# 获取当前生成的 tokens Nhận các tokens hiện tại đang được tạo.
		current_tokens = [input_ids[-1][-1]]

		# 检查连续出现的 tokens 是否匹配停止序列 Jiǎnchá liánxù chūxiàn de tokens shìfǒu pǐpèi tíngzhǐ xùliè Kiểm tra xem các mã thông báo liên tiếp có khớp với chuỗi dừng không
		self.current_sequence.extend(current_tokens)

		# 检查当前生成的 tokens 是否包含了特定的连续数字序列 Jiǎnchá dāngqián shēngchéng de tokens shìfǒu bāohánle tèdìng de liánxù shùzì xùliè Kiểm tra xem các mã thông báo hiện được tạo có chứa một chuỗi số liên tiếp cụ thể hay không
		if self.check_sequences(self.current_sequence, self.sequence_ids):
			return True  # 停止生成 Tíngzhǐ shēngchéng Dừng tạo

		return False

def token_counter(messages):
	"""Đếm số lượng token từ danh sách tin nhắn."""
	text = " ".join([msg.content for msg in messages])
	return len(tokenizer.encode(text)) 

def select_tools(state: State) -> State:
	query = state["messages"][-1].content
	tool_docs = tools_retriever.invoke(query)
	return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}

def reflect(state: State) -> State:
	class_map = {
		AIMessage: HumanMessage, 
		HumanMessage: AIMessage, 
		ToolMessage: HumanMessage 
	}
	translated = [reflection_prompt, state["messages"][0]] + [
		class_map[msg.__class__](content=msg.content) 
		for msg in state["messages"][1:]
	]
	answer = model.invoke(translated)
	return {"messages": [HumanMessage(content=answer.content)]}

def should_continue(state: State):
	if len(state["messages"]) > 6:
		return END
	else:
		return "reflect"

def chatbot(state: State) -> State:
	selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
	answer = model.bind_tools(selected_tools).invoke([generate_prompt] + state["messages"])
	return {"messages": [answer]}

def main():
	"""Thực thi chương trình."""
	builder = StateGraph(State)

	builder.add_node("select_tools", select_tools)
	builder.add_node("chatbot", chatbot)
	builder.add_node("tools", ToolNode(tools))
	builder.add_node("reflect", reflect)

	builder.add_edge(START, "select_tools")
	builder.add_edge("select_tools", "chatbot")
	builder.add_conditional_edges("chatbot", tools_condition)
	builder.add_edge("tools", "chatbot")
	builder.add_conditional_edges("chatbot", should_continue)
	builder.add_edge("reflect", "chatbot")
	
	graph = builder.compile(checkpointer=MemorySaver())

	user_input = {
		"messages": [HumanMessage("""What is Large Language Model?""")]
	}
	for chunk in graph.stream(user_input, config):
		print(chunk)

if __name__ == "__main__":
	fire.Fire(main)
