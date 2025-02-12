import fire 
from datetime import datetime
from typing_extensions import Literal

from langgraph.checkpoint.memory import MemorySaver 
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def save_chat_history(user_input, assistant_response):
	"""Lưu lịch sử hội thoại vào ChromaDB."""
	timestamp =  datetime.now().strftime("%Y-%m-%d_%H:%M:%S-%f")
	text_to_store = f"""[{timestamp}] User: {user_input}\n[{timestamp}] Assistant: {assistant_response}"""
	vector_db.add_texts(
		texts=[text_to_store],
		metadatas=[{"source": "chat_history"}]
	)

def get_llm(
		port=2026, host="127.0.0.1", version="v1", 
		temperature=0, openai_api_key="chwenjun225",
		model_name="/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct"
	):
	"""Khởi tạo mô hình ngôn ngữ lớn DeepSeek-R1 từ LlamaCpp-Server."""
	openai_api_base = f"""http://{host}:{port}/{version}""" 
	return ChatOpenAI(
		model=model_name, openai_api_base=openai_api_base, 
		openai_api_key=openai_api_key, temperature=temperature
	)

def print_stream(stream):
	for s in stream:
		message = s["messages"][-1]
		if isinstance(message, tuple):
			print(message)
		else:
			message.pretty_print()

@tool
def get_weather(city: Literal["nyc", "sf"]):
	"""Use this to get weather information."""
	if city == "nyc":
		return "It might be cloudy in nyc"
	elif city == "sf":
		return "It's always sunny in sf"
	else:
		raise AssertionError("Unknown city")

if True:
	checkpointer=MemorySaver()
	store = InMemoryStore()
	persist_directory = "/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
	collection_name = "foxconn_ai_research"
	embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
	prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
	embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
	vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model, collection_name=collection_name)

	tools = [get_weather]
	memory = MemorySaver()
	model = get_llm(
		port=2026, 
		host="127.0.0.1", 
		version="v1", 
		temperature=0, 
		openai_api_key="chwenjun225",
		model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct"
	)
	graph = create_react_agent(
		name="foxconn_ai_research", 
		model=model, 
		tools=tools, 
		checkpointer=memory, 
		prompt=prompt
	)
	config = {"configurable": {"thread_id": "42"}}

def main():
	inputs = {"messages": [("user", "what is the weather in SF, CA?")]}
	print_stream(graph.stream(inputs, config=config, stream_mode="values"))
	snapshot = graph.get_state(config)
	print(">>> Next step: ", snapshot.next)
	print_stream(graph.stream(None, config, stream_mode="values"))

if __name__ == "__main__":
	fire.Fire(main)












































# if False:
# 	class WeatherInput(BaseModel):
# 		location: str = Field(description="The city and state, e.g. San Francisco, CA")
# 		unit: str = Field(enum=["celsius", "fahrenheit"])

# 	class Step(BaseModel):
# 		explanation: str
# 		output: str

# 	class CoT_Response(BaseModel):
# 		steps: list[Step]
# 		final_answer: str



# 		)

# 	def print_stream(graph, inputs, config):
# 		for s in graph.stream(inputs, config, stream_mode="values"):
# 			message = s["messages"][-1]
# 			if isinstance(message, tuple):
# 				print(message)
# 			else:
# 				message.pretty_print()

# 	@tool("get_current_weather", args_schema=WeatherInput)
# 	def get_current_weather(location: str, unit: str):
# 		"""Get the current weather in a given location."""
# 		return f"Now the weather in {location} is 22 {unit}"

# 	def main():
# 		tools = [get_current_weather]
# 		tool_node = ToolNode(tools)
# 		llm = get_llm(
# 			port=2026, host="127.0.0.1", version="v1", 
# 			temperature=0,
# 			openai_api_key="chwenjun225", 
# 			model_name="/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct"
# 		)
# 		llm_with_tools = llm.bind_tools(
# 			tools=[get_current_weather],
# 			tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
# 		)
# 		ai_msg = llm_with_tools.invoke(
# 			"what is the weather like in Ho Chi Minh city in celsius",
# 		)
# 		print(ai_msg)
# 		graph = create_react_agent(
# 			name="deepseek_r1_foxconn_ai_research",
# 			model=llm, 
# 			tools=tool_node, 
# 			prompt=prompt, 
# 			# response_format=CoT_Response, 
# 			checkpointer=checkpointer, 
# 			store=store, 
# 			debug=False
# 		)
# 		config = {"configurable": {"thread_id": "thread-1"}}
# 		inputs = {"messages": [("user", "What's the weather in hn?")]}
# 		print_stream(graph=graph, inputs=inputs, config=config)

# 	if __name__ == "__main__":
# 		fire.Fire(main)



























# if False:
# 	class WeatherInput(BaseModel):
# 		location: str = Field(description="The city and state, e.g. San Francisco, CA")
# 		unit: str = Field(enum=["celsius", "fahrenheit"])

# 	@tool("get_current_weather", args_schema=WeatherInput)
# 	def get_current_weather(location: str, unit: str):
# 		"""Get the current weather in a given location."""
# 		return f"Now the weather in {location} is 22 {unit}"
	
# 	tools = [get_current_weather]
# 	tool_node = ToolNode(tools)
# 	llm = get_llm(
# 		port=2026, host="127.0.0.1", version="v1", 
# 		temperature=0,
# 		openai_api_key="chwenjun225", 
# 		model_name="/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct"
# 	)
# 	llm_with_tools = llm.bind_tools(
# 		tools=[get_current_weather],
# 		tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
# 	)
# 	ai_msg = llm_with_tools.invoke(
# 		"what is the weather like in Ho Chi Minh city in celsius",
# 	)
# 	print(ai_msg.tool_calls)




