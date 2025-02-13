import fire 
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.checkpoint.memory import MemorySaver 
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import (create_react_agent, ToolNode)

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
		model=model_name, 
		openai_api_base=openai_api_base, 
		openai_api_key=openai_api_key, 
		temperature=temperature
	)

def rag(query, vector_db, num_retrieved_docs=3):
	"""Truy vấn RAG từ ChromaDB."""
	retriever_docs = vector_db.similarity_search(query, k=num_retrieved_docs)
	retrieved_texts = "\n".join([retriever_doc.page_content for retriever_doc in retriever_docs])
	return retrieved_texts

def planning(prompt_user, prompt_planning, rag_output):
	"""Xử lý thông tin từ RAG để tạo prompt phù hợp cho LLM."""
	planning_prompt = prompt_planning.invoke({"question": prompt_user, "context": rag_output})
	return planning_prompt

def print_stream(stream):
	"""A utility to pretty print the stream."""
	for s in stream:
		message = s["messages"][-1]
		if isinstance(message, tuple):
			print(message)
		else:
			message.pretty_print()

if True:
	checkpointer=MemorySaver()
	store = InMemoryStore()
	persist_directory = "/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
	collection_name = "foxconn_ai_research"
	embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
	prompt = """You are an intelligent, reasoning-driven AI agent. Your role is to **observe**, **reason**, and **act** based on the given input. You have access to external tools that can help you retrieve factual data, perform calculations, or execute specific functions. Your responses must be **accurate, concise, and well-structured**.

## **Reasoning & Thought Process**
- Always break down complex problems logically before responding.
- Use **step-by-step reasoning** to ensure the accuracy of your answer.
- If a question is unclear, ask for clarification rather than making assumptions.

## **Using Tools**
- When needed, invoke the most **relevant tool** to assist in answering the query.
- Clearly state **why** you are using a tool before invoking it.
- After receiving tool output, analyze and summarize the result in a **clear and useful manner**.
- If no tool is appropriate, respond with reasoning based on available knowledge.

## **Response Formatting**
- Keep your answers **structured and readable**.
- Use **bullet points, numbered lists, or short paragraphs** when necessary.
- If responding with **technical information**, provide **concise explanations**.
- If subjective, clarify that it is an **opinion or general perspective**.

## **Rules & Constraints**
- **DO NOT** hallucinate information—only use tools or knowledge you are confident in.
- **DO NOT** assume data if a tool fails—indicate the issue and suggest alternatives.
- **DO NOT** fabricate sources or tool outputs.
- **DO NOT** engage in unethical, biased, or harmful discussions.

## **Example Usage**
### **Query:** "What is the current weather in New York?"
- **Step 1:** Recognize that real-time data is required.
- **Step 2:** Use `get_weather` tool to fetch the latest weather.
- **Step 3:** Summarize the response concisely.

**Example Response:**
_"I will check the latest weather for New York using the `get_weather` tool."_

**Tool Call:**
```json
{
	"tool": "get_weather",
	"args": {
		"location": "New York"
	}
}

"""
	embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
	vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model, collection_name=collection_name)
	tools = [
		add, multiply, 
		get_weather,
		get_coolest_cities, 
		recommend_maintenance_strategy, 
		diagnose_fault_of_machine, 
		remaining_useful_life_prediction
	]
	tool_node = ToolNode(tools)
	tools_by_name = {tool.name: tool for tool in tools}
	checkpointer = MemorySaver()
	model = get_llm(
		port=2026, 
		host="127.0.0.1", 
		version="v1", 
		temperature=0, 
		openai_api_key="chwenjun225",
		model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct"
	)

def main():
	config = {
		"configurable": {"thread_id": "42"}, 
		"recursion_limit": 5,
		"memory_saving": True
	}
	graph = create_react_agent(
		model=model, 
		tools=tool_node, 
		checkpointer=checkpointer, 
		prompt=prompt, 
		response_format=State
	)
	messages = []
	while True:
		user_input = input(">>> 👨 User: ")
		if user_input.lower() == "exit":
			print("\n >>>👋 Bye! See you again!\n")
			break
		messages.append(("user", user_input))
		inputs = {"messages": messages}
		stream = graph.stream(inputs, config=config, stream_mode="values")
		print_stream(stream)
		for response in stream:
			assistant_response = response["messages"][-1]
			messages.append(("assistant", assistant_response))

if __name__ == "__main__":
	fire.Fire(main)







	# inputs = {"messages": [("user", "What's the weather in NYC?")]}



	# response = graph.invoke(inputs)
	# print(response["structured_response"])


	# # Before HumanInTheLoop
	# snapshot = graph.get_state(config)
	# print("Next step: ", snapshot.next)
	# print_stream(graph.stream(None, config, stream_mode="values"))

	# # HumanInTheLoop
	# state = graph.get_state(config)
	# last_message = state.values["messages"][-1]
	# last_message.tool_calls[0]["args"] = {"location": "San Francisco"}

	# graph.update_state(config, {"messages": [last_message]})














# """
# You are an AI assistant specializing in Prognostics and Health Management (PHM) for industrial systems. Your responsibilities include diagnosing faults, predicting Remaining Useful Life (RUL), and recommending maintenance strategies.\n\nAt the end of the answer you should summarize the result and provide recommendations.
# **TOOLS:**
# {tools}

# **Available Tool Names (use exactly as written):**
# {tool_names}

# **FORMAT:**
# Thought: [Your reasoning]
# Action: [Tool Name]
# Action Input: [Input to the Tool as JSON]
# Observation: [Result]
# Final Answer: [Answer to the User]

# **Examples:**
# - To predict the Remaining Useful Life (RUL) of a component:
# 	Thought: I will predict the RUL of this component based on the provided sensor data.
# 	Action: remaining_useful_life_prediction
# 	Action Input: {{
# 		"sensor_data": {{"temperature": [75, 78, 79], "vibration": [0.12, 0.15, 0.18], "pressure": [101, 99, 98]}}, 
# 		"operating_conditions": {{"load": 85, "speed": 1500}}
# 		}}
# 	Observation: {{
# 		"predicted_rul": 150, 
# 		"confidence": 0.85, 
# 		"recommendations": "Reduce operating load to extend lifespan."
# 		}}
# 	Final Answer: I have calculated that the RUL of this component is 150 hours. I recommend reducing the operating load to extend its lifespan.

# - To diagnose a fault in a rotating machine:
# 	Thought: I will identify the fault based on vibration and temperature data.
# 	Action: diagnose_fault_of_machine
# 	Action Input: {{
# 		"sensor_data": {{"vibration": [0.20, 0.35, 0.50], "temperature": [90, 92, 94]}}
# 		}}
# 	Observation: {{
# 		"fault": "Overheating", 
# 		"recommendation": "Reduce workload and inspect the cooling system."
# 		}}
# 	Final Answer: Based on the provided data, the system is experiencing overheating. I recommend reducing the workload and checking the cooling system for potential issues.

# - To recommend a maintenance strategy:
# 	Thought: I will suggest the best maintenance strategy to minimize downtime and costs.
# 	Action: recommend_maintenance_strategy
# 	Action Input: {{
# 		"historical_data": {{"failures": 5, "downtime_cost": 3000, "maintenance_cost": 500}}, 
# 		"failure_probability": 0.03
# 		}}
# 	Observation: {{
# 		"strategy": "Preventive Maintenance",
# 		"justification": "Failure probability is 0.03, making preventive maintenance the most cost-effective solution."
# 		}}
# 	Final Answer: Based on the analysis, I recommend implementing a Preventive Maintenance strategy to minimize downtime and costs.

# **Begin!**

# Question: {input}
# {agent_scratchpad}
# """









# (Foxer_env) (base) chwenjun225@chwenjun225:~/Projects/Foxer/ai_agentic$ python agentic.py 
# ================================ Human Message =================================

# who built you?
# ================================== Ai Message ==================================
# Name: foxconn_ai_research
# Tool Calls:
#   get_weather (chatcmpl-tool-c9809fb7064c484088e2ef99972a82d9)
#  Call ID: chatcmpl-tool-c9809fb7064c484088e2ef99972a82d9
#   Args:
#     city: nyc
#     properties: {'enum': '["nyc", "sf"]', 'type': 'string'}
# ================================= Tool Message =================================
# Name: get_weather

# It might be cloudy in nyc
# ================================== Ai Message ==================================
# Name: foxconn_ai_research

# <|python_tag|>{"name": "get_weather", "parameters": {"city": "nyc", "properties": {"enum": "[\"nyc\", \"sf\"]", "type": "string"}}}

# "It might be cloudy in nyc"
# (Foxer_env) (base) chwenjun225@chwenjun225:~/Projects/Foxer/ai_agentic$ python agentic.py 
# ================================ Human Message =================================

# what is the weather in sf?
# ================================== Ai Message ==================================
# Name: foxconn_ai_research
# Tool Calls:
#   get_weather (chatcmpl-tool-bd21cee544374b1e8b865852b4503782)
#  Call ID: chatcmpl-tool-bd21cee544374b1e8b865852b4503782
#   Args:
#     properties: {'city': 'sf', 'type': 'string'}
# ================================= Tool Message =================================
# Name: get_weather

# Error: 1 validation error for get_weather
# city
#   Field required [type=missing, input_value={'properties': {'city': 'sf', 'type': 'string'}}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.10/v/missing
#  Please fix your mistakes.
# ================================== Ai Message ==================================
# Name: foxconn_ai_research

# <|python_tag|>{"name": "get_weather", "parameters": {"properties": {"city": "sf", "type": "string"}}}

# {
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Use this to get weather information.",
#         "parameters": {
#             "properties": {
#                 "city": "sf",
#                 "type": "string"
#             },
#             "required": [
#                 "city"
#             ],
#             "type": "object"
#         }
#     }
# }


































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




