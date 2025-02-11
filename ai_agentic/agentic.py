import fire 
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver 
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent 

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from tools_use import (
	rag, planning, search, add, multiply, check_weather, 
	recommend_maintenance_strategy, 
	diagnose_fault_of_machine, 
	remaining_useful_life_prediction, 
)

if "Khai báo cấu hình":
	checkpointer=MemorySaver()
	store = InMemoryStore()
	persist_directory = "/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
	collection_name = "foxconn_ai_research"
	embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
	planning_prompt = ChatPromptTemplate.from_messages([("human", """Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise. 
Question: {question}\nContext: {context}\nAnswer:""")]) 
	embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
	vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model, collection_name=collection_name)
	system_prompt = """You are a friendly and helpful assistant. 
Your job is to answer human questions with care and detail. 
Keep your answers short and concise when possible."""

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
	openai_api_base = f"""http://{host}:{port}/{version}/"""
	return ChatOpenAI(
		model=model_name, openai_api_base=openai_api_base, 
		openai_api_key=openai_api_key, temperature=temperature
	)

def main():
	model = get_llm(
		port=2026, host="127.0.0.1", version="v1", 
		temperature=0, openai_api_key="chwenjun225",
		model_name="/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct"
	)
	tools = [check_weather]
	graph = create_react_agent(model=model, tools=tools)
	inputs = {"messages": [("user", "What is the weather in Hanoi?")]}
	for s in graph.stream(inputs, stream_mode="values"):
		message = s["messages"][-1]
		if isinstance(message, tuple):
			print(message)
		else:
			message.pretty_print()

if __name__ == "__main__":
	fire.Fire(main)


























































# Archive
# 	tools = [
# 			Tool(name="remaining_useful_life_prediction", func=remaining_useful_life_prediction, description="Predict the Remaining Useful Life (RUL) of a component based on the provided sensor data."),
# 			Tool(name="diagnose_fault_of_machine", func=diagnose_fault_of_machine, description="Identify the fault of a machine based on the provided sensor data."),
# 			Tool(name="recommend_maintenance_strategy", func=recommend_maintenance_strategy, description="Suggest the best maintenance strategy to minimize downtime and costs.")
# 		]
# 	tool_names = ", ".join([tool.name for tool in tools])
# 	tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
# 	prompt_template = PromptTemplate(
# 		input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
# 		template="""
# You are an AI assistant specializing in Prognostics and Health Management (PHM) for industrial systems.
# Your responsibilities include diagnosing faults, predicting Remaining Useful Life (RUL), and recommending maintenance strategies.
# At the end of the answer you should summarize the result and provide recommendations.
# **TOOLS:**
# {tools}

# **Available Tool Names (use exactly as written):**
# {tool_descriptions}

# **FORMAT:**
# Thought: [Your reasoning]
# Action: [Tool Name]
# Action Input: [Input to the Tool as JSON]
# Observation: [Result]
# Final Answer: [Answer to the User]

# **Examples:**
# - To predict the Remaining Useful Life (RUL) of a component:
# 	Think: I will predict the RUL of this component based on the provided sensor data.
# 	Action: remaining_useful_life_prediction
# 	Action Input: 
# 	{
# 		"sensor_data": {"temperature": [75, 78, 79], "vibration": [0.12, 0.15, 0.18], "pressure": [101, 99, 98]}, 
# 		"operating_conditions": {"load": 85, "speed": 1500}
# 	}
# 	Observation: 
# 	{
# 		"predicted_rul": 150, 
# 		"confidence": 0.85, 
# 		"recommendations": "Reduce operating load to extend lifespan."
# 	}
# 	Final Answer: I have calculated that the RUL of this component is 150 hours. I recommend reducing the operating load to extend its lifespan.

# - To diagnose a fault in a rotating machine:
# 	Think: I will identify the fault based on vibration and temperature data.
# 	Action: diagnose_fault_of_machine
# 	Action Input: 
# 	{
# 		"sensor_data": {"vibration": [0.20, 0.35, 0.50], "temperature": [90, 92, 94]}
# 	}
# 	Observation: 
# 	{
# 		"fault": "Overheating", 
# 		"recommendation": "Reduce workload and inspect the cooling system."
# 	}
# 	Final Answer: Based on the provided data, the system is experiencing overheating. I recommend reducing the workload and checking the cooling system for potential issues.

# - To recommend a maintenance strategy:
# 	Think: I will suggest the best maintenance strategy to minimize downtime and costs.
# 	Action: recommend_maintenance_strategy
# 	Action Input: 
# 	{
# 		"historical_data": {"failures": 5, "downtime_cost": 3000, "maintenance_cost": 500}, 
# 		"failure_probability": 0.03
# 	}
# 	Observation: 
# 	{
# 		"strategy": "Preventive Maintenance",
# 		"justification": "Failure probability is 0.03, making preventive maintenance the most cost-effective solution."
# 	}
# 	Final Answer: Based on the analysis, I recommend implementing a Preventive Maintenance strategy to minimize downtime and costs.

# **Begin!**

# Question: {input}
# {agent_scratchpad}"""
# 	)








































# Archive 
# model = get_llm(
# 	port=2026, 
# 	host="127.0.0.1", 
# 	version="v1",
# 	temperature=0, 
# 	openai_api_key="chwenjun225",
# 	model_name="deepseek_r1_foxconn_ai_research"
# )
# agent_exe = create_react_agent(
# 	version="v1", 
# 	model=model, 
# 	tools=tools, 
# 	store=store, 
# 	prompt=system_prompt, 
# 	checkpointer=checkpointer, 
# )
# final_state = agent_exe.invoke(
#     input={"messages": [{"role": "user", "content": "what is the weather in vn?"}]}, config={"configurable": {"thread_id": 42}}, 
# 	output_keys=["messages"], 
# 	debug=True
# )
# print(final_state["messages"][-1].content)










# with open("chroma_logs.txt", "a") as f:
	# sys.stdout = f  # Chuyển tất cả print() vào file
	# llm = get_llm(
	# 	port=2026, host="127.0.0.1", openai_api_key="chwenjun225", 
	# 	model_name="1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct", 
	# 	temperature=0
	# )
	# search = TavilySearchResults(max_results=2)
	# tools = [search]
	# model_with_tools = llm.bind_tools(tools)

	# # user_input = input(">>> 👨 User: ") 
	# user_input = "What's the difference between revenue year of 2023 and 2024?"
	# # user_input = "Hello"
	# # 🔍 Truy vấn RAG từ ChromaDB
	# rag_output = rag(query=user_input, num_retrieved_docs=1)
	# # print(f">>> <rag_output>{rag_output}</rag_output>")

	# # 🧠 Lập kế hoạch phản hồi từ Planning Module
	# planning_module_output = planning(user_input, rag_output)
	# # print(f">>> <planning_module>{planning_module_output}</planning_module>")

	# # 🤖 Gửi vào LLM để nhận phản hồi
	# response_template = ChatPromptTemplate.from_messages([
	# 	("system", "You are a friendly and helpful assistant. Your job is to answer human questions with care and detail. Keep your answers short and concise when possible."),
	# 	("user", "{input}")
	# ])
	# formatted_response_template = response_template.invoke({"input": planning_module_output})
	# response = llm.invoke(formatted_response_template)
	# assistant_response = response.content
	# print(f">>> 🤖 Assistant:\n{assistant_response}")

	# # 📝 Lưu lịch sử hội thoại vào ChromaDB
	# save_chat_history_to_chromadb(user_input, assistant_response)
	# sys.stdout = sys.__stdout__  # Reset lại stdout về mặc định
