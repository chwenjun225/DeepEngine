import fire 
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.checkpoint.memory import MemorySaver 
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import (create_react_agent, ToolNode)

from tools_use import (add, multiply, get_weather, get_coolest_cities, recommend_maintenance_strategy, diagnose_fault_of_machine, remaining_useful_life_prediction)
from state import State
from prompts import prompt
from langchain.agents import AgentExecutor
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

