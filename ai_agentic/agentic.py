from pprint import pprint
import fire 
from datetime import datetime
from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.messages import (
	AIMessage, 
	SystemMessage, 
	HumanMessage, 
	trim_messages, 
	filter_messages
)
from langgraph.graph import (StateGraph, START, END)
from langgraph.checkpoint.memory import MemorySaver 

from tools_use import (
	add, multiply, get_weather, get_coolest_cities, 
	recommend_maintenance_strategy, diagnose_fault_of_machine, 
	remaining_useful_life_prediction
)
from state import State
from prompts import prompt

def token_counter(messages):
    """Đếm số lượng token từ danh sách tin nhắn"""
    text = " ".join([msg.content for msg in messages])
    return len(tokenizer.encode(text)) 

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
		model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct"
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

if True:
	tokenizer = AutoTokenizer.from_pretrained("/home/chwenjun225/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B")
	builder = StateGraph(State)
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
	checkpointer = MemorySaver()
	model = get_llm(
		port=2026, 
		host="127.0.0.1", 
		version="v1", 
		temperature=0, 
		openai_api_key="chwenjun225",
		model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct"
	)
	trimmer = trim_messages(
		max_tokens=65,
		strategy="last",
		token_counter=token_counter,
		include_system=True,
		allow_partial=False,
		start_on="human",
	)

def chatbot(state: State):
	answer = model.invoke(state["messages"])
	return {"messages": [answer]}

def main():
	builder.add_node("chatbot", chatbot)
	builder.add_edge(START, 'chatbot')
	builder.add_edge('chatbot', END)
	messages = [
		SystemMessage("you are a good assistant", id="1"),
		HumanMessage("example input", id="2", name="example_user"),
		AIMessage("example output", id="3", name="example_assistant"),
		HumanMessage("real input", id="4", name="bob"),
		AIMessage("real output", id="5", name="alice"),
	]
	messages = [
		SystemMessage("you are a good assistant", id="1"),
		HumanMessage("example input", id="2", name="example_user"),
		AIMessage("example output", id="3", name="example_assistant"),
		HumanMessage("real input", id="4", name="bob"),
		AIMessage("real output", id="5", name="alice"),
	]
	pprint(filter_messages(messages, include_types="human"))
	# TODO: https://learning.oreilly.com/library/view/learning-langchain/9781098167271/ch05.html

if __name__ == "__main__":
	fire.Fire(main)

